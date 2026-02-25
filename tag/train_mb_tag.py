#!/usr/bin/env python

"""Train TAG-ALIGNN model on a MatBench fold folder with formatted dataset."""
import os
import torch.distributed as dist
import csv
import sys
import json
import zipfile
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl_prop
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
import time
from jarvis.core.atoms import Atoms
import random
from ase.stress import voigt_6_to_full_3x3_stress

import warnings
warnings.filterwarnings('ignore')

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network - Text-Attributed Graph"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_prop.csv and structure files",
)
parser.add_argument(
    "--id_prop_file",
    default="id_prop",
    help="Filename for the id_prop main file (without .csv extension)",
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--target_key",
    default="total_energy",
    help="Name of the key for graph level data such as total_energy",
)

parser.add_argument(
    "--id_key",
    default="jid",
    help="Name of the key for graph level id such as id",
)

parser.add_argument(
    "--force_key",
    default="forces",
    help="Name of key for gradient level data such as forces, (Natoms x p)",
)

parser.add_argument(
    "--atomwise_key",
    default="forces",
    help="Name of key for atomwise level data: forces, charges (Natoms x p)",
)

parser.add_argument(
    "--stresswise_key",
    default="stresses",
    help="Name of the key for stress (3x3) level data such as forces",
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)

parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)

parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)


def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    id_prop_file="id_prop",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
):
    """Train TAG-ALIGNN for a MatBench fold folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)
    ##### Load dataset file, csv or json #####
    id_prop_json = os.path.join(root_dir, f"{id_prop_file}.json")
    id_prop_json_zip = os.path.join(root_dir, f"{id_prop_file}.json.zip")
    id_prop_csv = os.path.join(root_dir, f"{id_prop_file}.csv")
    id_prop_csv_file = False
    multioutput = False
    if os.path.exists(id_prop_json_zip):
        dat = json.loads(
            zipfile.ZipFile(id_prop_json_zip).read(f"{id_prop_file}.json")
        )
    elif os.path.exists(id_prop_json):
        dat = loadjson(os.path.join(root_dir, f"{id_prop_file}.json"))
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")
    ##### Load config file with config_name arg #####
    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)

    train_grad = False
    train_stress = False
    train_atom = False
    target_atomwise = None
    target_grad = None
    target_stress = None

    n_outputs = []
    dataset = []
    for i in dat:
        if "target" in i[1]:
            continue
    ##### In the case of id_prop.csv, dat is a list of rows
        info = {}
        if id_prop_csv_file:
            # file_name is the POSCAR filename as written by mb/run.py
            # e.g. "mb-jdft2d-0" — no extension appended
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = tmp
            # Structure files are inside root_dir (the fold directory)
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            info["atoms"] = atoms.to_dict()
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]
        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]
        if train_stress:
            if len(i[stresswise_key]) == 6:
                stress = voigt_6_to_full_3x3_stress(i[stresswise_key])
            else:
                stress = i[stresswise_key]
            info["stresses"] = stress
            target_stress = "stresses"
        if "extra_features" in i:
            info["extra_features"] = i["extra_features"]
        dataset.append(info)
    print("len dataset", len(dataset))
    del dat
    lists_length_equal = True
    line_graph = False

    if config.model.alignn_layers > 0:
        line_graph = True

    if multioutput:
        print("multioutput", multioutput)
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]
        print("lists_length_equal", lists_length_equal, len(n_outputs[0]))
        if lists_length_equal:
            config.model.output_features = len(n_outputs[0])
        else:
            raise ValueError("Make sure the outputs are of same size.")
    model = None
    if restart_model_path is not None:
        print("Restarting the model training:", restart_model_path)
        if config.model.name == "alignn_atomwise":
            rest_config = loadjson(
                restart_model_path.replace("current_model.pt", "config.json")
            )
            tmp = ALIGNNAtomWiseConfig(**rest_config["model"])
            print("Rest config", tmp)
            model = ALIGNNAtomWise(tmp)
            print("model", model)
            model.load_state_dict(
                torch.load(restart_model_path, map_location=device)
            )
            model = model.to(device)

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        tag=True,   # TAG: inject atom-level text embeddings into node features
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=True,   # Always keep order: id_prop.csv is ordered train-val-test
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
    )
    t1 = time.time()
    print("rank", rank)
    print("world_size", world_size)
    train_dgl_prop(
        config,
        model=model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        rank=rank,
        world_size=world_size,
    )
    t2 = time.time()
    print("Time taken (s)", t2 - t1)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = 1
    print("world_size", world_size)
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.id_prop_file,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.id_prop_file,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass
