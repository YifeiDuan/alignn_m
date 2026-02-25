"""Run TAG-ALIGNN model on MatBench dataset using official benchmark splits."""
# Ref: https://www.nature.com/articles/s41524-021-00650-1

import glob
import os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY

import argparse

import warnings
warnings.filterwarnings('ignore')

##### Initalize MatbenchBenchmark object that configures prop tasks #####
mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        "matbench_jdft2d",
        # "matbench_dielectric",
        # "matbench_phonons",
        # "matbench_log_gvrh",
        # "matbench_log_kvrh",
        # "matbench_perovskites",
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ],
)

python_executable = sys.executable
script_dir = os.path.dirname(os.path.abspath(__file__))
train_mb_tag_script = os.path.join(script_dir, "train_mb_tag.py")


def train_tasks(
    mb=None, config_template="config_example.json", file_format="poscar"
):
    """Train TAG-ALIGNN on MatBench classification and regression tasks."""
    for task in mb.tasks:
        task.load()
        if task.metadata.task_type == CLF_KEY:
            classification = True
        else:
            classification = False

        # Classification tasks
        if classification:
            for ii, fold in enumerate(task.folds):
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                train_df["is_metal"] = train_df["is_metal"].astype(int)
                test_df["is_metal"] = test_df["is_metal"].astype(int)
                target = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                fold_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_fold_"
                    + str(ii)
                )
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                os.chdir(fold_name)
                f = open("id_prop.csv", "w")
                for jj, j in train_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                # Use a portion of training set as validation set
                val_df = train_df[0 : len(test_df)]
                for jj, j in val_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                config = loadjson(config_template)
                config["n_train"] = n_train
                config["n_val"] = n_val
                config["n_test"] = n_test
                config["keep_data_order"] = True
                config["batch_size"] = 32
                config["epochs"] = 40
                config["classification_threshold"] = 0.01
                fname = "config_fold_" + str(ii) + ".json"
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                cmd = (
                    python_executable + " "
                    + train_mb_tag_script
                    + " --root_dir "
                    + fold_name
                    + " --config_name "
                    + fold_name
                    + "/"
                    + fname
                    + " --file_format="
                    + file_format
                    + " --classification_threshold=0.01"
                    + " --output_dir="
                    + outdir_name
                )
                print(cmd)
                os.system(cmd)

        # Regression tasks
        if not classification:
            for ii, fold in enumerate(task.folds):
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                target = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                fold_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_fold_"
                    + str(ii)
                )
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                os.chdir(fold_name)
                f = open("id_prop.csv", "w")
                for jj, j in train_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                # Use a portion of training set as validation set
                val_df = train_df[0 : len(test_df)]
                for jj, j in val_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                config = loadjson(config_template)
                config["filename"] = fold_name
                config["n_train"] = n_train
                config["n_val"] = n_val
                config["n_test"] = n_test
                config["keep_data_order"] = True
                config["batch_size"] = 32
                config["test_batch_size"] = 32
                config["epochs"] = 500
                fname = "config_fold_" + str(ii) + ".json"
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                cmd = (
                    python_executable + " "
                    + train_mb_tag_script
                    + " --root_dir "
                    + fold_name
                    + " --config_name "
                    + fold_name
                    + "/"
                    + fname
                    + " --file_format="
                    + file_format
                    + " --output_dir="
                    + outdir_name
                )
                print(cmd)
                os.system(cmd)


def compile_results(key="matbench_phonons", regression=True):
    """Compile fold-based results for each task."""
    maes = []
    roc_aucs = []
    results = defaultdict()

    for i in glob.glob(key + "*/prediction_results_test_set.csv"):
        fold = int(i.split("/")[0].split("_")[-1])
        df = pd.read_csv(i)

        target_vals = df.target.values
        pred_vals = df.prediction.values
        if regression:
            mae = mean_absolute_error(target_vals, pred_vals)
            maes.append(mae)
            print("MAE", fold, mae)
        if not regression:
            roc = roc_auc_score(target_vals, pred_vals)
            roc_aucs.append(roc)
            print("ROC", fold, roc)
            pred_vals = [True if i == 1 else False for i in pred_vals]
        results[fold] = pred_vals

    if regression:
        maes = np.array(maes)
        print(key, maes, np.mean(maes), np.std(maes))
    if not regression:
        roc_aucs = np.array(roc_aucs)
        print(key, roc_aucs, np.mean(roc_aucs), np.std(roc_aucs))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAG-ALIGNN MatBench run_mb_tag.py args"
    )
    parser.add_argument(
        "--config_file_name",
        default="config_example_textpca100.json",
        help="Name of the TAG config file (looked up in the tag/ directory). "
             "Must set atom_input_features = cgcnn_dim + text_embedding_dim.",
    )

    args = parser.parse_args(sys.argv[1:])

    ##### Load TAG config file (should have atom_input_features adjusted for text embeddings) #####
    config_file_name = args.config_file_name
    config_template = os.path.abspath(
        os.path.join(script_dir, config_file_name)
    )
    config = loadjson(config_template)

    ##### Run the training loop for all tasks in mb #####
    train_tasks(mb=mb, config_template=config_template, file_format="poscar")
