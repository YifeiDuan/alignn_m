#!/usr/bin/env python

"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN_infer, ALIGNNConfig
import tempfile
import torch
import sys
import re
import numpy as np
import pandas as pd
from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph

from alignn.config import TrainingConfig
from IPython import embed

# Name of the model, figshare link, number of outputs
all_models = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ],
    "jv_optb88vdw_total_energy_alignn": [
        "https://figshare.com/ndownloader/files/31459642",
        1,
    ],
    "jv_optb88vdw_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31459636",
        1,
    ],
    "jv_mbj_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31458694",
        1,
    ],
    "jv_spillage_alignn": [
        "https://figshare.com/ndownloader/files/31458736",
        1,
    ],
    "jv_slme_alignn": ["https://figshare.com/ndownloader/files/31458727", 1],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "jv_shear_modulus_gv_alignn": [
        "https://figshare.com/ndownloader/files/31458724",
        1,
    ],
    "jv_n-Seebeck_alignn": [
        "https://figshare.com/ndownloader/files/31458718",
        1,
    ],
    "jv_n-powerfact_alignn": [
        "https://figshare.com/ndownloader/files/31458712",
        1,
    ],
    "jv_magmom_oszicar_alignn": [
        "https://figshare.com/ndownloader/files/31458685",
        1,
    ],
    "jv_kpoint_length_unit_alignn": [
        "https://figshare.com/ndownloader/files/31458682",
        1,
    ],
    "jv_avg_elec_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458643",
        1,
    ],
    "jv_avg_hole_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458646",
        1,
    ],
    "jv_epsx_alignn": ["https://figshare.com/ndownloader/files/31458667", 1],
    "jv_mepsx_alignn": ["https://figshare.com/ndownloader/files/31458703", 1],
    "jv_max_efg_alignn": [
        "https://figshare.com/ndownloader/files/31458691",
        1,
    ],
    "jv_ehull_alignn": ["https://figshare.com/ndownloader/files/31458658", 1],
    "jv_dfpt_piezo_max_dielectric_alignn": [
        "https://figshare.com/ndownloader/files/31458652",
        1,
    ],
    "jv_dfpt_piezo_max_dij_alignn": [
        "https://figshare.com/ndownloader/files/31458655",
        1,
    ],
    "jv_exfoliation_energy_alignn": [
        "https://figshare.com/ndownloader/files/31458676",
        1,
    ],
    "mp_e_form_alignnn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignnn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "qm9_U0_alignn": ["https://figshare.com/ndownloader/files/31459054", 1],
    "qm9_U_alignn": ["https://figshare.com/ndownloader/files/31459051", 1],
    "qm9_alpha_alignn": ["https://figshare.com/ndownloader/files/31459027", 1],
    "qm9_gap_alignn": ["https://figshare.com/ndownloader/files/31459036", 1],
    "qm9_G_alignn": ["https://figshare.com/ndownloader/files/31459033", 1],
    "qm9_HOMO_alignn": ["https://figshare.com/ndownloader/files/31459042", 1],
    "qm9_LUMO_alignn": ["https://figshare.com/ndownloader/files/31459045", 1],
    "qm9_ZPVE_alignn": ["https://figshare.com/ndownloader/files/31459057", 1],
    "hmof_co2_absp_alignnn": [
        "https://figshare.com/ndownloader/files/31459198",
        5,
    ],
    "hmof_max_co2_adsp_alignnn": [
        "https://figshare.com/ndownloader/files/31459207",
        1,
    ],
    "hmof_surface_area_m2g_alignnn": [
        "https://figshare.com/ndownloader/files/31459222",
        1,
    ],
    "hmof_surface_area_m2cm3_alignnn": [
        "https://figshare.com/ndownloader/files/31459219",
        1,
    ],
    "hmof_pld_alignnn": ["https://figshare.com/ndownloader/files/31459216", 1],
    "hmof_lcd_alignnn": ["https://figshare.com/ndownloader/files/31459201", 1],
    "hmof_void_fraction_alignnn": [
        "https://figshare.com/ndownloader/files/31459228",
        1,
    ],
}
parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Pretrained Models"
)
# parser.add_argument(
#     "--model_name",
#     default="jv_formation_energy_peratom_alignn",
#     help="Choose a model from these "
#     + str(len(list(all_models.keys())))
#     + " models:"
#     + ", ".join(list(all_models.keys())),
# )

parser.add_argument(
    "--prop_name", default="matbench_jdft2d_exfoliation_en", help="use a matbench dataset"
)

parser.add_argument(
    "--fold", default=0, help="fold 0-4"
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--file_path",
    default="matbench_jdft2d_exfoliation_en_fold_0/mb-jdft2d-001",
    help="Path to file.",
)

parser.add_argument(
    "--output_path",
    default=None,
    help="Path to Output.",
)

parser.add_argument(
    "--cutoff",
    default=8,
    help="Distance cut-off for graph constuction"
    + ", usually 8 for solids and 5 for molecules.",
)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


activation = {}
activation_tuple1 = {}
activation_tuple2 = {}
activation_tuple3 = {}

def get_activation(name):
    def hook(model, input, output):
        if (type(output)!=tuple):    
            #print(output.shape)
            activation[name] = output.detach()
        elif (type(output)==tuple): 
            #print(len(output))
            if len(output)==2:
                out1, out2 = output
                activation_tuple1[name] = out1.detach()
                activation_tuple2[name] = out2.detach()
                activation_tuple3[name] = None
            elif len(output)==3:
                out1, out2, out3 = output
                activation_tuple1[name] = out1.detach()
                activation_tuple2[name] = out2.detach()
                activation_tuple3[name] = out3.detach()   
    return hook   

def get_activation_tuple(name):
    def hook(model, input, output):
        if (type(output)==tuple): 
            #print(len(output))
            out1, out2 = output
            activation_tuple1[name] = out1.detach()
            activation_tuple2[name] = out2.detach()
    return hook      

def get_prediction(
    prop_name="matbench_jdft2d_exfoliation_en",
    fold=0,
    atoms=None,
    cutoff=8,
    output_path=None,
    file_path=None
):
    """Load Model with config and saved .pt state_dict"""
    folder_path = prop_name + f"_outdir_{fold}"
    config_path = os.path.join(folder_path, "config.json")
    model_path = os.path.join(folder_path, "best_model.pt")

    config_dict = loadjson(config_path)       # Load config.json
    
    config = TrainingConfig(**config_dict)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    model = ALIGNN_infer(config.model)   
    # Config the ALIGNN model, using the modified class ALIGNN_infer to have atom, bond, angle features returned
    model.load_state_dict(torch.load(model_path, weights_only=True))    # Load state dict for the saved ALIGNN model
    model = model.to(device)
    model.eval()

    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff))
    #print(g)
    #print(lg)
    out_data, act_list_x, act_list_y, act_list_z = (
        model([g.to(device), lg.to(device)])
    )


    substring = file_path.split('/')[-1]
    struct_file = substring.split('.vasp')[0]    
    # In case there is ".vasp" in the file name. In the matbench cases, there shouldn't be any.
    
    embed()
    for i in range(len(act_list_x)):
        act_list_x[i] = act_list_x[i].detach().cpu().numpy()

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np_act_x = np.concatenate(act_list_x, axis=0)            
    df_act_x = pd.DataFrame(np_act_x)
    df_act_x.to_csv('{}/{}_x.csv'.format(output_path, struct_file), index=False)    

    for i in range(len(act_list_y)):
        act_list_y[i] = act_list_y[i].detach().cpu().numpy()

    np_act_y = np.concatenate(act_list_y, axis=0)            
    df_act_y = pd.DataFrame(np_act_y)
    df_act_y.to_csv('{}/{}_y.csv'.format(output_path, struct_file), index=False)     

    for i in range(len(act_list_z)):
        act_list_z[i] = act_list_z[i].detach().cpu().numpy()    

    np_act_z = np.concatenate(act_list_z, axis=0)            
    df_act_z = pd.DataFrame(np_act_z)
    df_act_z.to_csv('{}/{}_z.csv'.format(output_path, struct_file), index=False)         

    out_data = out_data.detach().cpu().numpy().flatten().tolist()


    return out_data


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    prop_name = args.prop_name
    fold = args.fold
    file_path = args.file_path
    file_format = args.file_format
    output_path = args.output_path
    cutoff = args.cutoff
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", file_format)

    # atoms is a single compound!
    out_data = get_prediction(
        prop_name=prop_name, fold=fold, cutoff=float(cutoff), atoms=atoms, output_path= output_path, file_path=file_path
    )

    print("Predicted value:", prop_name, file_path, out_data)

