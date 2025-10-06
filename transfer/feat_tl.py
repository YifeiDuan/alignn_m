import sys, os
import numpy as np
import pandas as pd

import argparse

sys.path.append("/home/jupyter/YD/alignn_tag_v2/")
from zeo.alignn_feature_dataset_prep_zeo import *

import warnings
warnings.filterwarnings('ignore')

python_executable = sys.executable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Zeo Dac feat_zeo.py args"
    )
    parser.add_argument(
        "--struc_file_dir",
        default="../zeo/zeo_data/dac/MOR/output1/",
        help="path to the id_prop.csv file",
    )
    parser.add_argument(
        "--prop_name", default="hoa", help="use a matbench dataset"
    )

    parser.add_argument(
        "--id_prop_filename", default="id_prop_random_200_400.csv", help="use a matbench dataset"
    )


    parser.add_argument(
        "--if_sample",
        type=str,
        default="Y",
        help="0: no sampling; 1: sampling",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="sample size",
    )

    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="starting id for sampling",
    )

    parser.add_argument(
        "--train_ratio",type=float,
        default=0.75,
        help="ratio of training set",
    )

    parser.add_argument(
        "--file_format", default="cif", help="poscar/cif/xyz/pdb file format."
    )
    
    parser.add_argument(
        "--main_dir",
        default=None,
        help="Path to main folder.",
    )

    parser.add_argument(
        "--output_dir",
        default="../embed_dac_hoa/",
        help="Path to Output.",
    )
    args = parser.parse_args(sys.argv[1:])


    ##### 1. Activate structure embeddings from pretrained ALIGNN model
    cmd = (
        python_executable + " "+
        "../zeo/pretrained_activation_zeo.py --prop_name "
        + args.prop_name
        + " --id_prop_filename "
        + args.id_prop_filename
        + " --if_sample "
        + args.if_sample
        + " --sample_size "
        + str(args.sample_size)
        + " --start_id "
        + str(args.start_id)
        + " --train_ratio "
        + str(args.train_ratio)
        + " --file_format "
        + args.file_format
        + " --main_dir "
        + args.main_dir
        + " --file_dir "
        + args.struc_file_dir
        + " --output_dir "
        + args.output_dir
    )
    print(cmd)
    os.system(cmd)


    ##### 2. Prepare holistic graph embedding
    feat_prep(
        dataset = args.prop_name,
        feat_dir = args.main_dir,
        id_prop_dir = args.struc_file_dir,
        id_prop_file = args.id_prop_filename,
        start_id = args.start_id,
        sample_size = args.sample_size,
        train_ratio = args.train_ratio,
        identifier = 'jid',
        prop_col = 'target',
        path2 = ['x', 'y', 'z'],
        path3 = [9, 9, 5]
    )
    feat_path = os.path.join(args.main_dir,
                             f"embed_dac_{args.prop_name}/start_{args.start_id}_sample_{args.sample_size}_train_{args.train_ratio}")
    id_prop_file_path = os.path.join(os.path.join(args.struc_file_dir, args.prop_name), args.id_prop_filename)
    split_combined_feat(
        feat_path = feat_path,
        id_prop_file = id_prop_file_path,
        sample_size = args.sample_size,
        train_ratio = args.train_ratio
    )


    ##### 3. Merge graph features with text features
    ######## 3.1 robo text
    cmd = (
        python_executable + " "+
        "features.py --database "
        + "zeo"
        + " --prop "
        + f"dac_{args.prop_name}"
        + " --input_dir "
        + f"../text/zeoDAC_robo_start_{args.start_id}_sample_{args.sample_size}"
        + " --sample_size "
        + str(args.sample_size)
        + " --train_ratio "
        + str(args.train_ratio)
        + " --text "
        + "robo"
        + " --llm "
        + "matbert-base-cased"
        + " --gnn_file_dir "
        + f"../zeo/embed_dac_{args.prop_name}/start_{args.start_id}_sample_{args.sample_size}_train_{args.train_ratio}/xyz"
    )
    print(cmd)
    os.system(cmd)
    
    ######## 3.2 chemnlp text
    cmd = (
        python_executable + " "+
        "features.py --database "
        + "zeo"
        + " --prop "
        + f"dac_{args.prop_name}"
        + " --input_dir "
        + f"../text/zeoDAC_chemnlp_start_{args.start_id}_sample_{args.sample_size}"
        + " --sample_size "
        + args.sample_size
        + " --train_ratio "
        + args.train_ratio
        + " --text "
        + "chemnlp"
        + " --llm "
        + "matbert-base-cased"
        + " --gnn_file_dir "
        + f"../zeo/embed_dac_{args.prop_name}/start_{args.start_id}_sample_{args.sample_size}_train_{args.train_ratio}/xyz"
    )
    print(cmd)
    os.system(cmd)