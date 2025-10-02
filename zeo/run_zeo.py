"""Train ALIGNN model on Zeolite DAC dataset."""
# Dataset Ref: https://github.com/marko-petkovic/zeolite-property-prediction?tab=readme-ov-file/Data_raw

import glob
import os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score

import argparse

import warnings
warnings.filterwarnings('ignore')

python_executable = sys.executable

def train_zeo_dac(
    config_template="config_zeo.json", file_format="cif", 
    id_prop_path="zeo_data/dac/MOR/output1/hoa/id_prop.csv",
    sample_size=200, 
    start_id = 0,
    train_ratio=0.75
):
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    df = pd.read_csv(id_prop_path)
    # df = df.sample(n=sample_size, random_state=42)
    df = df[start_id:(start_id+sample_size)]
    train_df = df[:int(train_ratio*sample_size)]
    test_df = df[int(train_ratio*sample_size):]
    # Making sure there are not spaces or parenthesis which
    # can cause issue while creating folder
    prop = "hoa" if "hoa" in id_prop_path else "henry"
    fold_name = os.path.dirname(id_prop_path)

    id_prop_file = f"id_prop_random_{start_id}_{start_id+sample_size}"
    df.to_csv(os.path.join(fold_name, f"{id_prop_file}.csv"), index=False)
    # id_prop_file, ext = os.path.splitext(os.path.basename(id_prop_path))
    os.chdir(fold_name)     # create a folder for the current fold of the current prop dataset, and change the working directory here
    # ALIGNN requires the id_prop.csv file
    
    val_df = train_df[0 : len(test_df)]

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    config = loadjson(config_template)
    # FIXME: config["filename"]
    config["filename"] = fold_name
    config["n_train"] = n_train
    config["n_val"] = n_val
    config["n_test"] = n_test
    config["keep_data_order"] = True
    config["batch_size"] = 32
    config["test_batch_size"] = 4
    # TODO: after debugging, change epochs back to 500
    config["epochs"] = 100
    # config["epochs"] = 10
    fname = f"config_{sample_size}.json"
    dumpjson(data=config, filename=fname)

    print(script_dir)
    os.chdir(script_dir)      # change working directory back to the directory of .ipynb nb that calls run.py
    outdir_name = (
        "dac"
        + "_"
        + prop.replace(" ", "_")
        .replace("(", "-")
        .replace(")", "-")
        + "_start_"
        + str(start_id)
        + "_sample_"
        + str(sample_size)
        + "_train_"
        + str(train_ratio)
        + "_outdir_"
    )
    cmd = (
        python_executable + " "+
        "train_zeo.py --root_dir "
        + fold_name
        + " --id_prop_file "
        + id_prop_file
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Zeo Dac run_zeo.py args"
    )
    parser.add_argument(
        "--id_prop_path",
        default="zeo_data/dac/MOR/output1/hoa/id_prop.csv",
        help="path to the id_prop.csv file",
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
        help="start from an id in the id_prop_random.csv",
    )
    parser.add_argument(
        "--train_ratio",type=float,
        default=0.75,
        help="ratio of training set",
    )
    args = parser.parse_args(sys.argv[1:])

    ##### Load config file that contains model hyperparams #####
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    config = loadjson(config_template)

    ##### Run the training loop for all tasks in mb #####
    train_zeo_dac(config_template=config_template, file_format="cif", 
                  id_prop_path=args.id_prop_path,
                  sample_size=args.sample_size, 
                  start_id=args.start_id,
                  train_ratio=args.train_ratio)

    # run_dir = "."
    # # run_dir = "/wrk/knc6/matbench/benchmarks/matbench_v0.1_alignn"

    # cwd = os.getcwd()

    # os.chdir(run_dir)

    # results = defaultdict()
    # for task in mb.tasks:
    #     task.load()
    #     task_name = task.dataset_name
    #     regr = True
    #     if "is" in task_name:
    #         regr = False
    #     results = compile_results(task_name, regression=regr)
    #     for ii, fold in enumerate(task.folds):
    #         train_df = task.get_train_and_val_data(fold, as_type="df")
    #         test_df = task.get_test_data(
    #             fold, include_target=True, as_type="df"
    #         )
    #         pred_vals = results[fold]
    #         task.record(fold, pred_vals, params=config)
    # os.chdir(cwd)
    # mb.add_metadata({"algorithm": "ALIGNN"})
    # mb.to_file("results.json.gz")
