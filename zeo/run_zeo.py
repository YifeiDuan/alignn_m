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

import warnings
warnings.filterwarnings('ignore')

python_executable = sys.executable

def train_zeo_dac(
    config_template="config_zeo.json", file_format="cif"
):
    maes = []
    for ii, fold in enumerate(task.folds):
        ### Benchmarking splits are acquired with MatbenchTask methods get_train_and_val_data & get_test_data
        train_df = task.get_train_and_val_data(fold, as_type="df")
        test_df = task.get_test_data(
            fold, include_target=True, as_type="df"
        )
        # Name of the target property
        target = [
            col
            for col in train_df.columns
            if col not in ("id", "structure", "composition")
        ][0]
        # Making sure there are not spaces or parenthesis which
        # can cause issue while creating folder
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
        os.chdir(fold_name)     # create a folder for the current fold of the current prop dataset, and change the working directory here
        # ALIGNN requires the id_prop.csv file
        f = open("id_prop.csv", "w")
        for jj, j in train_df.iterrows():       # fill in the id_prop.csv file
            id = j.name
            atoms = pmg_to_atoms(j.structure)   # convert pymatgen object to JARVIS.Atoms
            pos_name = id
            atoms.write_poscar(pos_name)        # the mb-[prop]-[id] poscar object
            val = j[target]
            line = str(pos_name) + "," + str(val) + "\n"
            f.write(line)
        # There is no pre-defined validation splt, so we will use
        # a portion of training set as validation set, and
        # keep test set intact
        val_df = train_df[0 : len(test_df)]
        for jj, j in val_df.iterrows():
            # for jj, j in test_df.iterrows():
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
        # FIXME: config["filename"]
        config["filename"] = fold_name
        config["n_train"] = n_train
        config["n_val"] = n_val
        config["n_test"] = n_test
        config["keep_data_order"] = True
        config["batch_size"] = 32
        config["test_batch_size"] = 32
        # TODO: after debugging, change epochs back to 500
        config["epochs"] = 500
        # config["epochs"] = 10
        fname = "config_fold_" + str(ii) + ".json"
        dumpjson(data=config, filename=fname)
        f.close()
        os.chdir("..")      # change working directory back to the parent directory of .ipynb nb that calls run.py
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
            python_executable + " "+
            "train_folder.py --root_dir "
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
    """Compile fold based results for each task."""
    # Some of the jobs such as mp_e_form takes a couple of
    # days to complete for each fold
    # so we compile the results as follows
    maes = []
    roc_aucs = []
    results = defaultdict()

    for i in glob.glob(key + "*/prediction_results_test_set.csv"):
        fold = int(i.split("/")[0].split("_")[-1])
        # print (i,fold)
        df = pd.read_csv(i)

        target_vals = df.target.values
        # id_vals = df.id.values
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
    ##### Load config file that contains model hyperparams #####
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    config = loadjson(config_template)

    ##### Run the training loop for all tasks in mb #####
    train_tasks(mb=mb, config_template=config_template, file_format="poscar")

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
