import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import glob
import os
import pandas as pd
from collections import defaultdict
import configparser
import itertools

SEED = 1

parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--input_dir', help='input data directory', default="./data", type=str,required=False)
parser.add_argument('--session_name', help="specify property", default='dac_hoa_sample_200_train_0.75', required=False)
parser.add_argument('--model', help='model to train', choices=['rf', 'mlp'], default='mlp', type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'], default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm embedding to use', default='matbert-base-cased', type=str,required=False)
parser.add_argument('--gnn', help='pre-trained gnn embedding to use', default='alignn', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default="./results", type=str, required=False)
args =  parser.parse_args()


def find_subdirs_with_string(directory, search_str):
    """
    Find all subdirs that contain search_str, for a given directory
    """
    matching_subdirs = []
    for root, dirs, _ in os.walk(directory):
        for subdir in dirs:
            if search_str in subdir:
                matching_subdirs.append(os.path.join(root, subdir))
    return matching_subdirs



   
# Main function
def run_regressor_rf(args):
    # 0. Process data dir path strings
    session_name = args.session_name
    data_dir = os.path.join(args.input_dir, session_name)
    # 1. Get preprocessed train, val, test data for the dataset (might be a fold)
    df_base_name = f"dataset_{args.gnn}_{args.llm}_{args.text}_prop_{session_name}"
    df_train = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_train.csv")).reset_index(drop=True)
    df_val   = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_val.csv")).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_test.csv")).reset_index(drop=True)
    ### 1.1 Separate X (input) and y (target)
    X_train = df_train.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_train = df_train["target"]
    X_val = df_val.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_val = df_val["target"]
    X_test = df_test.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_test = df_test["target"]

    # 2. Model training with hyperparam tuning
    rf = RandomForestRegressor(random_state=42)
    ### 2.1 Prepare hyperparams
    param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000],
        'max_depth': [None, 10, 20]
    }
    ### 2.2 Training with validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                    cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=3)
    grid_search.fit(X_train, y_train)
    ### 2.2 Training with validation
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparams: {best_params}, Val MAE: {best_score}")
    ### 3.1 Process save_dir name
    output_subdir = os.path.join(args.output_dir, f"{args.gnn}_{args.text}+{args.llm}")
    
    save_dir = os.path.join(output_subdir, session_name)
    ### 3.2 Save model params
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_dict = {
                    "hyperparameters": best_params,
                    "val_mae": best_score
    }
    with open(os.path.join(save_dir, "rf_cv_best_params.json"), "w") as f:
        json.dump(model_dict, f, indent=4)
    
    # 4. Evaluate best model on data splits
    ### 4.1 Train set
    y_train_pred = best_model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    print(f"Train MAE: {mae_train}")
    train_json = {
        "ids": list(df_train["jid"]),
        "y_true": list(y_train),
        "y_pred": list(y_train_pred),
        "train_mae": mae_train
    }
    with open(os.path.join(save_dir, "rf_cv_eval_train.json"), "w") as f:
        json.dump(train_json, f, indent=4)
    ### 4.2 Test set
    y_test_pred = best_model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"Test MAE: {mae_test}")
    test_json = {
        "ids": list(df_test["jid"]),
        "y_true": list(y_test),
        "y_pred": list(y_test_pred),
        "test_mae": mae_test
    }
    with open(os.path.join(save_dir, "rf_cv_eval_test.json"), "w") as f:
        json.dump(test_json, f, indent=4)




def run_regressor_mlp(args):
    # 0. Process data dir path strings
    session_name = args.session_name
    data_dir = os.path.join(args.input_dir, session_name)
    # 1. Get preprocessed train, val, test data for the dataset (might be a fold)
    df_base_name = f"dataset_{args.gnn}_{args.llm}_{args.text}_prop_{session_name}"
    df_train = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_train.csv")).reset_index(drop=True)
    df_val   = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_val.csv")).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(data_dir, f"{df_base_name}_test.csv")).reset_index(drop=True)
    ### 1.1 Separate X (input) and y (target)
    X_train = df_train.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_train = df_train["target"]
    X_val = df_val.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_val = df_val["target"]
    X_test = df_test.drop(columns=["jid", "ids", "target"], errors="ignore")
    y_test = df_test["target"]

    # 2. Model training with hyperparam tuning
    mlp = MLPRegressor(random_state=42)
    ### 2.1 Prepare hyperparams
    param_grid = {
        'hidden_layer_sizes': [
            (1024,),  # Single large layer
            (1024, 512),  # Two-layer decreasing
            (1024, 512, 256),  # Three-layer decreasing
            (1024, 512, 256, 128),  # Deeper but still decreasing
            (512,),  # Moderate single-layer model
            (512, 256),  # Two-layer moderate
            (512, 256, 128),  # Three-layer moderate
        ],  
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500]  # Number of iterations
    }
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                    cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=3)
    grid_search.fit(X_train, y_train)
    ### 2.2 Training with validation
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparams: {best_params}, Val MAE: {best_score}")

    # 3. Save best model
    output_subdir = os.path.join(args.output_dir, f"{args.gnn}_{args.text}+{args.llm}")
    
    save_dir = os.path.join(output_subdir, session_name)
    ### 3.2 Save model params
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_dict = {
                    "hyperparameters": best_params,
                    "weights": [w.tolist() for w in best_model.coefs_],  # Convert NumPy arrays to lists
                    "biases": [b.tolist() for b in best_model.intercepts_],  # Convert NumPy arrays to lists
                    "val_mae": best_score
    }
    with open(os.path.join(save_dir, "mlp_cv_best_params.json"), "w") as f:
        json.dump(model_dict, f, indent=4)
    
    # 4. Evaluate best model on data splits
    ### 4.1 Train set
    y_train_pred = best_model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    print(f"Train MAE: {mae_train}")
    train_json = {
        "ids": list(df_train["jid"]),
        "y_true": list(y_train),
        "y_pred": list(y_train_pred),
        "train_mae": mae_train
    }
    with open(os.path.join(save_dir, "mlp_cv_eval_train.json"), "w") as f:
        json.dump(train_json, f, indent=4)
    ### 4.2 Test set
    y_test_pred = best_model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"Test MAE: {mae_test}")
    test_json = {
        "ids": list(df_test["jid"]),
        "y_true": list(y_test),
        "y_pred": list(y_test_pred),
        "test_mae": mae_test
    }
    with open(os.path.join(save_dir, "mlp_cv_eval_test.json"), "w") as f:
        json.dump(test_json, f, indent=4)

            

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    if args.model == "rf":
        df_rst = run_regressor_rf(args)
    elif args.model == "mlp":
        df_rst = run_regressor_mlp(args)
   

