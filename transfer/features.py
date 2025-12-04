
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
import re

SEED = 1
props_jv_mp = ['formation_energy_peratom', 'ehull', 'mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon']
props_mb = [
                "matbench_jdft2d",
                "matbench_phonons",
                "matbench_dielectric",
                "matbench_log_gvrh",
                "matbench_log_kvrh",
                "matbench_perovskites",
                "matbench_mp_e_form",
                "matbench_mp_gap",
                "matbench_mp_is_metal",
            ]
# props = ['formation_energy_peratom']


parser = argparse.ArgumentParser(description='run ml regressors on dataset')
parser.add_argument('--database', help='the source database of the property dataset', default="matbench", type=str, required=False)
parser.add_argument('--input_dir', help='input text data directory', default="../text/matbench_jdft2d", type=str,required=False)   # For text embeddings
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo', 'combo'], default='robo', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--gen_llm', help='llm used to generate the text', default='llama-3-8B-instruct', type=str,required=False)
parser.add_argument('--gen_prompt', help='llm used to generate the text', default='cif', type=str,required=False)
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--gnn_only', action='store_true')
parser.add_argument('--gnn_file_dir', help='pretrained gnn embedding directory', default="../alignn/embed_matbench_jdft2d/organized", type=str, required=False)
parser.add_argument('--gnn_file_path', help='pretrained gnn embedding file', default="../alignn/embed_matbench_jdft2d/fold_0/xyz/data_final_9_9_5.csv", type=str, required=False)
parser.add_argument('--split_dir', type=str, required=False)
parser.add_argument('--sample_size', type=int, required=False)
parser.add_argument('--start_id', type=int, required=False)
parser.add_argument('--train_ratio', type=float, required=False)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--skip_sentence', help='skip the ith sentence', default=None, required=False)
parser.add_argument('--mask_words', help='skip the ith word', default=None, required=False)
parser.add_argument('--prop', help="specify property from \
                    'matbench_jdft2d',\
                    'matbench_phonons',\
                    'matbench_dielectric',\
                    'matbench_log_gvrh',\
                    'matbench_log_kvrh',\
                    'matbench_perovskites',\
                    'matbench_mp_e_form',\
                    'matbench_mp_gap',\
                    'matbench_mp_is_metal',\
                    'formation_energy_peratom', 'ehull', 'mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon'", default='all', required=False)

args =  parser.parse_args()
config = configparser.ConfigParser()

selected_samples = ["JVASP-1151"]


def prepare_dataset_jv_mp(args, prop):
    embeddings = []
    labels = []
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file)]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0)
    dat = data('dft_3d')
    ids = []


    for i in tqdm(dat, desc="Preparing data"):
        if args.sample:
            if i['jid'] not in selected_samples:
                continue
        if i[prop]!='na':
            if i['jid'] in df_embed.index:
                embeddings.append(df_embed.loc[i['jid']].values)
                labels.append(i[prop])
                ids.append(i['jid'])
    

    num_cols = len(embeddings[0])
    col_names = [i for i in range(num_cols)]
    df_data = pd.DataFrame(embeddings, columns=col_names)
    df_data[prop] = labels
    df_data["ids"] = ids
    if args.gnn_only:
        dataset_filename = f"dataset_only_prop_{prop}"
    else:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}_prop_{prop}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}_prop_{prop}"
    dataset_path = f"./data/{dataset_filename}"
    df_data['ids'] = df_data['ids'] + '.vasp'
    # TODO: ALIGNN features
    if args.gnn_file_path:    
        df_gnn = pd.read_csv(args.gnn_file_path)
        dataset_path = dataset_path.replace("dataset_", "dataset_alignn_")
        if args.gnn_only:
            df_gnn = pd.read_csv(args.gnn_file_path)
            df_gnn['id'] = df_gnn['id'] + '.vasp'


            df_data = df_data[[prop, "ids"]].merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))

        else:
            df_gnn['id'] = df_gnn['id'] + '.vasp'
            df_data = df_data.merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))
            print(df_data.head())
        df_data[prop] = df_data.pop(prop)
        df_data["ids"] = df_data.pop("ids")

    if args.split_dir:
        split_path = os.path.join(args.split_dir, f"dataset_split_{prop}.json")
        assert prop in split_path
        # for subset in ["test", "val", "train"]:
        #     sub_filename = f"{dataset_filename}_{subset}.csv"
        #     df_sub = pd.read_csv(os.path.join(args.split_dir, sub_filename))
        #     df_sub['ids'] = df_sub['ids'] + '.vasp'
        #     df_datasub = df_data[df_data["ids"].isin(df_sub["ids"])].drop(columns={"jid", "jid.1"})
        #     df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
        #     logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
        with open(split_path, 'r') as json_file:
            split_dic = json.load(json_file)
        for subset in ["test", "val", "train"]:
            sub_ids = [val+'.vasp' for val in split_dic[f"id_{subset}"]]
            if not set(sub_ids).issubset(df_data['ids']):
                logging.error(f"Subset {subset} not found in GNN or LLM embedding dataset for {prop} property. Skipping...")
                return None, None
            df_datasub = df_data[df_data['ids'].isin(sub_ids)].drop(columns={"id", "full"}, errors='ignore')
            df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
            print(f"{dataset_path}_{subset}: {len(df_datasub)}")
            logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
    
    else:
        logging.info(f"Constructed {df_data.shape[0]} samples for {prop} property")
        df_data.to_csv(f"{dataset_path}.csv")
        logging.info(f"Saved dataset to {dataset_path}.csv")


    return embeddings, labels

def prepare_dataset_mb(args, prop):

    # 1. Load text embeddings
    text_embed_subdir = f"embedding_*"
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(text_embed_subdir, file_path)
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)    # Should be something like "YY/text/matbench_XX/embedding_XX/embeddings_MM_robo_*.csv"
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"   # "(\d+)" matches a sequence of digits
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file) and "cluster" not in file]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0).reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    if args.gnn_only:
        dataset_filename = f"dataset_alignn_only_prop_{prop}"
    else:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}_prop_{prop}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}_prop_{prop}"

    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/{prop}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # TODO: Process split-specific merging
    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file
    split_dirs = find_subdirs_with_string(gnn_file_dir, "split_fold")       # Find matching subdirs with "split_fold" in name
    if len(split_dirs) != 0:
        for split_dir in split_dirs:

            split_name = os.path.basename(split_dir)
            split_data_save_dir = os.path.join(data_save_dir, split_name)
            if not os.path.exists(split_data_save_dir):
                os.makedirs(split_data_save_dir)

            for subset in ["train", "val", "test"]:
                df_subset = pd.read_csv(os.path.join(split_dir, f"data_{subset}.csv"))
                df_subset = df_subset.merge(df_embed, how='inner', left_on="id", right_on="ids", suffixes=('_gnn', '_lm'))
                print(df_subset.head())
                df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
                df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
                df_subset["id"] = df_subset.pop("id")     # Reordering columns, "id" to the last col
                df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
                ### Save the subset of merged multimodal data
                save_path = os.path.join(split_data_save_dir, f"{dataset_filename}_{subset}.csv")
                df_subset.to_csv(save_path)
                logging.info(f"Saved subset dataset to {save_path}")


def prepare_dataset_zeo(args, prop="dac_hoa"):
    sample_size = args.sample_size
    start_id = args.start_id
    train_ratio = args.train_ratio

    # 1. Load text embeddings
    text_embed_subdir = f"embedding_*"
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(text_embed_subdir, file_path)
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)    # Should be something like "YY/text/matbench_XX/embedding_XX/embeddings_MM_robo_*.csv"
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"   # "(\d+)" matches a sequence of digits
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file) and "cluster" not in file]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0).reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    if args.gnn_only:
        dataset_filename = f"dataset_alignn_only_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    else:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"

    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file


    for subset in ["train", "val", "test"]:
        df_subset = pd.read_csv(os.path.join(gnn_file_dir, f"data_{subset}.csv"))
        df_subset = df_subset.merge(df_embed, how='inner', left_on="jid", right_on="ids", suffixes=('_gnn', '_lm'))
        print(df_subset.head())
        df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
        df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
        df_subset["jid"] = df_subset.pop("jid")     # Reordering columns, "id" to the last col
        df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
        ### Save the subset of merged multimodal data
        save_path = os.path.join(data_save_dir, f"{dataset_filename}_{subset}.csv")
        df_subset.to_csv(save_path)
        logging.info(f"Saved subset dataset to {save_path}")


# TODO: Run this
def prepare_dataset_zeo_llm(args, prop="dac_hoa"):
    sample_size = args.sample_size
    start_id = args.start_id
    train_ratio = args.train_ratio

    # 1. Load text embeddings
    file_path = f"embeddings_{args.gen_llm}_{args.llm.replace('/', '_')}_*.csv"
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)    # Should be something like "YY/text/matbench_XX/embedding_XX/embeddings_MM_robo_*.csv"
    embed_file = glob.glob(file_path)
    if len(embed_file)>1:
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0).reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    dataset_filename = f"dataset_alignn_{args.gen_llm}-input-{args.gen_prompt}_{args.llm.replace('/', '_')}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    
    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file


    for subset in ["train", "val", "test"]:
        df_subset = pd.read_csv(os.path.join(gnn_file_dir, f"data_{subset}.csv"))
        df_subset = df_subset.merge(df_embed, how='inner', left_on="jid", right_on="ids", suffixes=('_gnn', '_lm'))
        print(df_subset.head())
        df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
        df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
        df_subset["jid"] = df_subset.pop("jid")     # Reordering columns, "id" to the last col
        df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
        ### Save the subset of merged multimodal data
        save_path = os.path.join(data_save_dir, f"{dataset_filename}_{subset}.csv")
        # print("save_path: " + save_path)
        df_subset.to_csv(save_path)
        logging.info(f"Saved subset dataset to {save_path}")



def prepare_dataset_zeo_text_only(args, prop="dac_hoa"):
    sample_size = args.sample_size
    start_id = args.start_id
    train_ratio = args.train_ratio

    # 1. Load text embeddings
    text_embed_subdir = f"embedding_*"
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(text_embed_subdir, file_path)
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)    # Should be something like "YY/text/matbench_XX/embedding_XX/embeddings_MM_robo_*.csv"
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"   # "(\d+)" matches a sequence of digits
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file) and "cluster" not in file]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0).reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    dataset_filename = f"dataset_{args.llm}_{args.text}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_{args.llm}_{args.text}_skip_{args.skip_sentence}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_{args.llm}_{args.text}_mask_{args.mask_words}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"

    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/text_only_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # TODO: Process split-specific merging
    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file


    for subset in ["train", "val", "test"]:
        df_subset = pd.read_csv(os.path.join(gnn_file_dir, f"data_{subset}.csv"))
        df_subset = df_subset[["jid", "target"]]
        df_subset = df_subset.merge(df_embed, how='inner', left_on="jid", right_on="ids", suffixes=('_gnn', '_lm'))
        print(df_subset.head())
        df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
        df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
        df_subset["jid"] = df_subset.pop("jid")     # Reordering columns, "id" to the last col
        df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
        ### Save the subset of merged multimodal data
        save_path = os.path.join(data_save_dir, f"{dataset_filename}_{subset}.csv")
        df_subset.to_csv(save_path)
        logging.info(f"Saved subset dataset to {save_path}")

def prepare_dataset_zeo_llm_text_only(args, prop="dac_hoa"):
    sample_size = args.sample_size
    start_id = args.start_id
    train_ratio = args.train_ratio

    # 1. Load text embeddings
    file_path = f"embeddings_{args.gen_llm}_{args.llm.replace('/', '_')}_*.csv"
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0).reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    dataset_filename = f"dataset_{args.gen_llm}_{args.llm}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"

    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/text_only_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # TODO: Process split-specific merging
    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file


    for subset in ["train", "val", "test"]:
        df_subset = pd.read_csv(os.path.join(gnn_file_dir, f"data_{subset}.csv"))
        df_subset = df_subset[["jid", "target"]]
        df_subset = df_subset.merge(df_embed, how='inner', left_on="jid", right_on="ids", suffixes=('_gnn', '_lm'))
        print(df_subset.head())
        df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
        df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
        df_subset["jid"] = df_subset.pop("jid")     # Reordering columns, "id" to the last col
        df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
        ### Save the subset of merged multimodal data
        save_path = os.path.join(data_save_dir, f"{dataset_filename}_{subset}.csv")
        df_subset.to_csv(save_path)
        logging.info(f"Saved subset dataset to {save_path}")


def prepare_dataset_zeo_rand_text(args, prop="dac_hoa"):
    sample_size = args.sample_size
    start_id = args.start_id
    train_ratio = args.train_ratio

    # 1. Load text embeddings
    text_embed_subdir = f"embedding_*"
    file_path = f"embeddings_{args.llm.replace('/', '_')}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_robo_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_robo_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(text_embed_subdir, file_path)
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)    # Should be something like "YY/text/matbench_XX/embedding_XX/embeddings_MM_robo_*.csv"
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"   # "(\d+)" matches a sequence of digits
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file) and "cluster" not in file]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0)

    # Random numbers
    df_random = pd.DataFrame(0.01*np.random.randn(*df_embed.shape), 
                            index=df_embed.index,
                            columns=df_embed.columns)
    df_random = df_random.reset_index().rename(columns={'index': 'ids'})

    # 2. Prepare save names
    dataset_filename = f"dataset_{args.llm}_randtext_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_{args.llm}_randtext_skip_{args.skip_sentence}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_{args.llm}_randtext_mask_{args.mask_words}_prop_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"

    # 3. Multimodal feature concat: Merge text emebddings with GNN-inferred embeddings
    data_save_dir = f"./data/randtext_{prop}_start_{start_id}_sample_{sample_size}_train_{train_ratio}"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # TODO: Process split-specific merging
    if args.gnn_file_dir:
        gnn_file_dir = args.gnn_file_dir      # direct parent dir of the gnn embedding file


    for subset in ["train", "val", "test"]:
        df_subset = pd.read_csv(os.path.join(gnn_file_dir, f"data_{subset}.csv"))
        df_subset = df_subset[["jid", "target"]]
        df_subset = df_subset.merge(df_random, how='inner', left_on="jid", right_on="ids", suffixes=('_gnn', '_lm'))
        print(df_subset.head())
        df_subset["target"] = df_subset.pop("target")       # Reordering columns, "matbench_PROP" to the last col
        df_subset["ids"] = df_subset.pop("ids")     # Reordering columns, "ids" to the last col
        df_subset["jid"] = df_subset.pop("jid")     # Reordering columns, "id" to the last col
        df_subset = df_subset.drop(df_subset.filter(like='Unnamed').columns, axis=1)
        ### Save the subset of merged multimodal data
        save_path = os.path.join(data_save_dir, f"{dataset_filename}_{subset}.csv")
        df_subset.to_csv(save_path)
        logging.info(f"Saved subset dataset to {save_path}")
    



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




if __name__ == "__main__":
    print(args.database)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    if args.database == "jarvis" or args.database == "mp":
        if args.prop != 'all':
            assert args.prop in props_jv_mp
            props = [args.prop]
        else:
            props = props_jv_mp
        for prop in props:
            prepare_dataset_jv_mp(args, prop)
    elif args.database == "matbench":
        if args.prop != 'all':
            assert args.prop in props_mb
            props = [args.prop]
        else:
            props = props_mb
        for prop in props:
            prepare_dataset_mb(args, prop)
    elif args.database == "zeo":
        prop = args.prop
        prepare_dataset_zeo(args, prop)
    elif args.database == "zeo_llm":
        prop = args.prop
        prepare_dataset_zeo_llm(args, prop)
    elif args.database == "zeo_text":
        prop = args.prop
        prepare_dataset_zeo_text_only(args, prop)
    elif args.database == "zeo_llm_text":
        prop = args.prop
        prepare_dataset_zeo_llm_text_only(args, prop)
    elif args.database == "zeo_rand_text":
        prop = args.prop
        prepare_dataset_zeo_rand_text(args, prop)

