from jarvis.core.atoms import Atoms
import os
import wget

import pandas as pd
import numpy as np

def download_cif_from_url(
        url_dir="https://raw.githubusercontent.com/marko-petkovic/zeolite-property-prediction/refs/heads/main/Data_raw/output1",
        struc_dir="zeo_data/zeolite-property"
        ):
    if not os.path.exists(struc_dir):
        os.makedirs(struc_dir)

    file_name = "MOR_100.cif"
    url_subdir = file_name.split("_")[1].split(".")[0]

    url = os.path.join(os.path.join(url_dir, url_subdir), file_name)
    file_path = os.path.join(struc_dir, file_name)

    wget.download(url, file_path)