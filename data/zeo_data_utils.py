from jarvis.core.atoms import Atoms
import os
import wget

import pandas as pd
import numpy as np

def download_file_from_url(
        url_dir="https://raw.githubusercontent.com/marko-petkovic/zeolite-property-prediction/refs/heads/main/Data_raw/output1",
        save_dir="zeo_data/zeolite-property",
        filename="100/MOR_100.cif"
        ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url = os.path.join(url_dir, filename)
    file_name = filename
    if ".cif" in filename:
        file_name = filename.split("/")[1]
    file_path = os.path.join(save_dir, file_name)

    wget.download(url, file_path)

def prep_zeo_dac_raw_data(zeo="MOR", props=["hoa", "henry"], sample_size=None, 
                        url_dir="https://raw.githubusercontent.com/marko-petkovic/zeolite-property-prediction/refs/heads/main/Data_raw/output1",
                        save_dir="zeo_data/dac",
                        ):
    
    # Download properpty .dat file
    save_subdir = os.path.join(save_dir, zeo)
    
    path_note = url_dir.split("/")[-1]
    if "output" in path_note:
        save_subdir = os.path.join(save_subdir, path_note)

    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)

    for prop in props:
        download_file_from_url(url_dir=url_dir, save_dir=save_subdir, filename=f"{prop}.dat")

        # Process the property data

        dat = np.loadtxt(os.path.join(save_subdir, f"{prop}.dat"))
        """
        dat has 3 columns:
        - 0: id
        - 1: target (hoa, or henry)
        - 2: error
        """

        df = pd.DataFrame({"jid": [f"{zeo}_" + str(int(j)) for j in dat[:, 0]],     # jid: MOR_100
                           "target": dat[:,1]})

        if sample_size:
            df = df.head(sample_size)
            save_subdir = os.path.join(save_subdir, f"sample_{sample_size}")
        
        save_propdir = os.path.join(save_subdir, prop)
        if not os.path.exists(save_propdir):
            os.makedirs(save_propdir)

        df.to_csv(os.path.join(save_propdir, "id_prop.csv"), index=False)

    # Download corresnponding cif files
    for jid in df["jid"]:
        idx = jid.split("_")[-1]    # "MOR_100" -> "100"
        code = f"{idx}/{jid}"
        download_file_from_url(url_dir=url_dir, save_dir=save_subdir, filename=f"{code}.cif")