from pymatgen.core import Structure
from jarvis.core.atoms import Atoms
import os
import wget

import pandas as pd
import numpy as np

def extract_compositions_from_cifs(cif_dir, output_path="compositions.csv"):
    records = []
    all_elements = set()

    # Step 1: Parse all CIFs and record element counts
    for fname in os.listdir(cif_dir):
        if not fname.endswith(".cif"):
            continue
        path = os.path.join(cif_dir, fname)
        try:
            framework = fname.split("_")[0]
            structure = Structure.from_file(path)
            composition = structure.composition
            data = {"filename": fname, 
                    "framework":framework, 
                    "formula": composition.reduced_formula}
            for el, count in composition.get_el_amt_dict().items():
                data[el] = count
                all_elements.add(el)
            records.append(data)
        except Exception as e:
            print(f"Could not parse {fname}: {e}")

    if not records:
        print("No valid CIF files found.")
        return

    # Step 2: Ensure all element columns exist
    all_elements = sorted(all_elements)
    for rec in records:
        for el in all_elements:
            if el not in rec:
                rec[el] = 0

    # Step 3: Create DataFrame and save
    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} entries to {output_path}")