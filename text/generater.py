import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import pandas as pd
import os
# import chemnlp
from chemnlp.utils.describe import atoms_describer
from robocrys import StructureCondenser, StructureDescriber
import warnings
from collections import defaultdict
import logging


warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='get embeddings on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--database', help='the source database of the property dataset', default="matbench", type=str, required=False)
parser.add_argument('--prop_name', help='name of the property dataset, eg. matbench_jdft2d', type=str, required=False)
parser.add_argument('--struc_dir', help='path to access the directory containing all struc files', type=str, required=False)
parser.add_argument('--start', default=0, type=int,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--end', type=int, required=False)
parser.add_argument('--id_len', type=int,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo', 'combo'],default='raw', type=str, required=False)
parser.add_argument('--skip_sentence', help='skip the ith sentence or a specific topic', default="none", required=False)
args,_ = parser.parse_known_args()

def describe_chemical_data(data, skip="none"):
    description = ""
    if 'chemical_info' in data and skip != 'chemical':
        description += "The chemical information include: "
        chem_info = data['chemical_info']
        description += f"The chemical has an atomic formula of {chem_info.get('atomic_formula', 'N/A')} with a prototype of {chem_info.get('prototype', 'N/A')};"
        description += f"Its molecular weight is {chem_info.get('molecular_weight', 'N/A')} g/mol; "
        description += f"The atomic fractions are {chem_info.get('atomic_fraction', 'N/A')}, and the atomic values X and Z are {chem_info.get('atomic_X', 'N/A')} and {chem_info.get('atomic_Z', 'N/A')}, respectively."

    if 'structure_info' in data and skip != 'structure':
        description += "The structure information include: "
        struct_info = data['structure_info']
        description += f"The lattice parameters are {struct_info.get('lattice_parameters', 'N/A')} with angles {struct_info.get('lattice_angles', 'N/A')} degrees; "
        description += f"The space group number is {struct_info.get('spg_number', 'N/A')} with the symbol {struct_info.get('spg_symbol', 'N/A')}; "
        description += f"The top K XRD peaks are found at {struct_info.get('top_k_xrd_peaks', 'N/A')} degrees; "
        description += f"The material has a density of {struct_info.get('density', 'N/A')} g/cm³, crystallizes in a {struct_info.get('crystal_system', 'N/A')} system, and has a point group of {struct_info.get('point_group', 'N/A')}; "
        description += f"The Wyckoff positions are {struct_info.get('wyckoff', 'N/A')}; "
        description += f"The number of atoms in the primitive and conventional cells are {struct_info.get('natoms_primitive', 'N/A')} and {struct_info.get('natoms_conventional', 'N/A')}, respectively; "
        
        if 'bond_distances' in struct_info and skip != 'bond':
            bond_distances = struct_info['bond_distances']
            bond_descriptions = ", ".join([f"{bond}: {distance} " for bond, distance in bond_distances.items()])
            description += f"The bond distances are as follows: {bond_descriptions}. "
    return description.strip()

def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "#\n"
        + " ".join([str(int(x)) for x in angles])
        + "@\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c]) + "&"
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str
def atoms_describer(
    atoms=[], xrd_peaks=5, xrd_round=1, cutoff=4, take_n_bomds=2,include_spg=True
):
    """Describe an atomic structure."""
    if include_spg:
       spg = Spacegroup3D(atoms)
    theta, d_hkls, intens = XRD().simulate(atoms=(atoms))
    dists = defaultdict(list)
    elements = atoms.elements
    for i in atoms.get_all_neighbors(r=cutoff):
        for j in i:
            key = "-".join(sorted([elements[j[0]], elements[j[1]]]))
            dists[key].append(j[2])
    bond_distances = {}
    for i, j in dists.items():
        dist = sorted(set([round(k, 2) for k in j]))
        if len(dist) >= take_n_bomds:
            dist = dist[0:take_n_bomds]
        bond_distances[i] = ", ".join(map(str, dist))
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    info = {}
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": (fracs),
        "atomic_X": ", ".join(
            map(str, [Specie(s).X for s in atoms.uniq_species])
        ),
        "atomic_Z": ", ".join(
            map(str, [Specie(s).Z for s in atoms.uniq_species])
        ),
    }
    struct_info = {
        "lattice_parameters": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.abc])
        ),
        "lattice_angles": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.angles])
        ),
        #"spg_number": spg.space_group_number,
        #"spg_symbol": spg.space_group_symbol,
        "top_k_xrd_peaks": ", ".join(
            map(
                str,
                sorted(list(set([round(i, xrd_round) for i in theta])))[
                    0:xrd_peaks
                ],
            )
        ),
        "density": round(atoms.density, 3),
        #"crystal_system": spg.crystal_system,
        #"point_group": spg.point_group_symbol,
        #"wyckoff": ", ".join(list(set(spg._dataset["wyckoffs"]))),
        "bond_distances": bond_distances,
        #"natoms_primitive": spg.primitive_atoms.num_atoms,
        #"natoms_conventional": spg.conventional_standard_structure.num_atoms,
    }
    if include_spg:
        struct_info["spg_number"]=spg.space_group_number
        struct_info["spg_symbol"]=spg.space_group_symbol
        struct_info["crystal_system"]=spg.crystal_system
        struct_info["point_group"]=spg.point_group_symbol
        struct_info["wyckoff"]=", ".join(list(set(spg._dataset["wyckoffs"])))
        struct_info["natoms_primitive"]=spg.primitive_atoms.num_atoms
        struct_info["natoms_conventional"]=spg.conventional_standard_structure.num_atoms
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    line = "The number of atoms are: "+str(atoms.num_atoms) +". " #, The elements are: "+",".join(atoms.elements)+". "
    for i, j in info.items():
        if not isinstance(j, dict):
            line += "The " + i + " is " + j + ". "
        else:
            #print("i",i)
            #print("j",j)
            for ii, jj in j.items():
                tmp=''
                if isinstance(jj,dict):
                   for iii,jjj in jj.items():
                        tmp+=iii+": "+str(jjj)+" "
                else:
                   tmp=jj
                line += "The " + ii + " is " + str(tmp) + ". "
    return line

def get_robo(structure=None):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    return description

def get_text(atoms, text):
    if text == 'robo':
        return get_robo(atoms.pymatgen_converter())
    elif text == 'raw':
        return Poscar(atoms).to_string()
    elif text == "chemnlp":
        return describe_chemical_data(atoms_describer(atoms=atoms), skip=args.skip_sentence)
    elif text == 'combo':
        return get_crystal_string_t(atoms)


def main_jarvis(args):  # The function to process jarvis datasets
    dat = data('dft_3d')
    text_dic = defaultdict(list)
    err_ct = 0
    end = len(dat)
    if args.end:
        end = min(args.end, len(dat))
    for entry in tqdm(dat[args.start:end], desc="Processing data"):

        text = get_text(Atoms.from_dict(entry['atoms']), args.text)
        text_dic['jid'].append(entry['jid'])
        text_dic['formula'].append(entry['formula'])
        text_dic['text'].append(text)
    df_text = pd.DataFrame.from_dict(text_dic)
    output_file = f"{args.text}_{args.start}_{end}_skip_{args.skip_sentence}.csv"
    if args.output_dir:
        output_file = os.path.join(args.output_dir, output_file)
    df_text.to_csv(output_file)
    logging.info(f"Saved output text to {output_file}")

def main_mb(args):  # The function to process matbench datasets
    # dat = data('dft_3d')
    text_dic = defaultdict(list)
    if not args.struc_dir:
        raise Exception("please specify struc_dir: the directory path to structure files")
    if not args.id_len:
        raise Exception("please specify id_len: the identifier code length for the dataset, e.g. 3 for XXX, 4 for XXXX")
    
    struc_dir = args.struc_dir    # folder that contains id_prop.csv and dataset-specific poscar files

    if args.output_dir:
        output_dir = args.output_dir
    else:
        parent_dir = os.path.dirname(struc_dir)     # The direct parent directory of the struc_dir
        output_dir = os.path.join(parent_dir, f"text_{args.text}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for idx in tqdm(range(args.start+1, args.end+1), desc="Processing data"):
    for idx in range(args.start+1, args.end+1):
        compound_identifier = "mb-" + args.prop_name.split("_")[1] + f"-{str(idx).zfill(args.id_len)}"
        print(f"Generating for {compound_identifier}")
        # e.g. mb-jdft2d-001
        file_path = os.path.join(struc_dir, compound_identifier)
        atoms = Atoms.from_poscar(file_path)

        text = get_text(atoms, args.text)
        text_dic['jid'].append(compound_identifier)
        text_dic['formula'].append(atoms.composition.formula)
        text_dic['text'].append(text)

        if idx%10==0 or idx==args.end:
            with open(f"{output_dir}/text_dic.json", 'w') as file:
                json.dump(data, file, indent=4)

    df_text = pd.DataFrame.from_dict(text_dic)
    output_file = f"{args.text}_{args.start}_{args.end}_skip_{args.skip_sentence}.csv"

    output_file = os.path.join(output_dir, output_file)
    df_text.to_csv(output_file)
    logging.info(f"Saved output text to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)     # for logging the code execution process, not functional for text generation
    if args.database == "matbench":
        main_mb(args)
    elif args.database == "jarvis":
        main_jarvis(args)
    logging.info(f"Finished generate text")