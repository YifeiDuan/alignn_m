import numpy as np
import pandas as pd
import os
import glob

from matbench.bench import MatbenchBenchmark

def feat_prep(
        dataset = "matbench_jdft2d",
        model_fold = 0,
        feat_dir = "../",
        id_prop_dir = "../../text",
        identifier = 'id',
        prop_col = 'target',
        path2 = ['x', 'y', 'z'],
        path3 = [9, 9, 5]
):
    
    feat_path = os.path.join(feat_dir, f"embed_{dataset}/fold_{model_fold}")
    input256 = [str(i) for i in range(256)]

    # 1
    # 1.1  Attach the appropriate property value and identifier (jid) to each of the extracted features file based on id_prop.csv
    id_list = pd.read_csv(os.path.join(id_prop_dir, f"{dataset}/id_prop_all.csv"))
    id_mat = list(id_list.iloc[:, 0])
    data_dict = {}
    for a in range(len(id_mat)):
        data_dict[str(id_mat[a]+'x')] = pd.read_csv("{}/{}_{}.csv".format(feat_path, id_mat[a], path2[0]))
        data_dict[str(id_mat[a]+'y')] = pd.read_csv("{}/{}_{}.csv".format(feat_path, id_mat[a], path2[1]))
        data_dict[str(id_mat[a]+'z')] = pd.read_csv("{}/{}_{}.csv".format(feat_path, id_mat[a], path2[2]))
    
    for i in range(len(id_mat)):
        data_dict[str(id_mat[i]+'x')]['id'] = id_list['id'][i]
        data_dict[str(id_mat[i]+'y')]['id'] = id_list['id'][i]
        data_dict[str(id_mat[i]+'z')]['id'] = id_list['id'][i]
            
            
        data_dict[str(id_mat[i]+'x')]["target"] = id_list["target"][i]
        data_dict[str(id_mat[i]+'y')]["target"] = id_list["target"][i]
        data_dict[str(id_mat[i]+'z')]["target"] = id_list["target"][i]

            
        data_dict[str(id_mat[i]+'x')].to_csv("{}/{}_{}.csv".format(feat_path, id_mat[i], path2[0]), index=False)
        data_dict[str(id_mat[i]+'y')].to_csv("{}/{}_{}.csv".format(feat_path, id_mat[i], path2[1]), index=False)
        data_dict[str(id_mat[i]+'z')].to_csv("{}/{}_{}.csv".format(feat_path, id_mat[i], path2[2]), index=False)
    

    # 2
    # 2.1  Create a seperate file for each of the features (atom, bond, angle) based on the extracted checkpoints
    file_path_x = '{}/{}'.format(feat_path, path2[0])
    file_path_y = '{}/{}'.format(feat_path, path2[1])
    file_path_z = '{}/{}'.format(feat_path, path2[2])

    if not os.path.exists(file_path_x):
        os.mkdir(file_path_x)
    if not os.path.exists(file_path_y):
        os.mkdir(file_path_y)
    if not os.path.exists(file_path_z):
        os.mkdir(file_path_z)    

    temp_input256 = input256
    temp_input256 += ["id", "target"]
    for x in range(9): 
        list_x = []
        list_y = []
        list_z = []
        for k in range(len(id_mat)):
            temp_df_x = data_dict[str(id_mat[k]+'x')][temp_input256]
            temp_df_y = data_dict[str(id_mat[k]+'y')][temp_input256]
            list_x.append(temp_df_x.iloc[x].tolist())
            list_y.append(temp_df_y.iloc[x].tolist())
            if x < 5:
                temp_df_z = data_dict[str(id_mat[k]+'z')][temp_input256]
                list_z.append(temp_df_x.iloc[x].tolist())
                

        dfx = pd.DataFrame(list_x, columns = temp_input256)
        dfy = pd.DataFrame(list_y, columns = temp_input256)
        dfx.to_csv("{}/{}/data{}.csv".format(feat_path, path2[0], x+1), index=False) 
        dfy.to_csv("{}/{}/data{}.csv".format(feat_path, path2[1], x+1), index=False) 
            
            
        if x < 5:
            dfz = pd.DataFrame(list_z, columns = temp_input256)
            dfz.to_csv("{}/{}/data{}.csv".format(feat_path, path2[2], x+1), index=False) 


    
    # 3
    # 3.1 Create combined features (in the order of atom, bond and angle) from same checkpoints. Use first 512 features for atom+bond and all features for atom+bon+angle as input for model training
    comb_path = f"{feat_path}/xyz"
    if not os.path.exists(comb_path):
        os.mkdir(comb_path)
    feat_x_list = [f"{i}_x" for i in range(256)]
    feat_y_list = [f"{i}_y" for i in range(256)]
    feat_z_list = [f"{i}_z" for i in range(256)]

    data_order_768 = feat_x_list + feat_y_list + feat_z_list + ["id", "target"]
    data_order_512 = feat_x_list + feat_y_list + ["id", "target"]

    input768_main = [str(i) for i in range(768)] + ["id", "target"]
    input512_main = [str(i) for i in range(512)] + ["id", "target"]

    for i in range(1, 5+1):
        data_x = pd.read_csv("{}/x/data{}.csv".format(feat_path,i))
        data_y = pd.read_csv("{}/y/data{}.csv".format(feat_path,i))
        data_z = pd.read_csv("{}/y/data{}.csv".format(feat_path,i)).set_axis(feat_z_list + ["id", "target"], axis=1)
        
        data_xy = pd.merge(data_x, data_y, on=["id", "target"], suffixes=["_x", "_y"])
        data_xyz = pd.merge(data_xy, data_z, on=["id", "target"])
                        
        data_xyz = data_xyz[data_order_768].set_axis(input768_main, axis=1)

        data_xyz.to_csv("{}/data{}.csv".format(comb_path,i), index=False)

    for i in range(6, 9+1):
        data_x = pd.read_csv("{}/x/data{}.csv".format(feat_path,i))
        data_y = pd.read_csv("{}/y/data{}.csv".format(feat_path,i))
        
        data_xy = pd.merge(data_x, data_y, on=["id", "target"], suffixes=["_x", "_y"])
        data_xy = data_xy[data_order_512].set_axis(input512_main, axis=1)

        data_xy.to_csv("{}/data{}.csv".format(comb_path,i), index=False)

    # 3.2 Create combined features (in the order of atom, bond and angle) from different checkpoints
    atom_feat_num = 9  # Max number: 9
    bond_feat_num = 9  # Max number: 9
    angle_feat_num = 5 # Max number: 5
        
    data_x = pd.read_csv("{}/x/data{}.csv".format(feat_path,atom_feat_num))
    data_y = pd.read_csv("{}/y/data{}.csv".format(feat_path,bond_feat_num))
    data_z = pd.read_csv("{}/z//data{}.csv".format(feat_path,angle_feat_num)).set_axis(feat_z_list + ["id", "target"], axis=1)

    data_xy = pd.merge(data_x, data_y, on=["id", "target"])
    data_xyz = pd.merge(data_xy, data_z,  on=["id", "target"])
    data_xyz = data_xyz[data_order_768].set_axis(input768_main, axis=1)

    data_xyz.to_csv(f"{comb_path}/data_final_{atom_feat_num}_{bond_feat_num}_{angle_feat_num}.csv", index=False)



def split_combined_feat(
        dataset = ["matbench_jdft2d"],
        model_fold = 0,
        feat_dir = "../"
    ):
    """
    all_mb_sets = [
            "matbench_jdft2d",
            # "matbench_dielectric",
            # "matbench_phonons",
            # "matbench_log_gvrh",
            # "matbench_log_kvrh",
            # "matbench_perovskites",
            # "matbench_mp_e_form",
            # "matbench_mp_gap",
            # "matbench_mp_is_metal",
        ]
    """
    mb = MatbenchBenchmark(
        autoload=False,
        subset=dataset,
    )
    feat_path = os.path.join(feat_dir, f"embed_{dataset}/fold_{model_fold}")

    data_xyz = pd.read_csv(glob.glob(f"{feat_path}/xyz/data_final_*.csv")[0])

    for task in mb.tasks:
        task.load()
        for ii, fold in enumerate(task.folds):
            train_df = task.get_train_and_val_data(fold, as_type="df")
            test_df = task.get_test_data(fold, as_type="df")        
            val_df = train_df[0: len(test_df)]
            
            train_ids, val_ids, test_ids = list(train_df.index), list(val_df.index), list(test_df.index)
            
            data_xyz_train = data_xyz[data_xyz["id"].isin(train_ids)].reset_index(drop=True)
            data_xyz_val   = data_xyz[data_xyz["id"].isin(val_ids)].reset_index(drop=True)
            data_xyz_test  = data_xyz[data_xyz["id"].isin(test_ids)].reset_index(drop=True)
            
            save_dir = f"{feat_path}/xyz/split_fold_{ii}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            data_xyz_train.to_csv(f"{save_dir}/data_train.csv")
            data_xyz_val.to_csv(f"{save_dir}/data_val.csv")
            data_xyz_test.to_csv(f"{save_dir}/data_test.csv")

