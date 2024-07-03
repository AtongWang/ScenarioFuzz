import os
import shutil
import torch
from data_load import create_pyg_data, train_test_split, load_json_data, preprocess_json
import random
import datetime,json



def dir_path2json(json_path):
    old_json = json.load(open(json_path))
    old_json['json_path'] = os.path.abspath(json_path)
    return old_json
    



def data_get(store_dir,source_dir,system_list):

    for system_name in system_list:
        if system_name.split('-')[0] == 'leaderboard' and system_name.split('-')[1].lower() == 'transfuser':
            target_folder  = f"{store_dir}/leaderboard-Transfuser"
        else:
            target_folder  = f"{store_dir}/{system_name}"
        workspace_path = f"{source_dir}/{system_name}"
        os.chdir(workspace_path)

        # 创建目标文件夹
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 收集所有out-artifact-*文件夹
        out_artifact_folders = [folder for folder in os.listdir() if folder.startswith("out-artifact-")]

        # 将所有JSON文件复制到目标文件夹
        #crash": true, "stuck": false, "lane_invasion": false, "red": false
        for folder in out_artifact_folders:
            scenario_data_path = os.path.join(folder, "scenario_data")
            if os.path.exists(scenario_data_path):
                json_files = [file for file in os.listdir(scenario_data_path) if file.endswith(".json")]
                for file in json_files:
                    src_path = os.path.join(scenario_data_path, file)
                    json_file = dir_path2json(src_path)
                    dst_path = os.path.join(target_folder, file)
                    with open(dst_path, 'w') as f:
                        json.dump(json_file, f)

        print(f"所有{workspace_path}内的JSON文件已成功复制到 {target_folder} 文件夹.")



def data_to_all_data(store_dir,system_list):
  
    target_folder_all = f"{store_dir}/all_data"
        # 创建目标文件夹
    if not os.path.exists(target_folder_all):
        os.makedirs(target_folder_all)
    # 生成所有数据集
    for system_name in system_list:
        if system_name.split('-')[0] == 'leaderboard' and system_name.split('-')[1].lower() == 'transfuser':
            workspace_path = f"{store_dir}/leaderboard-Transfuser"
        else:
            workspace_path  = f"{store_dir}/{system_name}"



        json_files = [file for file in os.listdir(workspace_path) if file.endswith(".json")]
        for file in json_files:
            src_path = os.path.join(workspace_path, file)
            dst_path = os.path.join(target_folder_all, file)
            shutil.copy(src_path, dst_path)

        print(f"所有{workspace_path}JSON文件已成功复制到 {target_folder_all} 文件夹.")
    print("所有数据集已生成。")


def data_to_all_data_part(store_dir,data_num,system_list):
    target_folder_all = f"{store_dir}/all_data_part"
        # 创建目标文件夹
    if not os.path.exists(target_folder_all):
        os.makedirs(target_folder_all)
    # 生成所有数据集
    for system_name in system_list:
        if system_name.split('-')[0] == 'leaderboard' and system_name.split('-')[1].lower() == 'transfuser':
            workspace_path = f"{store_dir}/leaderboard-Transfuser"
        else:
            workspace_path  = f"{store_dir}/{system_name}"
        # 将json_files 随机打乱
        
        json_files = [file for file in os.listdir(workspace_path) if file.endswith(".json")]
        random.shuffle(json_files)  
        for file in json_files[:data_num]:
            src_path = os.path.join(workspace_path, file)
            dst_path = os.path.join(target_folder_all, file)
            shutil.copy(src_path, dst_path)

        print(f"所有{workspace_path}JSON文件已成功复制到 {target_folder_all} 文件夹.")
    print("所有数据集已生成。")





def data_generate(store_dir,system_name_list):


    for system_name in system_name_list:
        data_dir_path =f'{store_dir}/{system_name}'
        #若train文件和test文件不存在，执行该函数。

        train_file = os.path.join(data_dir_path, f'train_data.pt') 
        test_file = os.path.join(data_dir_path, f'test_data.pt')

        json_data_list = load_json_data( data_dir_path)
        data_list = []
        for json_data in json_data_list:
            G,score,json_path,error_type,sem_predict = preprocess_json(json_data,only_crash=True) # type: ignore
            if G is None:
                continue
            data = create_pyg_data(G,score,json_path,error_type,sem_predict)
            data_list.append(data)
        # 转换数据并划分训练集和测试集,保存
        train_data, test_data = train_test_split(data_list, test_size=1.0)
        torch.save(train_data, train_file)
        torch.save(test_data, test_file)
    





if __name__ == "__main__":


    #system_list =['autoware', 'behavior', 'leaderboard-NEAT', 'basic', 'leaderboard-LAV', 'leaderboard-Transfuser']
    system_list = ['leaderboard-LAV']
    #system_list = ['behavior', 'leaderboard-NEAT', 'basic', 'leaderboard-LAV', 'leaderboard-Transfuser','leaderboard-LAV_V2','leaderboard-Transfuser_V2']
    # 指定工作目录
    ALL_DATA_LOAD = False
    DATA_GET = True
    DATA_GENERATE = True
    ALL_DATA_LOAD_PART = False


    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d%H")

    #store_dir = f"/workspace3/sceanrio_data/scenario_data_{date_time}_nofp"
    #store_dir ='/workspace3/sceanrio_data/scenario_data_2023062115/'
    store_dir ='/workspace3/sceanrio_data/scenario_data_2024030318_nofp'

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    #'scenario_fuzz_3','scenario_fuzz_4','scenario_fuzz_5',\
    #source_dir_list = ['scenario_fuzz_model_6','scenario_fuzz_model_5','scenario_fuzz_model_4']
    source_dir_list = ['scenario_fuzz_model_fn']
    #source_dir_list = ['scenario_fuzz_model']
    for dir in source_dir_list:
        source_dir = f"/workspace3/{dir}"
        if DATA_GET:
            data_get(store_dir,source_dir,system_list)
    if ALL_DATA_LOAD:
        data_to_all_data(store_dir,system_list)
    if ALL_DATA_LOAD_PART:
        data_to_all_data_part(store_dir,100,system_list)
    if DATA_GENERATE:
        data_generate(store_dir,system_list)