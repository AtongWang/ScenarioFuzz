import sys,os
sys.path.extend(['src','scenario_select'])
import scontant as S
import constants as C
import networkx as nx
import torch
from torch_geometric.data import Data,Batch
import random,json
import torch
from torch_geometric.loader import DataLoader



        
weather_list= ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']
node_attr_list = ['pos_x','pos_y','pos_z','type','mark_type','actor_name']
edge_attr_list = ['distance','type','way']
puddle_attr_list = ['level', 'x_loc_size', 'y_loc_size', 'z_loc_size', 'x_size', 'y_size', 'z_size']


class EdgeBatchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, normalize=True, **kwargs):
        super(EdgeBatchDataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)
        self.normalize = normalize

        if normalize:
            # 计算位置特征的均值和标准差
            node_positions = [data.x[:, :3] for data in self.dataset]
            node_positions = torch.cat(node_positions, dim=0)
            self.position_mean = node_positions.mean(dim=0)
            self.position_std = node_positions.std(dim=0)

            # 计算距离特征的最小值和范围
            edge_distances = [data.edge_attr[:, 0] for data in self.dataset]
            edge_distances = torch.cat(edge_distances, dim=0)
            self.distance_min = edge_distances.min()
            self.distance_range = edge_distances.max() - self.distance_min

            # 计算天气特征的均值和标准差
            weather_attrs = [data.weather_attr for data in self.dataset]
            weather_attrs = torch.cat(weather_attrs, dim=0)
            self.weather_mean = weather_attrs.mean(dim=0)
            self.weather_std = weather_attrs.std(dim=0)

    def __iter__(self):
        for batch in super().__iter__():
            # 检查 edge_index 的数据类型是否为长整型（LongTensor）
            if not batch.edge_index.dtype == torch.long:
                batch.edge_index = batch.edge_index.long()

            # 为每个边计算它属于哪个图
            edge_batch = []
            cum_edges = 0
            for i, data in enumerate(batch.to_data_list()):
                num_edges = data.edge_index.size(1)
                edge_batch.append(torch.full((num_edges,), i, dtype=torch.long))
                cum_edges += num_edges
                
            # 将 edge_batch 属性添加到 Batch 对象中
            batch.edge_batch = torch.cat(edge_batch, dim=0)

            # 标准化和归一化输入数据
            if self.normalize:
                # 标准化位置特征
                node_positions = batch.x[:, :3]
                node_positions = (node_positions - self.position_mean) / self.position_std
                batch.x[:, :3] = node_positions

                # 归一化距离特征
                edge_distances = batch.edge_attr[:, 0]
                edge_distances = (edge_distances - self.distance_min) / self.distance_range
                batch.edge_attr[:, 1] = edge_distances  # 注意这里的索引应该是0，如果您是想修改距离特征的话,还是有问题，暂时按1处理

                # 标准化天气特征
                weather_attrs = batch.weather_attr
                weather_attrs = (weather_attrs - self.weather_mean) / self.weather_std
                batch.weather_attr = weather_attrs

            yield batch

class EdgeBatchDataLoader_new(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, normalize=True, **kwargs):
        super(EdgeBatchDataLoader_new, self).__init__(dataset, batch_size, shuffle, **kwargs)
        self.normalize = normalize
        self.normalize_file = 'src/scenario_eval_model/model_dir/normalize.pt'
        if normalize:
            if os.path.exists(self.normalize_file):
                # 如果存在归一化文件，则加载它
                normalize_data = torch.load(self.normalize_file)
                self.position_mean = normalize_data['position_mean']
                self.position_std = normalize_data['position_std']
                self.distance_min = normalize_data['distance_min']
                self.distance_range = normalize_data['distance_range']
                self.weather_mean = normalize_data['weather_mean']
                self.weather_std = normalize_data['weather_std']
            else:
                # 计算位置特征的均值和标准差
                node_positions = [data.x[:, :3] for data in self.dataset]
                node_positions = torch.cat(node_positions, dim=0)
                self.position_mean = node_positions.mean(dim=0)
                self.position_std = node_positions.std(dim=0)

                # 计算距离特征的最小值和范围
                edge_distances = [data.edge_attr[:, 0] for data in self.dataset]
                edge_distances = torch.cat(edge_distances, dim=0)
                self.distance_min = edge_distances.min()
                self.distance_range = edge_distances.max() - self.distance_min

                # 计算天气特征的均值和标准差
                weather_attrs = [data.weather_attr for data in self.dataset]
                weather_attrs = torch.cat(weather_attrs, dim=0)
                self.weather_mean = weather_attrs.mean(dim=0)
                self.weather_std = weather_attrs.std(dim=0)

                # 保存归一化文件
                normalize_data = {
                    'position_mean': self.position_mean,
                    'position_std': self.position_std,
                    'distance_min': self.distance_min,
                    'distance_range': self.distance_range,
                    'weather_mean': self.weather_mean,
                    'weather_std': self.weather_std
                }

                print('Saving normalization data to', self.normalize_file)
                torch.save(normalize_data, self.normalize_file)
                print(f'position_mean: {self.position_mean}, position_std: {self.position_std},\
                        distance_min: {self.distance_min}, distance_range: {self.distance_range},\
                        weather_mean: {self.weather_mean}, weather_std: {self.weather_std}')

    def __iter__(self):
        for batch in super().__iter__():
            # 检查 edge_index 的数据类型是否为长整型（LongTensor）
            if not batch.edge_index.dtype == torch.long:
                batch.edge_index = batch.edge_index.long()

            # 为每个边计算它属于哪个图
            edge_batch = []
            cum_edges = 0
            for i, data in enumerate(batch.to_data_list()):
                num_edges = data.edge_index.size(1)
                edge_batch.append(torch.full((num_edges,), i, dtype=torch.long))
                cum_edges += num_edges
                
            # 将 edge_batch 属性添加到 Batch 对象中
            batch.edge_batch = torch.cat(edge_batch, dim=0)

            # 标准化和归一化输入数据
            if self.normalize:
                # 标准化位置特征
                node_positions = batch.x[:, :3]
                node_positions = (node_positions - self.position_mean) / self.position_std
                batch.x[:, :3] = node_positions

                # 归一化距离特征
                edge_distances = batch.edge_attr[:, 0]
                edge_distances = (edge_distances - self.distance_min) / self.distance_range
                batch.edge_attr[:, 0] = edge_distances  # 注意这里的索引应该是0，如果您是想修改距离特征的话,还是有问题，暂时按1处理

                # 标准化天气特征
                weather_attrs = batch.weather_attr
                weather_attrs = (weather_attrs - self.weather_mean) / self.weather_std
                batch.weather_attr = weather_attrs

            yield batch



# class EdgeBatchDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
#         super(EdgeBatchDataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)

#     def __iter__(self):
#         for batch in super().__iter__():
#             # 为每个边计算它属于哪个图
#             edge_batch = []
#             cum_edges = 0
#             for i, data in enumerate(batch.to_data_list()):
#                 num_edges = data.edge_index.size(1)
#                 edge_batch.append(torch.full((num_edges,), i, dtype=torch.long))
#                 cum_edges += num_edges

#             # 将 edge_batch 属性添加到 Batch 对象中
#             batch.edge_batch = torch.cat(edge_batch, dim=0)

#             yield batch

# 使用自定义的 EdgeBatchDataLoader





def preprocess_json(json_data,train=True,only_crash=False):
    weather_list= ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']
    # 提取天气参数
    
    G_data = json_data['data']
    if 'weather_attr' not in G_data:
        value = json_data['value']
        G_data['weather_attr'] = [value[weather] for weather in weather_list]

    if train:
        if not only_crash:
            risk_option = 1 if any(json_data['error'].values()) else 0
        else:
            risk_option = 1 if json_data['error']['crash'] else 0
        #crash": true, "stuck": false, "lane_invasion": false, "red": false, "speeding": false
        error_type = [k for k in ['crash', 'stuck', 'lane_invasion', 'red', 'speeding'
                                  ] if json_data['error'][k]] 
        if json_data['error']['other'] is not False:
            error_type.extend([json_data['error']['other']])
        score = [json_data['score'],risk_option]        
    else:
        score = [None,None]
        error_type = None
    
    if 'json_path' in json_data:
        json_path = json_data['json_path']
    else:
        json_path = None
    

    if 'predict_score' in json_data:
        predict_score = json_data['predict_score']
    else:
        predict_score = None
    if 'predict_label' in json_data:
        predict_label = json_data['predict_label']
    else:
        predict_label = None
    sem_predict = [predict_score,predict_label]
    return G_data,score,json_path,error_type,sem_predict
    
# 一个批量读取指定文件夹内所有的json文件的函数
def load_json_data(json_dir):
    json_data_list = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_data = json.load(open(os.path.join(json_dir, json_file)))
            json_data_list.append(json_data)
    return json_data_list    





def create_pyg_data(G_data, score=None, json_path=None,error_type = None,sem_predict = None):
      # 从 JSON 中提取数据
    node_features =  torch.tensor(G_data['x'], dtype=torch.float)
    edge_index = torch.tensor(G_data['edge_index'], dtype=torch.long)
    # edge_features 0,1维 倒过来
    edge_features = torch.tensor(G_data['edge_attr'], dtype=torch.float)
    # 如果edge 第2维的维度不是3，就转置
    if edge_features.size(1) != 3:
        edge_features = edge_features.transpose(0, 1)
    weather_features = torch.tensor(G_data['weather_attr'], dtype=torch.float)  # 假设是全局属性
    # 判断weather_features是否是二维的
    if len(weather_features.size())  != 2:
        weather_features = weather_features.reshape(1,-1)


    if score is not None:
        score_r = torch.tensor(score[0], dtype=torch.float)
        risk_opt = torch.tensor(score[1], dtype=torch.float)
        data = Data(x=node_features, edge_attr=edge_features, edge_index=edge_index, weather_attr=weather_features, y=score_r, risk_opt=risk_opt)
    else:
        data = Data(x=node_features, edge_attr=edge_features, edge_index=edge_index, weather_attr=weather_features)

    # Add json_path to Data if it's provided
    if json_path is not None:
        data.json_path = json_path
    if error_type is not None:
        data.error_type = error_type
    
    if sem_predict is not None:
        data.predict_score = sem_predict[0]
        data.predict_label = sem_predict[1]

    return data

def  train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42):
    if shuffle:
        random.seed(random_state)
        random.shuffle(data_list)
    test_size = int(len(data_list) * test_size)
    test_data = data_list[:test_size]
    train_data = data_list[test_size:]
    
    return train_data, test_data
    

# if __name__ ==  '__main__' :
#     pass
    # data_dir_path ='/workspace2/scenario_fuzz/leaderboard-LAV/out-artifact-2023-04-17-15-36/scenario_data'
    # json_data_list = load_json_data( data_dir_path)
    # data_list = []
    # for json_data in json_data_list:
    #     G,score = preprocess_json(json_data)
    #     data = create_pyg_data(G,score)
    #     data_list.append(data)
    # train_data, test_data = train_test_split(data_list)
    # torch.save(train_data, 'train_data.pt')
    # torch.save(test_data, 'test_data.pt')