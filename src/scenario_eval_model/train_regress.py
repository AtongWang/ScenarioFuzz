import sys,os
sys.path.append('src/scenario_eval_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from data_load import  create_pyg_data,train_test_split,load_json_data,preprocess_json,EdgeBatchDataLoader
from eval_model import GNNRegressionModel
from torch_geometric.data import Data

# system_name_list = ['autoware','behavior','leaderboard-NEAT','basic','leaderboard-LAV', 'leaderboard-Transfuser','all_data']
system_name = 'all_data'
data_dir_path =f'/workspace2/scenario_fuzz/sceanrio_data/{system_name}'
#若train文件和test文件不存在，执行该函数。
train_file =os.path.join(data_dir_path,'train_data_reg.pt') 
test_file = os.path.join(data_dir_path,'test_data_reg.pt')
if not os.path.exists(train_file)or not os.path.exists(test_file ):
    json_data_list = load_json_data( data_dir_path)
    data_list = []
    for json_data in json_data_list:
        G,score = preprocess_json(json_data)
        if score == 0:
            continue
        data = create_pyg_data(G,score)
        data_list.append(data)
    # 转换数据并划分训练集和测试集,保存
    train_data, test_data = train_test_split(data_list)
    torch.save(train_data, train_file)
    torch.save(test_data, test_file)
else:
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)

# 定义模型参数
node_input_dim = 6
edge_input_dim = 3
weather_input_dim = 8
hidden_dim = 64
output_dim = 1

# 初始化模型
model = GNNRegressionModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim, output_dim)


# 创建数据加载器
train_loader = EdgeBatchDataLoader(train_data, batch_size=64, shuffle=True)
test_loader =EdgeBatchDataLoader(test_data, batch_size=64, shuffle=False)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index,batch.batch,batch.edge_batch)
        loss = criterion(out, batch.y)
        avg_loss = loss/ out.size(0)
        avg_loss.backward()
        optimizer.step()
        
    # 评估模型
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index,batch.batch,batch.edge_batch)
            loss = criterion(out, batch.y)
            avg_loss = loss/out.size(0)
            test_loss += avg_loss.item()
    test_loss /= len(test_loader)
    print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}')
