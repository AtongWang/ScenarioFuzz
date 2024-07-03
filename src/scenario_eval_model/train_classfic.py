import sys,os
sys.path.append('src/scenario_eval_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from data_load import create_pyg_data, train_test_split, load_json_data, preprocess_json, EdgeBatchDataLoader_new
from eval_model import GNNBinaryClassificationModel, FocalLoss,ImprovedGNNBinaryClassificationModel
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter  # 导入 SummaryWriter

import time



# system_name_list = ['autoware','behavior','leaderboard-NEAT','basic','leaderboard-LAV', 'leaderboard-Transfuser','all_data']
system_name = 'all_data'
#data_dir_path =f'/workspace2/scenario_fuzz/sceanrio_data/{system_name}'
data_dir_path =f'/workspace3/sceanrio_data/scenario_data_2023062115/{system_name}'
#若train文件和test文件不存在，执行该函数。
train_file =os.path.join(data_dir_path,'train_data.pt') 
test_file = os.path.join(data_dir_path,'test_data.pt')
if not os.path.exists(train_file)or not os.path.exists(test_file ):
    json_data_list = load_json_data( data_dir_path)
    data_list = []
    for json_data in json_data_list:
        G,score = preprocess_json(json_data)
        data = create_pyg_data(G,score)
        data_list.append(data)
    # 转换数据并划分训练集和测试集,保存
    train_data, test_data = train_test_split(data_list)
    torch.save(train_data, train_file)
    torch.save(test_data, test_file)
else:
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)



model_dir = '/workspace3/SEM_train/{}/'.format(system_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)



# 创建一个唯一的子目录
unique_subdir = "run_" + time.strftime("%Y%m%d-%H%M%S")

# 将唯一子目录添加到 model_dir
log_dir= os.path.join(model_dir, unique_subdir)

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建 SummaryWriter 对象
writer = SummaryWriter(log_dir=log_dir)

# 定义模型参数
node_input_dim = 6
edge_input_dim = 3
weather_input_dim = 8
hidden_dim = 64
output_dim = 1
MODEL_TYPE = 1  # 0: GNN, 1: Improved GNN
# 初始化模型
if MODEL_TYPE == 0:
    model = GNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim)
if MODEL_TYPE ==1:
    model = ImprovedGNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim,num_heads=4) 

model.to(device)  # 将模型移动到 GPU 上


# 创建数据加载器
train_loader = EdgeBatchDataLoader_new(train_data, batch_size=16, shuffle=True,normalize=True)
test_loader =EdgeBatchDataLoader_new(test_data, batch_size=16, shuffle=True,normalize=True)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
criterion = FocalLoss(gamma=2, alpha=[0.05,0.95],device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




## 训练模型
num_epochs =1000

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        batch.to(device)  # 将批处理数据移动到 GPU 上
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index, batch.batch, batch.edge_batch)
        target = batch.risk_opt.view(-1, 1)
        loss = criterion(out, target)
        avg_loss = loss / out.size(0)
        avg_loss.backward()
        optimizer.step()
        train_loss += avg_loss.item()
        predicted = torch.round(torch.sigmoid(out))
        total += target.size(0)
        correct += (predicted == target).sum().item()
    train_loss /= len(train_loader)
    train_accuracy = correct / total

    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)  # 将批处理数据移动到 GPU 上
            out = model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index, batch.batch, batch.edge_batch)
            target = batch.risk_opt.view(-1, 1)
            loss = criterion(out, target)
            avg_loss = loss / out.size(0)
            test_loss += avg_loss.item()
            predicted = torch.round(torch.sigmoid(out))

            y_true += batch.risk_opt.tolist()
            y_pred += predicted.view(-1).tolist()
            # 计算准确率、精确率和召回率
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    # 计算TP，FP，TN和FN
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]

    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    test_loss /= len(test_loader)


    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.4f}')

    # 记录训练损失、训练精度、测试损失和测试精度
    writer.add_scalar('ConfusionMatrix/TP', tp, epoch)
    writer.add_scalar('ConfusionMatrix/FP', fp, epoch)
    writer.add_scalar('ConfusionMatrix/TN', tn, epoch)
    writer.add_scalar('ConfusionMatrix/FN', fn, epoch)
    writer.add_scalar('Accuracy', acc, epoch)
    writer.add_scalar('Precision', prec, epoch)
    writer.add_scalar('Recall', rec, epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', acc,epoch)

# 保存模型
torch.save(model.state_dict(), os.path.join(log_dir,f'gnn_binary_classification_model_{MODEL_TYPE}.pt'))

# 关闭 SummaryWriter
writer.close()
