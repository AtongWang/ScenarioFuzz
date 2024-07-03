import os
import torch

from torch_geometric.loader import DataLoader

from data_load import EdgeBatchDataLoader
from eval_model import GNNBinaryClassificationModel, FocalLoss

import os
import torch

from torch_geometric.loader import DataLoader

from data_load import EdgeBatchDataLoader
from eval_model import GNNBinaryClassificationModel, FocalLoss,ImprovedGNNBinaryClassificationModel
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_recall_curve
from scipy.stats import pearsonr

def eval_model(predictions,actual_results):


    # 转换为numpy数组以便计算
    predictions = np.array(predictions)
    actual_results = np.array(actual_results)

    # 计算皮尔逊相关系数
    pearson_corr, _ = pearsonr(predictions, actual_results)
    print(f"Pearson correlation coefficient: {pearson_corr}")

    # 计算AUC-ROC
    roc_auc = roc_auc_score(actual_results, predictions)
    print(f"AUC-ROC: {roc_auc}")

    # 计算Brier分数
    brier_score = brier_score_loss(actual_results, predictions)
    print(f"Brier Score: {brier_score}")

    # 生成ROC曲线
    fpr, tpr, thresholds = roc_curve(actual_results, predictions)

    # 这里可以添加代码来绘制ROC曲线
    # 例如使用matplotlib
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # 随机概率的对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



mpl.rc('font', family='Arial') 
def evaluate_model(model, system_name_list, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    criterion = FocalLoss(gamma=2, alpha=[0.2, 0.8],device=device)

    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []

    for system_name in system_name_list:
        data_dir_path = f'/workspace3/eval_model/scenario_fuzz/{system_name}'
        test_file_list = [f for f in os.listdir(data_dir_path) if f.startswith('test_data')]
        test_file_name = test_file_list[-1]
        test_file = os.path.join(data_dir_path, test_file_name )
        
        if not os.path.exists(test_file):
            print(f"Test data for {system_name} not found.")
            continue

        test_data = torch.load(test_file)
        test_loader = EdgeBatchDataLoader(test_data, batch_size=1, shuffle=True)
        
        test_loss = 0
        correct = 0
        total = 0
        TP = FP = TN = FN = 0
        Y_pred = []
        Y_true = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(device)
                out = model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index, batch.batch, batch.edge_batch)
                target = batch.risk_opt.view(-1, 1)
                loss = criterion(out, target)
                avg_loss = loss / out.size(0)
                test_loss += avg_loss.item()
                predicted = torch.round(out)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                Y_pred.append(out.cpu().squeeze().numpy())
                Y_true.append(target.cpu().squeeze().numpy())
                TP += ((predicted == 1) & (target == 1)).sum().item()
                FP += ((predicted == 1) & (target == 0)).sum().item()
                TN += ((predicted == 0) & (target == 0)).sum().item()
                FN += ((predicted == 0) & (target == 1)).sum().item()
                
        test_loss /= len(test_loader)
        test_accuracy = correct / total

        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
        print(f'System: {system_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
    
    return test_losses, test_accuracies, test_precisions, test_recalls, Y_pred, Y_true

def plot_results_line(system_name_list, test_losses, test_accuracies, test_precisions, test_recalls, store_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    system_name_list = ['autoware', 'behavior', 'basic', 'LAV', 'Transfuser', 'all system']
    ax1.set_xlabel('System')
    ax1.set_ylabel('Loss / Accuracy', color='tab:blue')
    ax1.plot(system_name_list, test_losses, 'o-', color='tab:blue', label='Test Loss')
    ax1.plot(system_name_list, test_accuracies, 's-', color='tab:green', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelrotation=45)
    ax1.legend(loc='upper left', prop={'size': 12})
    ax2.set_xlabel('System')
    ax2.set_ylabel('Precision / Recall', color='tab:red')
    ax2.plot(system_name_list, test_precisions, 'o-', color='tab:orange', label='Precision')
    ax2.plot(system_name_list, test_recalls, 's-', color='tab:red', label='Recall')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelrotation=45)
    # set legend size
    ax2.legend(loc='upper left', prop={'size': 12})

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    figname = os.path.join(store_dir, 'eval_classfic_model.pdf')
    plt.savefig(figname)
    plt.show()

def plot_results_bar(system_name_list, test_losses, test_accuracies, test_precisions, test_recalls, store_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    system_name_list = ['Autoware', 'Behavior', 'Basic', 'LAV', 'Transfuser','NEAT','All system']
    width = 0.35  # the width of the bars
    ind = np.arange(len(system_name_list))  # the label locations

    # For ax1
    rects1 = ax1.bar(ind - width/2, test_losses, width, color='tab:blue', label='Test Loss')
    rects2 = ax1.bar(ind + width/2, test_accuracies, width, color='tab:green', label='Test Accuracy')
    ax1.set_xlabel('System')
    ax1.set_ylabel('Loss / Accuracy', color='tab:blue')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(system_name_list, rotation=45)
    ax1.legend(loc='upper left', prop={'size': 12})

    # For ax2
    rects3 = ax2.bar(ind - width/2, test_precisions, width, color='tab:orange', label='Precision')
    rects4 = ax2.bar(ind + width/2, test_recalls, width, color='tab:red', label='Recall')
    ax2.set_xlabel('System')
    ax2.set_ylabel('Precision / Recall', color='tab:red')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(system_name_list, rotation=45)
    ax2.legend(loc='upper left', prop={'size': 12})

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    figname = os.path.join(store_dir, f'eval_classfic_model_{MODEL_VERSION}.pdf')
    plt.savefig(figname)
    plt.show()

# Define model parameters
node_input_dim = 6
edge_input_dim = 3
weather_input_dim = 8
hidden_dim = 64
output_dim = 1
MODEL_TYPE = 1
MODEL_VERSION = 2
# Initialize model
if MODEL_TYPE ==0:

    model = GNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim)
elif MODEL_TYPE ==1:
    model = ImprovedGNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim,num_heads=4) 
#system_name_list = ['autoware', 'behavior', 'basic', 'leaderboard-LAV', 'leaderboard-NEAT','leaderboard-Transfuser', 'all_data']
system_name_list = ['autoware']
model_path = f'src/scenario_eval_model/model_dir/improve_v{MODEL_VERSION}/gnn_binary_classification_model_{MODEL_TYPE}.pt'

#store_dir = model_path.split(f'gnn_binary_classification_model_{MODEL_TYPE}.pt')[0]
store_dir ='/workspace3/improve_method/'
test_losses, test_accuracies, test_precisions, test_recalls,Y_pred,Y_true = evaluate_model(model, system_name_list, model_path)
eval_model(Y_pred,Y_true)
#print(test_losses, test_accuracies, test_precisions, test_recalls)
#plot_results_bar(system_name_list, test_losses, test_accuracies, test_precisions, test_recalls, store_dir)
