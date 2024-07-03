import torch,os
import torch.nn as nn
import numpy as np


# 获取环境变量
STORE_DIR = os.environ.get('STORE_DIR')

# 检查变量是否存在
if STORE_DIR is not None:
    print(f"STORE_DIR: {STORE_DIR}")
else:
    print("STORE_DIR is not set")

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.betas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centers, 0, 1)
        nn.init.constant_(self.betas, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).sqrt()
        return torch.exp(-self.betas.unsqueeze(0) * distances)

# 初始化中心点的函数可以保留原样，只是改为使用 PyTorch 张量
class InitCentersRandom:
    def __init__(self, X):
        self.X = X

    def __call__(self, shape):
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return torch.from_numpy(self.X[idx, :]).float()


class RBFNetwork(nn.Module):
    def __init__(self, input_dim, rbf_units, output_dim):
        super(RBFNetwork, self).__init__()
        self.rbf_layer = RBFLayer(input_dim, rbf_units)
        self.linear = nn.Linear(rbf_units, output_dim)
    
    def forward(self, x):
        x = self.rbf_layer(x)
        x = self.linear(x)
        return x

class Model:
    def __init__(self, no_of_neurons, cluster):
        self.train(no_of_neurons, np.array(cluster))

    def train(self, no_of_neurons, cluster):
        X = torch.from_numpy(cluster[:, 0:11]).float()
        y = torch.from_numpy(cluster[:, 11:12]).float()
        y[y < 0] = 0
        y[y > 1] = 1

        # 模型和优化器
        self.model = RBFNetwork(11, no_of_neurons, 1)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        # 训练过程
        for epoch in range(1000):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 保存模型
        model_file = os.path.join(STORE_DIR, 'model.pth')
        torch.save(self.model.state_dict(), model_file)

    def predict(self, val):
        model_file = os.path.join(STORE_DIR, 'model.pth')
        self.model.load_state_dict(torch.load(model_file))  # 加载模型
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():
            value = torch.from_numpy(np.array([val])).float()
            y_pred = self.model(value)
            y_pred = y_pred.numpy().squeeze()
            return np.clip(y_pred, 0, 1)  # 将预测值限制在 0 和 1 之间

    def test(self, cluster):
        mae = 0
        for i in range(len(cluster)):
            y_act = cluster[i][11]
            Y_pred = self.predict(cluster[i][:11])
            if y_act > 1:
                y_act =1
            if y_act < 0:
                y_act =0
            mae = mae + abs(y_act - Y_pred)
        self.mae = mae / len(cluster)
