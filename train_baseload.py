import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import train_sb3 as train
import datetime
import os
from hem.env import ami
import time


# 自定义Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dict, time_steps, scaler=None, mode='train'):
        self.time_steps = time_steps
        self.mode = mode

        # 合并特征
        features = pd.concat([
            data_dict['base_load'].rename('base_load'),  # 这里的rename是改变pandas.dataframe的列名，前面的引用是字典的key
            data_dict['month'].rename('month'),
            data_dict['day'].rename('day'),
            data_dict['hour'].rename('hour'),
            data_dict['minute'].rename('minute'),
            data_dict['day_of_week'].rename('day_of_week')
        ], axis=1)

        # 数据标准化
        if scaler is None:
            self.scaler = {}
            for i, col in enumerate(features.columns):
                scaler = StandardScaler()
                features[col] = scaler.fit_transform(features[[col]].values)
                self.scaler[col] = scaler
        else:
            self.scaler = scaler
            for i, col in enumerate(features.columns):
                features[col] = self.scaler[col].transform(features[[col]].values)

        # 创建序列数据
        X, y = self.create_sequences(features.values, features['base_load'].values)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def create_sequences(self, data, target):
        X = []
        y = []
        for i in range(self.time_steps, len(data)):
            X.append(data[i - self.time_steps:i, :])
            y.append(target[i])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 训练函数（添加设备支持）
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # 数据转移到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(train_loader.dataset)


# 评估函数（添加设备支持）
def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(eval_loader.dataset)


if __name__ == "__main__":
    # 参数配置
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    # 从环境加载数据
    t_data, e_data = train.load_train_eval_data(minutes_per_time_step=ami.PRE_USED_MINUTES_PER_TIME_STEP)

    # 设备检测
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 创建存储文件夹
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(f'./checkpoint/base_load_pre/{timestamp}', exist_ok=True)

    # 准备训练数据
    train_data = {
        'base_load': t_data.base_load,
        'month': t_data.month,
        'day': t_data.day,
        'hour': t_data.hour,
        'minute': t_data.minute,
        'day_of_week': t_data.day_type
    }

    # 准备评估数据（假设raw_eval_env有相同的结构）
    eval_data = {
        'base_load': e_data.base_load,
        'month': e_data.month,
        'day': e_data.day,
        'hour': e_data.hour,
        'minute': e_data.minute,
        'day_of_week': e_data.day_type
    }

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(train_data, ami.PRE_USED_TIME_STEPS)
    eval_dataset = TimeSeriesDataset(eval_data, ami.PRE_USED_TIME_STEPS, scaler=train_dataset.scaler, mode='eval')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型并转移到设备
    model = ami.TimeSeriesModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    eval_losses = []
    for epoch in range(NUM_EPOCHS):
        s_time = time.time()
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        eval_loss = evaluate_model(model, eval_loader, criterion, device)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Time: {time.time() - s_time:.2f}s')
        print(f'Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss:.6f}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            # 保存时移除设备依赖
            torch.save(model.state_dict(), f'./checkpoint/base_load_pre/{timestamp}/best_model.pth')

    # 最终评估（确保使用最佳模型）
    model.load_state_dict(torch.load(f'./checkpoint/base_load_pre/{timestamp}/best_model.pth', map_location=device))
    final_eval_loss = evaluate_model(model, eval_loader, criterion, device)
    print(f'Final Evaluation Loss: {final_eval_loss:.6f}')

    # 保存训练过程中的损失
    np.savez(f'./checkpoint/base_load_pre/{timestamp}/losses.npz', train_losses=train_losses,
             eval_losses=eval_losses)

    # 保存scaler到文件
    scaler_path = f'./checkpoint/base_load_pre/{timestamp}/scaler.npz'
    np.savez(scaler_path, scaler=train_dataset.scaler)
