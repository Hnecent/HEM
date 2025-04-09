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
    def __init__(self, data_dict, time_steps,
                 scaler_num=None, scaler_time=None, mode='train'):
        self.time_steps = time_steps
        self.mode = mode

        # 分离数值特征和时间特征（确保数值特征为二维DataFrame）
        num_features = data_dict['base_load'].to_frame(name='base_load')  # 转换为DataFrame
        time_features = pd.concat([
            data_dict['month'].rename('month'),
            data_dict['day'].rename('day'),
            data_dict['hour'].rename('hour'),
            data_dict['minute'].rename('minute'),
            data_dict['day_of_week'].rename('day_of_week')
        ], axis=1)

        # 标准化数值特征
        if scaler_num is None:
            self.scaler_num = StandardScaler()
            num_features = self.scaler_num.fit_transform(num_features.values)
        else:
            self.scaler_num = scaler_num
            num_features = self.scaler_num.transform(num_features.values)

        # 标准化时间特征
        if scaler_time is None:
            self.scaler_time = {}
            time_features = time_features.copy()
            for col in time_features.columns:
                scaler = StandardScaler()
                time_features[col] = scaler.fit_transform(time_features[[col]].values)
                self.scaler_time[col] = scaler
        else:
            self.scaler_time = scaler_time
            time_features = time_features.copy()
            for col in time_features.columns:
                time_features[col] = self.scaler_time[col].transform(time_features[[col]].values)

        # 创建序列数据
        X_num, X_time, y = self.create_sequences(num_features, time_features.values, num_features[:, 0])
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_time = torch.tensor(X_time, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def create_sequences(self, data_num, data_time, target):
        X_num, X_time, y = [], [], []
        for i in range(self.time_steps, len(data_num)):
            X_num.append(data_num[i - self.time_steps:i, :])
            X_time.append(data_time[i - self.time_steps:i, :])
            y.append(target[i])
        return np.array(X_num), np.array(X_time), np.array(y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_time[idx]), self.y[idx]

    def __len__(self):  # 关键修复：添加长度方法
        return len(self.y)  # 直接返回标签的数量


# 训练函数（添加设备支持）
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for (inputs_enc, inputs_mark), targets in train_loader:
        inputs_enc, inputs_mark, targets = inputs_enc.to(device), inputs_mark.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs_enc, inputs_mark, None, None)  # 确保模型支持多输入
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs_enc.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (inputs_enc, inputs_mark), targets in eval_loader:
            inputs_enc, inputs_mark, targets = inputs_enc.to(device), inputs_mark.to(device), targets.to(device)
            outputs = model(inputs_enc, inputs_mark, None, None)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item() * inputs_enc.size(0)
    return total_loss / len(eval_loader.dataset)


# 定义配置参数（修正enc_in为1）
class Config:
    def __init__(self):
        self.seq_len = ami.PRE_USED_TIME_STEPS  # 输入时间步长度
        self.label_len = 0  # 添加此行，设置为0（根据任务需求调整）
        self.pred_len = 1                      # 预测步长（单步预测）
        self.task_name = 'short_term_forecast' # 任务类型
        self.enc_in = 1                        # 输入特征维度（单变量）
        self.d_model = 512                     # 模型隐藏层维度
        self.embed = 'timeF'                   # 时间嵌入方式
        self.freq = 'h'                        # 时间频率（假设小时级）
        self.dropout = 0.1                     # dropout概率
        self.e_layers = 3                      # TimesNet层数
        self.top_k = 5                         # 傅里叶分析选择的周期数
        self.num_kernels = 6                   # Inception Block的核数
        self.c_out = 1                         # 输出维度（预测值）
        self.num_class = 0                     # 分类任务参数（无需）
        self.d_ff = 2048                       # 前馈层维度



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

    # 准备训练数据（确保字典键与Dataset处理一致）
    train_data = {
        'base_load': t_data.base_load,
        'month': t_data.month,
        'day': t_data.day,
        'hour': t_data.hour,
        'minute': t_data.minute,
        'day_of_week': t_data.day_type
    }

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
    eval_dataset = TimeSeriesDataset(
        eval_data,
        ami.PRE_USED_TIME_STEPS,
        scaler_num=train_dataset.scaler_num,
        scaler_time=train_dataset.scaler_time,
        mode='eval'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型并转移到设备
    config = Config()
    model = ami.TimeSeriesModel(config).to(device)

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
            torch.save(model.state_dict(), f'./checkpoint/base_load_pre/{timestamp}/best_model.pth')

    # 最终评估
    model.load_state_dict(torch.load(f'./checkpoint/base_load_pre/{timestamp}/best_model.pth', map_location=device))
    final_eval_loss = evaluate_model(model, eval_loader, criterion, device)
    print(f'Final Evaluation Loss: {final_eval_loss:.6f}')

    # 保存训练过程中的损失和标准化器
    np.savez(f'./checkpoint/base_load_pre/{timestamp}/losses.npz',
             train_losses=train_losses, eval_losses=eval_losses)
    np.savez(f'./checkpoint/base_load_pre/{timestamp}/scaler.npz',
             scaler_num_mean=train_dataset.scaler_num.mean_,
             scaler_num_scale=train_dataset.scaler_num.scale_,
             scaler_time={k: (v.mean_, v.scale_) for k, v in train_dataset.scaler_time.items()})
