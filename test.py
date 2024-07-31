#测试程序
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from sklearn.datasets import load_iris

import warnings
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
model = torch.load("")
model.eval()
device = torch.device("cpu")
data_path = ""
data = pd.read_csv(data_path)
data = shuffle(data)
#进行预测
for i in tqdm(range(len(data))):
    x = torch.tensor(data.iloc[i, 1:6].values, dtype=torch.float32).unsqueeze(0).to(device)
    # 使用模型进行预测
    y_pred = model(x)
    # 打印预测结果,保留两位小数
    print(f"预测结果: {y_pred.item():.2f}")
    time.sleep(0.01)



