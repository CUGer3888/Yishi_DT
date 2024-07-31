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

from sklearn.datasets import load_iris

import warnings

df = pd.read_csv("", encoding='gbk')
# 数据预处理,shuffle处理
new_df = df

new_df = shuffle(new_df)
print(new_df.head())

x = new_df[['map', 'used_time', 'last_time', 'last_num','isNow']]
y = new_df['isNext']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 数据类型转换
x_train_tensor = torch.from_numpy(x_train.to_numpy()).float()
y_train_tensor = torch.from_numpy(y_train.to_numpy()).float()

x_test_tensor = torch.from_numpy(x_test.to_numpy()).float()
y_test_tensor = torch.from_numpy(y_test.to_numpy()).float()

#
class Classifier(nn.Module):
    def __init__(self,features):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
#

clf = Classifier(5)

loss_func = nn.BCELoss()

optimizer = optim.Adam(clf.parameters(), lr=0.001)
#设置为CPU
device = torch.device("cpu")

x_train_tensor  = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

clf = clf.to(device)
loss_func = loss_func.to(device)
def accuracy(y_pred, y_true):

    pred_value = y_pred.ge(0.5).view(-1)
    acc = (pred_value == y_true).sum().float() /len(y_true)
    return acc
for epoch in range(1000):
    y_pred = clf(x_train_tensor)
    y_pred = torch.squeeze(y_pred)
    print(y_pred)
    print(y_train_tensor)
    train_loss = loss_func(y_pred, y_train_tensor)
    if epoch % 100 == 0:
        acc = accuracy(y_pred, y_train_tensor)


        y_test_pred = clf(x_test_tensor)
        y_test_pred = torch.squeeze(y_test_pred)
        test_loss = loss_func(y_test_pred, y_test_tensor)
        test_acc = accuracy(y_test_pred, y_test_tensor)

        print(f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Train Acc: {acc}, Test Loss: {test_loss.item()}, Test Acc: {test_acc}")
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


MODEL_PATh = "model.pth"
torch.save(clf, MODEL_PATh)
# test_data = torch.tensor([4,30,4,1,0],dtype=torch.float)
# test_data = test_data.to(device)
# with torch.no_grad():
#     pred = model(test_data)
#     print("预测结果：", pred.item())
# if pred.item() > 0.5:
#     print("是")
