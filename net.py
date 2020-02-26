import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

fin = h5py.File('../data/wave.h5', 'r')
ADC = fin['waveform']
TDC = fin['hit']

TRAIN_SAMPLES = 17613*18
VALID_SAMPLES = 17613
SPLIT_1 = TRAIN_SAMPLES
SPLIT_2 = TRAIN_SAMPLES+VALID_SAMPLES

x_train = torch.from_numpy(ADC[:SPLIT_1].reshape(TRAIN_SAMPLES, 1, 1000)).float()
y_train = torch.from_numpy(TDC[:SPLIT_1]).float()
x_valid = torch.from_numpy(ADC[SPLIT_1:SPLIT_2].reshape(VALID_SAMPLES, 1, 1000)).float()
y_valid = torch.from_numpy(TDC[SPLIT_1:SPLIT_2]).float()
x_test = torch.from_numpy(ADC[SPLIT_2:].reshape(VALID_SAMPLES, 1, 1000)).float()
y_test = torch.from_numpy(TDC[SPLIT_2:]).float()

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

fin.close()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

model = MLP().cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001) # start with large learning rate, for example 0.1, and then reduce the number by hand

def WLLoss(pred, label):
    pred = pred/(torch.sum(pred, dim=-1, keepdim=True)+1e-14)
    label = label/(torch.sum(label, dim=-1, keepdim=True)+1e-14)
    pred_sum = torch.cumsum(pred, 1)
    label_sum = torch.cumsum(label, 1)
    delta = (pred_sum - label_sum).abs()
    loss = delta.sum(1).mean()
    return loss

criterion = nn.MSELoss()


best_loss = 10000.
for epoch in range(60):
    start_time = time.time()
    running_loss = 0.0
    for i, (data, label) in enumerate(train_loader):
        data = data.float().cuda()
        label = label.cuda()

        out = model(data)
        loss = WLLoss(out, label)
#         loss = WLLoss(out, label)
        running_loss += loss.data.item()*label.size(0)

        if i%1000==0:
            print(loss.data.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        valid_loss = 0.0
        for (data, label) in valid_loader:
            data = data.float().cuda()
            label = label.cuda()
            out = model(data)
            loss = WLLoss(out, label)
#             loss = WLLoss(out, label)
            valid_loss += loss.data.item() * label.size(0)
    end_time = time.time()
    delta_time = end_time - start_time
    for p in optimizer.param_groups:
        print('learning rate is', p['lr'])
        if epoch%5==4: p['lr'] *= 0.8
#         p['lr'] *= 0.8
    print('Finish {} epoch, Train Loss: {:.6f}, Valid Loss: {:.6f}, Cost Time: {:.6f}s'.format(epoch+1, running_loss/(len(train_dataset)), valid_loss/(len(valid_dataset)), delta_time))
    cur_loss = valid_loss / (len(valid_dataset))
    if cur_loss < best_loss:
        best_model = copy.deepcopy(model)
        best_loss = cur_loss

torch.save(best_model.state_dict(), 'model_conv1d_WL.pt')
