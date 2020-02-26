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

x_test = torch.from_numpy(ADC[SPLIT_2:].reshape(VALID_SAMPLES, 1, 1000)).float()
y_test = torch.from_numpy(TDC[SPLIT_2:]).float()

test_dataset = TensorDataset(x_test, y_test)

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
model.load_state_dict(torch.load('model_conv1d_WL.pt'))

from scipy import stats
nevt = len(test_dataset)

predict = np.zeros((nevt, 1000), dtype=np.float32)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()
threshold = 0.15
tstart = time.time()
# with torch.no_grad():
for (i,(pulse, eid_)) in enumerate(test_loader):
    if i%10000==0: print(i)
    pulse = pulse.cuda()
    out = model(pulse)
    predict[i] = out.cpu().detach().numpy()
#         predict[i] = out.detach().numpy()
    predict[i] = predict[i]

tend = time.time()
print(tend-tstart)

w_dists = np.zeros(nevt)

for i in range(nevt):
    if i%1000==0: print(i)
    idx_n = np.where(y_test[i][:800].numpy()>0)[0]
    w2 = y_test[i][idx_n]
    if(len(idx_n)==0):
        print(i)
        w_dists[i] = 0
        continue
    idx_p = np.where(predict[i][:800]>0.1)[0]
#     print(idx_n)
#     print(idx_p)
    w1 = predict[i][idx_p]
#     print(w1)
    if(len(w1)==0):
        idx_p = [295]
        w1 = [1]
#     w_dists[i] = stats.wasserstein_distance(idx_n, idx_p, v_weights=w1)
    w_dists[i] = stats.wasserstein_distance(idx_n, idx_p, u_weights=w2, v_weights=w1)
    if w_dists[i]>40:
        print(i, w_dists[i])

plt.hist(w_dists, bins=40, range=(0,40))
print(w_dists.mean(), w_dists.std())
plt.xlabel('Wasserstein distance', fontsize=14)
plt.ylabel('Number of channels', fontsize=14)
plt.text(20, 4000, "mean = %d"%w_dists.mean(), size = 15)
plt.text(20, 3000, "std = %d"%w_dists.std(), size = 15)
