import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pyIAAS import get_logger

from compare import record_data

plt.ion()
plot = False # draw train process

class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            assert False

        return x


class MLP(nn.Module):

    def __init__(self, input_shape, layer, widths, mean_lr, gate_lr, scale_lr):
        super(MLP, self).__init__()
        self.input_size = input_shape[0] * input_shape[1]
        self.input_dim = input_shape[1]
        self.time_length = input_shape[0]
        self.activate_fn = nn.SELU
        self.base = nn.Sequential(
            nn.Linear(self.input_size, widths),
            self.activate_fn()
        )
        self.res_modules = [nn.Sequential(nn.Linear(widths, widths),
                                          self.activate_fn())
                            for i in range(layer)]
        for i in range(len(self.res_modules)):
            self.add_module(f'res{i}', self.res_modules[i])
        self.output = nn.Linear(widths, 1)

        self.dean = DAIN_Layer(mode='full', mean_lr=mean_lr, gate_lr=gate_lr, scale_lr=scale_lr, input_dim=self.time_length)

    def forward(self, x):
        # x = x.transpose(2,1)
        # Expecting  (n_samples, dim,  n_feature_vectors)
        x = self.dean(x)
        x = x.contiguous().view(x.size(0), self.input_size)
        x = self.base(x)
        for block in self.res_modules:
            x_ = block(x)
            x = x_
        x = self.output(x)
        x = torch.flatten(x)
        return x


use_GPU = True


def dain_predict(data, parameters, save_file, log_file):
    X_train, y_train, X_test, y_test = data
    layer = parameters[0]
    widths = parameters[1]
    mean_lr = parameters[2]
    gate_lr = parameters[3]
    scale_lr = parameters[4]
    input_shape = X_train.shape[1:]

    net = MLP(input_shape, layer, widths, mean_lr, gate_lr, scale_lr)
    logger = get_logger('dain', log_file, logging.INFO)

    # use gpu
    if use_GPU:
        net = net.cuda()
        X_train, y_train, X_test, y_test = X_train.cuda(), \
                                           y_train.cuda(), \
                                           X_test.cuda(), \
                                           y_test.cuda()
    # train model
    epoch = 500
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=128,
                        shuffle=True)
    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = nn.MSELoss()
    best_test = None
    for i in range(epoch):
        losses = []
        net.train()
        for index, (X, y) in enumerate(loader):
            y_pred = net(X)
            loss = loss_fn(y_pred, y) ** 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if i % 50 == 0:
            with torch.no_grad():
                net.eval()
                y_pred = net(X_test)
                loss = loss_fn(y_pred, y_test) ** 0.5
                data = pd.DataFrame({'pred': y_pred.view(-1).detach().cpu(), 'truth': y_test.view(-1).detach().cpu()})
                if plot:
                    plt.clf()
                    plt.plot(data)
                    plt.pause(0.01)
                if best_test is None or best_test > loss.item():
                    best_test = loss.item()
                    data.to_csv(f'{save_file}.csv')
                    logger.info(
                        f'dain {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best')
                else:
                    logger.info(f'dain {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_test
