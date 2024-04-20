import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyIAAS import get_logger


class resnet_model(nn.Module):
    def __init__(self, input_shape, layer_number, width, activation, dropout_rate):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        self.activate_fn = activation
        self.base = nn.Sequential(
            nn.Linear(self.input_size, width),
            self.activate_fn()
        )
        self.res_modules = [nn.Sequential(nn.Linear(width, width),
                                          self.activate_fn(),
                                          torch.nn.Dropout(p=dropout_rate),
                                          nn.Linear(width, width))
                            for i in range(layer_number)]
        for i in range(len(self.res_modules)):
            self.add_module(f'res{i}', self.res_modules[i])
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        x = x.reshape((-1, self.input_size))
        x = self.base(x)
        for block in self.res_modules:
            x_ = block(x)
            x = x + x_
        x = self.output(x)
        x = torch.flatten(x)
        return x


use_GPU = True


def resnet_predict(data, parameters, save_file, log_file):
    X_train, y_train, X_test, y_test = data
    layer = parameters[0]
    widths = parameters[1]
    activation = parameters[2]
    batch_size = parameters[3]
    lr = parameters[4]
    dropout_rate = parameters[5]
    input_shape = X_train.shape[1:]

    net = resnet_model(input_shape, layer, widths, activation, dropout_rate)
    logger = get_logger('res', log_file, logging.INFO)

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
                        batch_size=batch_size,
                        shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
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
                if best_test is None or best_test > loss.item():
                    best_test = loss.item()
                    data = pd.DataFrame({'pred': y_pred.detach().cpu(), 'truth': y_test.detach().cpu()})
                    data.to_csv(f'{save_file}.csv')
                    logger.info(
                        f'res {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best')
                else:
                    logger.info(f'res {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_test
