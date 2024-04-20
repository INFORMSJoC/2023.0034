import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyIAAS import get_logger


class cnn_model(nn.Module):
    def __init__(self, layer_widths, activation, dropout_rate, input_shape):
        super().__init__()
        layers = []
        in_channels = input_shape[1]
        for width in layer_widths:
            layers.append(nn.Conv1d(in_channels, width, 3, padding=1))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_rate))
            in_channels = width
        self.cnn = nn.Sequential(*layers)
        input_sample = torch.zeros((1, *input_shape[::-1]))
        output = self.cnn(input_sample)
        dense_input = output.shape[1] * output.shape[2]
        self.regressor = nn.Sequential(nn.Linear(dense_input, 64),
                                       activation(),
                                       nn.Linear(64, 64),
                                       activation(),
                                       nn.Linear(64, 1),
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        x = torch.flatten(x)
        return x


use_GPU = True


def cnn_predict(data, parameters, save_file, log_file):
    X_train, y_train, X_test, y_test = data
    layer_widths = parameters[0]
    activation = parameters[1]
    batch_size = parameters[2]
    lr = parameters[3]
    dropout_rate = parameters[4]
    input_shape = X_train.shape[1:]
    assert len(layer_widths) > 0, 'layer width list should contains at least 1 element'
    for i in layer_widths:
        assert i > 0, 'layer width should larger than 0'
    cnn = cnn_model(layer_widths, activation, dropout_rate, input_shape)
    logger = get_logger('cnn', log_file, logging.INFO)
    # use gpu
    if use_GPU:
        cnn = cnn.cuda()
        X_train, y_train, X_test, y_test = X_train.cuda(), \
                                           y_train.cuda(), \
                                           X_test.cuda(), \
                                           y_test.cuda()
    # train model
    epoch = 500
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size,
                        shuffle=True)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_test = None
    for i in range(epoch):
        losses = []
        cnn.train()
        for index, (X, y) in enumerate(loader):
            y_pred = cnn(X)
            loss = loss_fn(y_pred, y) ** 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if i % 50 == 0:
            with torch.no_grad():
                cnn.eval()
                y_pred = cnn(X_test)
                loss = loss_fn(y_pred, y_test) ** 0.5
                if best_test is None or best_test > loss.item():
                    best_test = loss.item()
                    data = pd.DataFrame({'pred': y_pred.detach().cpu(), 'truth': y_test.detach().cpu()})
                    data.to_csv(f'{save_file}.csv')
                    logger.info(
                        f'cnn {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best')
                else:
                    logger.info(f'cnn {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_test
