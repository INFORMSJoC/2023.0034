import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyIAAS import get_logger


class cnn_lstm_model(nn.Module):
    def __init__(self, layer_num, layer_widths, activation, dropout_rate, input_shape):
        super().__init__()
        cnn_layers = []
        in_channels = input_shape[1]
        cnn_layer_num = layer_num // 2
        lstm_layer_num = layer_num - cnn_layer_num
        for i in range(cnn_layer_num):
            cnn_layers.append(nn.Conv1d(in_channels, layer_widths, 3, padding=1))
            cnn_layers.append(activation())
            cnn_layers.append(nn.Dropout(p=dropout_rate))
            in_channels = layer_widths
        self.cnn = nn.Sequential(*cnn_layers)
        input_sample = torch.zeros((1, *input_shape[::-1]))
        output = self.cnn(input_sample)
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=output.shape[1],
                            hidden_size=layer_widths,
                            dropout=dropout_rate,
                            num_layers=lstm_layer_num)
        dense_input = output.shape[1] * output.shape[2]
        self.regressor = nn.Sequential(nn.Linear(dense_input, layer_widths),
                                       activation(),
                                       nn.Linear(layer_widths, layer_widths),
                                       activation(),
                                       nn.Linear(layer_widths, 1),
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        x = torch.flatten(x)
        return x


use_GPU = True


def cnn_lstm_predict(data, parameters, save_file, log_file):
    X_train, y_train, X_test, y_test = data
    layer_num = parameters[0]
    layer_widths = parameters[1]
    activation = parameters[2]
    batch_size = parameters[3]
    lr = parameters[4]
    dropout_rate = parameters[5]
    input_shape = X_train.shape[1:]
    assert layer_widths > 0, 'layer width should greater than 0'
    assert layer_num > 0, 'layer number should greater than 0'
    net = cnn_lstm_model(layer_num, layer_widths, activation, dropout_rate, input_shape)
    logger = get_logger('cnn_lstm', log_file, logging.INFO)

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
                        f'{save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best')
                else:
                    logger.info(f'{save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_test
