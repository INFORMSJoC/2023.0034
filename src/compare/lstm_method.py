import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyIAAS import get_logger


class lstm_model(nn.Module):
    def __init__(self, layer_num, layer_widths, activation, dropout_rate, input_shape):
        super().__init__()
        in_channels = input_shape[1]
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=in_channels,
                            hidden_size=layer_widths,
                            dropout=dropout_rate,
                            num_layers=layer_num)
        self.activate = activation()
        self.regressor = nn.Sequential(nn.Linear(layer_widths * input_shape[0], layer_widths),
                                       activation(),
                                       nn.Linear(layer_widths, layer_widths),
                                       activation(),
                                       nn.Linear(layer_widths, 1),
                                       )

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = self.activate(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        x = torch.flatten(x)
        return x


use_GPU = True


def lstm_predict(data, parameters, save_file, log_file):
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
    lstm = lstm_model(layer_num, layer_widths, activation, dropout_rate, input_shape)
    logger = get_logger('lstm', log_file, logging.INFO)

    # use gpu
    if use_GPU:
        lstm = lstm.cuda()
        X_train, y_train, X_test, y_test = X_train.cuda(), \
                                           y_train.cuda(), \
                                           X_test.cuda(), \
                                           y_test.cuda()
    # train model
    epoch = 500
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size,
                        shuffle=True)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_test = None
    for i in range(epoch):
        losses = []
        lstm.train()
        for index, (X, y) in enumerate(loader):
            y_pred = lstm(X)
            loss = loss_fn(y_pred, y) ** 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if i % 50 == 0:
            with torch.no_grad():
                lstm.eval()
                y_pred = lstm(X_test)
                loss = loss_fn(y_pred, y_test) ** 0.5
                if best_test is None or best_test > loss.item():
                    best_test = loss.item()
                    data = pd.DataFrame({'pred': y_pred.detach().cpu(), 'truth': y_test.detach().cpu()})
                    data.to_csv(f'{save_file}.csv')
                    logger.info(
                        f'lstm {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best')
                else:
                    logger.info(f'lstm {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_test
