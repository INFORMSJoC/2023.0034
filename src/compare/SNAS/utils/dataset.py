import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from setting import data_path


class Scaler:
    def __init__(self, data, missing_value=np.inf):
        values = data[data != missing_value]
        self.mean = values.mean()
        self.std = values.std()

    def transform(self, data):
        return data
        # return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data
        # return data * self.std + self.mean


class TrafficDataset:
    def __init__(self, path, train_prop, valid_prop,
                 num_nodes, in_length, out_length,
                 batch_size_per_gpu, num_gpus, device):
        logging.info('initialize %s DataWrapper', path)
        self._path = path
        self._train_prop = train_prop
        self._valid_prop = valid_prop
        self._num_nodes = num_nodes
        self._in_length = in_length
        self._out_length = out_length
        self._batch_size_per_gpu = batch_size_per_gpu
        self._num_gpus = num_gpus

        self.device = device

        # self.build_graph()
        self.build_data_loader()

    def build_data_loader(self):
        logging.info('initialize data loader')
        train, valid, test = self.read_traffic()
        self.scaler = Scaler(train.values, missing_value=0)
        # data for search
        self.search_train = self.get_data_loader(train, shuffle=True, tag='search train',
                                                 num_gpus=self._num_gpus)  # for weight update
        self.search_valid = self.get_data_loader(valid, shuffle=True, tag='search valid',
                                                 num_gpus=self._num_gpus)  # for arch update
        # data for training & evaluation
        self.train = self.get_data_loader(train, shuffle=True, tag='train', num_gpus=1)
        self.valid = self.get_data_loader(valid, shuffle=False, tag='valid', num_gpus=1)
        self.test = self.get_data_loader(test, shuffle=False, tag='test', num_gpus=1)

    def get_data_loader(self, data, shuffle, tag, num_gpus):
        logging.info('load %s inputs & labels', tag)

        num_timestamps = data.shape[0]

        # fill missing value
        data_f = self.fill_traffic(data)

        # transform data distribution
        data_f = np.expand_dims(self.scaler.transform(data_f.values), axis=-1)  # [T, N, 1]

        # time in day
        time_ft = (data.index.values - data.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_ft = np.tile(time_ft, [1, self._num_nodes, 1]).transpose((2, 1, 0))  # [T, N, 1]

        # day in week
        # day_ft = np.zeros(shape=(num_timestamps, self._num_nodes, 7)) # [T, N, 7]
        # day_ft[np.arange(num_timestamps), :, data.index.dayofweek] = 1

        # put all input features together
        in_data = np.concatenate([data_f, time_ft, ], axis=-1)  # [T, N, D]
        out_data = np.expand_dims(data.values, axis=-1)  # [T, N, 1]

        # create inputs & labels
        inputs, labels = [], []
        for i in range(self._in_length):
            inputs += [in_data[i: num_timestamps + 1 - self._in_length - self._out_length + i]]
        for i in range(self._out_length):
            labels += [out_data[self._in_length + i: num_timestamps + 1 - self._out_length + i]]
        inputs = np.stack(inputs).transpose((1, 3, 2, 0))
        labels = np.stack(labels).transpose((1, 3, 2, 0))

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]

        # create dataset
        dataset = TensorDataset(
            torch.from_numpy(inputs).to(dtype=torch.float),
            torch.from_numpy(labels).to(dtype=torch.float)
        )

        # create sampler
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=self._batch_size_per_gpu * num_gpus) if shuffle else SequentialSampler(
            dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size_per_gpu * num_gpus, sampler=sampler,
                                 num_workers=4, drop_last=False)
        return data_loader

    def read_traffic(self):
        data = pd.read_hdf(os.path.join(data_path, self._path, 'pems-bay.h5'))
        num_train = int(data.shape[0] * self._train_prop)
        num_eval = int(data.shape[0] * self._valid_prop)
        num_test = data.shape[0] - num_train - num_eval
        return data[:num_train].copy(), data[num_train: num_train + num_eval].copy(), data[-num_test:].copy()

    def fill_traffic(self, data):
        data = data.copy()
        data[data < 1e-5] = float('nan')
        data = data.fillna(method='pad')
        data = data.fillna(method='bfill')
        return data

    @property
    def batch_size_per_gpu(self):
        return self._batch_size_per_gpu


class MyDataset(TrafficDataset):

    def __init__(self, path, train_prop, valid_prop, num_nodes, in_length, out_length, batch_size_per_gpu, num_gpus,
                 device, season):
        self._season = season
        super().__init__(path, train_prop, valid_prop, num_nodes, in_length, out_length, batch_size_per_gpu, num_gpus,
                         device)

    def build_data_loader(self):
        logging.info('initialize data loader')
        train, test = self.read_data()
        self.scaler = Scaler(train[1])
        # data for search
        self.search_train = self.get_data_loader(train, shuffle=True, tag='search train',
                                                 num_gpus=self._num_gpus)  # for weight update
        self.search_valid = self.get_data_loader(test, shuffle=True, tag='search valid',
                                                 num_gpus=self._num_gpus)  # for arch update
        # data for training & evaluation
        self.train = self.get_data_loader(train, shuffle=True, tag='train', num_gpus=1)
        self.valid = self.get_data_loader(test, shuffle=False, tag='valid', num_gpus=1)
        self.test = self.get_data_loader(test, shuffle=False, tag='test', num_gpus=1)

    def get_data_loader(self, data, shuffle, tag, num_gpus):
        logging.info('load %s inputs & labels', tag)
        inputs = data[0].unsqueeze(dim=2)
        labels = data[1].reshape(-1, 1, 1, 1)

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]
        # create dataset
        dataset = TensorDataset(
            inputs.to(dtype=torch.float),
            labels.to(dtype=torch.float)
        )
        # create sampler
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=self._batch_size_per_gpu * num_gpus) if shuffle else SequentialSampler(
            dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size_per_gpu * num_gpus, sampler=sampler,
                                 num_workers=4, drop_last=False)
        return data_loader


    def read_data(self):
        def four_season_train_test_split(X: torch.Tensor, y: torch.Tensor):
            def train_test_fix_test_length_spilt(X: torch.Tensor, y: torch.Tensor, test_length):
                length = X.shape[0]
                train_length = length - test_length
                X_train, y_train, X_test, y_test = X[:train_length], y[:train_length], X[train_length:], y[
                                                                                                         train_length:]
                return (X_train, y_train), (X_test, y_test)

            spring_end = 24 * (31 + 29 + 31)
            summer_end = 24 * (30 + 31 + 30) + spring_end
            autumn_end = 24 * (31 + 31 + 30) + summer_end
            test_length = 24 * 5
            spring = train_test_fix_test_length_spilt(X[:spring_end], y[:spring_end], test_length)
            summer = train_test_fix_test_length_spilt(X[spring_end: summer_end], y[spring_end: summer_end], test_length)
            autumn = train_test_fix_test_length_spilt(X[summer_end: autumn_end], y[summer_end: autumn_end], test_length)
            winter = train_test_fix_test_length_spilt(X[autumn_end:], y[autumn_end:], test_length)
            return spring, summer, autumn, winter

        with open(self._path, 'rb') as f:
            data = pickle.load(f)
            X = data['X']
            y = data['y']
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        y = y.reshape((-1, 1))
        X = X.permute(0, 2, 1)
        spring, summer, autumn, winter = four_season_train_test_split(X, y)
        if self._season == 'spring':
            return spring
        if self._season == 'summer':
            return summer
        if self._season == 'autumn':
            return autumn
        if self._season == 'winter':
            return winter