import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from compare.SNAS.setting import config as cfg
from compare.SNAS.model.net import Net
from compare.SNAS.run_model import RunModel

from pyIAAS import get_logger


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


class Dataset:
    def __init__(self, data, train_prop, valid_prop,
                 num_nodes, in_length, out_length,
                 batch_size_per_gpu, num_gpus, device):
        self._data = data
        self._train_prop = train_prop
        self._valid_prop = valid_prop
        self._num_nodes = num_nodes
        self._in_length = in_length
        self._out_length = out_length
        self._batch_size_per_gpu = batch_size_per_gpu
        self._num_gpus = num_gpus
        self.device = device

        self.build_data_loader()

    def build_data_loader(self):
        logging.info('initialize data loader')
        train, test = (self._data[0], self._data[1]), (self._data[2], self._data[3])
        self.scaler = Scaler(train[0], missing_value=0)
        # data for search
        self.search_train = self.get_data_loader(train, shuffle=True, tag='search train',
                                                 num_gpus=self._num_gpus)  # for weight update
        self.search_valid = self.get_data_loader(test, shuffle=True, tag='search test',
                                                 num_gpus=self._num_gpus)  # for arch update
        # data for training & evaluation
        self.train = self.get_data_loader(train, shuffle=True, tag='train', num_gpus=1)
        self.valid = self.get_data_loader(test, shuffle=False, tag='test', num_gpus=1)
        self.test = self.get_data_loader(test, shuffle=False, tag='test', num_gpus=1)

    def get_data_loader(self, data, shuffle, tag, num_gpus):
        logging.info('load %s inputs & labels', tag)

        inputs = torch.unsqueeze(data[0], dim=2).permute(0, 3, 2, 1)
        labels = data[1].view(-1, 1, 1, 1)

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]

        # create dataset
        dataset = TensorDataset(
            inputs,
            labels
        )

        # create sampler
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=self._batch_size_per_gpu * num_gpus) if shuffle else SequentialSampler(
            dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size_per_gpu * num_gpus, sampler=sampler,
                                 num_workers=0, drop_last=False)
        return data_loader

    @property
    def batch_size_per_gpu(self):
        return self._batch_size_per_gpu


plt.ion()
plot = False  # draw train process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def snas_predict(data, parameters, save_file, log_file):
    X_train, y_train, X_test, y_test = data
    batch_size = parameters[0]
    layer_per_cell = parameters[1]
    search_epoch = parameters[2]
    search_iterations = parameters[3]
    cfg_path = parameters[4]
    input_shape = X_train.shape[1:]

    logger = get_logger('dain', log_file, logging.INFO)

    os.environ['NUMEXPR_MAX_THREADS'] = '16'

    epoch = 500

    cfg.load_config(cfg_path)

    # load data
    dataset = Dataset(
        data=data,
        train_prop=cfg.data.train_prop,
        valid_prop=cfg.data.valid_prop,
        num_nodes=cfg.data.num_nodes,
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        batch_size_per_gpu=batch_size,
        num_gpus=1,  # torch.cuda.device_count()
        device=device,
    )

    net = Net(
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        num_nodes=cfg.data.num_nodes,
        node_emb_dim=cfg.model.node_emb_dim,
        graph_hidden=cfg.model.graph_hidden,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        scale_channels=cfg.model.scale_channels,
        end_channels=cfg.model.end_channels,
        layer_structure=cfg.model.layer_structure,
        num_layer_per_cell=layer_per_cell,
        candidate_op_profiles_1=cfg.model.candidate_op_profiles_1,
        candidate_op_profiles_2=cfg.model.candidate_op_profiles_2
    )

    run_model = RunModel(
        name=save_file,
        net=net,
        dataset=dataset,

        arch_lr=cfg.trainer.arch_lr,
        arch_lr_decay_milestones=cfg.trainer.arch_lr_decay_milestones,
        arch_lr_decay_ratio=cfg.trainer.arch_lr_decay_ratio,
        arch_decay=cfg.trainer.arch_decay,
        arch_clip_gradient=cfg.trainer.arch_clip_gradient,

        weight_lr=cfg.trainer.weight_lr,
        weight_lr_decay_milestones=cfg.trainer.weight_lr_decay_milestones,
        weight_lr_decay_ratio=cfg.trainer.weight_lr_decay_ratio,
        weight_decay=cfg.trainer.weight_decay,
        weight_clip_gradient=cfg.trainer.weight_clip_gradient,

        num_search_iterations=search_iterations,
        num_search_arch_samples=cfg.trainer.num_search_arch_samples,
        num_train_iterations=cfg.trainer.num_train_iterations,

        criterion=cfg.trainer.criterion,
        metric_names=cfg.trainer.metric_names,
        metric_indexes=cfg.trainer.metric_indexes,
        print_frequency=cfg.trainer.print_frequency,

        device_ids=cfg.model.device_ids,  # range(torch.cuda.device_count())
    )

    run_model.load(mode='search')
    run_model.search(search_epoch, save_file)
    run_model.clear_records()
    run_model.initialize()
    best_test = run_model.train(epoch, save_file)

    return best_test.get_value()['rmse'][0]
