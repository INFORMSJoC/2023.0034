import sys
import itertools
import logging
import os
import shutil
from collections import namedtuple
from copy import deepcopy

import pandas
import pandas as pd
import torch.nn

from compare.Sklearn_method import svr_predict, rf_predict, rr_predict
from compare.cnn_lstm_method import cnn_lstm_predict
from compare.cnn_method import cnn_predict
from compare.dain import dain_predict
from compare.lstm_method import lstm_predict
from compare.resnet import resnet_predict
from compare.resnetplus import resnetplus_predict
from compare.snas4mtf import snas_predict
from compare.transformer import transformer_predict
from pyIAAS import *
from pyIAAS.utils.data_process import _load_feature_value

search_task = namedtuple('search_task', ('function', 'parameters', 'name', 'save_file'))


def run_search_tasks(data, log_file, task_record_file, tasks, tasks_record):
    random.shuffle(tasks)
    for task in tasks:
        performance = task.function(data, task.parameters, task.save_file, log_file)
        tasks_record = pandas.concat([tasks_record, pd.DataFrame({'task': [task.name], 'loss': [performance]})], )
        tasks_record.to_csv(task_record_file)


def grid_search_cnn_lstm(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'cnn_lstm'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'cnn_lstm', 'record.csv')
    log_file = os.path.join(task_save_dir, 'cnn_lstm', 'log.txt')
    # create tasks
    cartesian_product = itertools.product(layers, width, activations, batch_sizes, learn_rates, dropout_rates)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} act {i[2][1]} bs {i[3]} lr {i[4]} dropout {i[5]}'
        tasks.append(search_task(cnn_lstm_predict,
                                 (i[0], i[1], i[2][0], i[3], i[4], i[5]),
                                 task_name,
                                 os.path.join(task_save_dir, 'cnn_lstm', task_name)))
    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_cnn(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'cnn'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'cnn', 'record.csv')
    log_file = os.path.join(task_save_dir, 'cnn', 'log.txt')
    # create tasks
    cartesian_product = itertools.product(layers, width, activations, batch_sizes, learn_rates, dropout_rates)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} act {i[2][1]} bs {i[3]} lr {i[4]} dropout {i[5]}'
        tasks.append(search_task(cnn_predict,
                                 (i[0] * [i[1]], i[2][0], i[3], i[4], i[5]),
                                 task_name,
                                 os.path.join(task_save_dir, 'cnn', task_name)))
    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_lstm(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'lstm'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'lstm', 'record.csv')
    log_file = os.path.join(task_save_dir, 'lstm', 'log.txt')
    # create tasks
    cartesian_product = itertools.product(layers, width, activations, batch_sizes, learn_rates, dropout_rates)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} act {i[2][1]} bs {i[3]} lr {i[4]} dropout {i[5]}'
        tasks.append(search_task(lstm_predict,
                                 (i[0], i[1], i[2][0], i[3], i[4], i[5]),
                                 task_name,
                                 os.path.join(task_save_dir, 'lstm', task_name)))
    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_svr(task_save_dir, data):
    set_seed(seed)
    data = [i.numpy() for i in data]
    kernels = ['rbf', 'sigmoid']
    poly_degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    regularization = [1, 10, 100, 1000]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    os.makedirs(os.path.join(task_save_dir, 'svr'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'svr', 'record.csv')
    log_file = os.path.join(task_save_dir, 'svr', 'log.txt')
    cartesian_product = list(itertools.product(regularization, epsilons))

    # create tasks
    tasks = []
    for i in kernels:
        for j in cartesian_product:
            task_name = f'kernel {i} C {j[0]} eps {j[1]}'
            tasks.append(search_task(svr_predict,
                                     (i, j[0], j[1]),
                                     task_name,
                                     os.path.join(task_save_dir, 'svr', task_name)))
    for i in poly_degree:
        for j in cartesian_product:
            task_name = f'kernel poly {i} C {j[0]} eps {j[1]}'
            tasks.append(search_task(svr_predict,
                                     ('poly', i, j[0], j[1]),
                                     task_name,
                                     os.path.join(task_save_dir, 'svr', task_name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_rf(task_save_dir, data):
    set_seed(seed)
    data = [i.numpy() for i in data]
    criterion = ['squared_error', 'absolute_error', 'poisson']
    n_estimators = [16, 32, 48, 64, 80, 96, 112, 128]
    os.makedirs(os.path.join(task_save_dir, 'rf'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'rf', 'record.csv')
    log_file = os.path.join(task_save_dir, 'rf', 'log.txt')
    cartesian_product = itertools.product(criterion, n_estimators, )

    # create tasks
    tasks = []
    for i in cartesian_product:
        name = f'criterion {i[0]} n_estimators {i[1]}'
        tasks.append(search_task(rf_predict,
                                 i,
                                 name,
                                 os.path.join(task_save_dir, 'rf', name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_rr(task_save_dir, data):
    set_seed(seed)
    data = [i.numpy() for i in data]
    solver = ['auto']
    alpha = [0.1+0.1*i for i in range(40)]
    os.makedirs(os.path.join(task_save_dir, 'rr'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'rr', 'record.csv')
    log_file = os.path.join(task_save_dir, 'rr', 'log.txt')
    cartesian_product = itertools.product(solver, alpha, )

    # create tasks
    tasks = []
    for i in cartesian_product:
        name = f'solver {i[0]} alpha {i[1]}'
        tasks.append(search_task(rr_predict,
                                 i,
                                 name,
                                 os.path.join(task_save_dir, 'rr', name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_res(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'res'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'res', 'record.csv')
    log_file = os.path.join(task_save_dir, 'res', 'log.txt')
    # create tasks
    cartesian_product = itertools.product(layers, width, activations, batch_sizes, learn_rates, dropout_rates)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} act {i[2][1]} bs {i[3]} lr {i[4]} dropout {i[5]}'
        tasks.append(search_task(resnet_predict,
                                 (i[0], i[1], i[2][0], i[3], i[4], i[5]),
                                 task_name,
                                 os.path.join(task_save_dir, 'res', task_name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_resplus(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'resplus'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'resplus', 'record.csv')
    log_file = os.path.join(task_save_dir, 'resplus', 'log.txt')
    # create tasks
    cartesian_product = itertools.product(layers, width, activations, batch_sizes, learn_rates, dropout_rates)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} act {i[2][1]} bs {i[3]} lr {i[4]} dropout {i[5]}'
        tasks.append(search_task(resnetplus_predict,
                                 (i[0], i[1], i[2][0], i[3], i[4], i[5]),
                                 task_name,
                                 os.path.join(task_save_dir, 'resplus', task_name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_dain(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'dain'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'dain', 'record.csv')
    log_file = os.path.join(task_save_dir, 'dain', 'log.txt')
    mean_lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    gate_lr = [1e-5, 1e-4, 1e-3, 1e-2]
    scale_lr = [1e-5, 1e-4, 1e-3, 1e-2]
    # create tasks
    cartesian_product = itertools.product(layers, width, mean_lrs, gate_lr, scale_lr)
    tasks = []
    for i in cartesian_product:
        task_name = f'layer {i[0]} width {i[1]} mean lr{i[2]} gate lr{i[3]} scale lr {i[4]}'
        tasks.append(search_task(dain_predict,
                                 (i),
                                 task_name,
                                 os.path.join(task_save_dir, 'dain', task_name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search_snas(task_save_dir, data):
    set_seed(seed)
    os.makedirs(os.path.join(task_save_dir, 'snas'), exist_ok=True)
    task_record_file = os.path.join(task_save_dir, 'snas', 'record.csv')
    log_file = os.path.join(task_save_dir, 'snas', 'log.txt')

    # create tasks
    layer_per_cell = [1,2,3,4]
    search_epoch = [50,60,70,80,90,100]
    search_iterations = list(range(64, 256+1,64))
    cartesian_product = itertools.product(batch_sizes, layer_per_cell,search_epoch,search_iterations, [snas_cfg_file])
    tasks = []
    for i in cartesian_product:
        task_name = f'bs {i[0]} lc {i[1]} se {i[2]} si {i[3]}'
        tasks.append(search_task(snas_predict,
                                 i,
                                 task_name,
                                 os.path.join(task_save_dir, 'snas', task_name)))

    tasks_record = pd.DataFrame()
    # run tasks
    run_search_tasks(data, log_file, task_record_file, tasks, tasks_record)


def grid_search(record_save_dir, dataset_list, data_dir, target_name, timeLength, predictLength, test_ratio):
    os.makedirs(cache_dir, exist_ok=True)
    for dataset in dataset_list:
        task_name = dataset.split('.')[0]
        x, y = get_data(cache_dir, os.path.join(data_dir, dataset), target_name,
                        timeLength, predictLength)
        data = train_test_split(x, y, test_ratio)
        data = [torch.tensor(i, dtype=torch.float) for i in data]
        task_save_dir = os.path.join(record_save_dir, task_name)
        os.makedirs(task_save_dir, exist_ok=True)

        # uncomment the comparison method you want to run
        grid_search_cnn_lstm(task_save_dir, data)
        grid_search_cnn(task_save_dir, data)
        grid_search_lstm(task_save_dir, data)
        grid_search_svr(task_save_dir, data)
        grid_search_rf(task_save_dir, data)
        grid_search_rr(task_save_dir, data)
        grid_search_res(task_save_dir, data)
        grid_search_resplus(task_save_dir, data)
        grid_search_dain(task_save_dir, data)
        grid_search_snas(task_save_dir, data)


def get_sequence_data(cache_dir, input_file, target_name, time_length: int, predict_length: int) -> (
        np.ndarray, np.ndarray):
    """
    preprocessing data from original data and return the numpy result
    the data is formed to a seq2seq format
    if data is preprocessed before, return the previously stored result
    :param predict_length: predicting future time from current time
    :param target_name: name of target value in CSV file
    :param input_file: input CSV data file
    :param cache_dir: middle cache dir to store arranged data
    :param time_length: length of time in feature
    :return: X,y is feature, target.
    """
    pkl_data_path = os.path.join(cache_dir, os.path.split(input_file)[-1].replace('.csv',
                                                                                  f'_{time_length}_seq{predict_length}.pkl'))
    if os.path.exists(pkl_data_path):
        with open(pkl_data_path, 'rb') as f:
            data = pickle.load(f)
            X = data['X']
            y = data['y']
            X_mark = data['X_mark']
            y_mark = data['y_mark']
            return X, y, X_mark, y_mark
    data = pd.read_csv(input_file)

    # check target_name exist
    if not target_name in data.columns:
        raise RuntimeError(f'not column named {target_name} in input file {input_file}')

    df_stamp = data[['date']]
    cols = list(data.columns)
    cols.remove('date')
    data = data[cols]

    # convert all data to float values, (make sure step)
    try:
        for column in data.columns:
            data.loc[:, column] = data.loc[:, column].apply(
                lambda x: float(x))
    except:
        raise RuntimeError(f'not all data be float value, please check again or use custom data processing method')
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], 1)

    X, y = [], []
    X_mark, y_mark = [], []
    for i in range(time_length, data.shape[0] - predict_length):
        X.append(data.loc[i - time_length:i - 1, :].to_numpy())
        X_mark.append(data_stamp.loc[i - time_length:i - 1, :].to_numpy())
        y.append(data.loc[i:i + predict_length - 1, [target_name]])
        y_mark.append(data_stamp.loc[i:i + predict_length - 1, :].to_numpy())
    X = np.array(X)
    y = np.array(y)
    X_mark = np.array(X_mark)
    y_mark = np.array(y_mark)
    with open(pkl_data_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'X_mark': X_mark, 'y_mark': y_mark}, f)
    with open(pkl_data_path, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
        X_mark = data['X_mark']
        y_mark = data['y_mark']
        return X, y, X_mark, y_mark


def transformer_compare(record_save_dir, dataset_list, data_dir, target_name, timeLength, predictLength, train_size,
                        test_size, sliding_size, model_type):
    set_seed(seed)
    slide_total_num = 5
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(record_save_dir, exist_ok=True)
    log_file = os.path.join(record_save_dir, 'log.txt')
    logger = get_logger('transformer', log_file, logging.INFO)
    transformer_model = None
    for dataset in dataset_list:
        task_name = dataset.split('.')[0]
        X, y, X_mark, y_mark = get_sequence_data(cache_dir, os.path.join(data_dir, dataset), target_name,
                                                 timeLength, predictLength)
        for step in range(slide_total_num):
            window_start = step * sliding_size
            window_end = window_start + train_size + test_size
            data = train_test_split(X[window_start:window_end], y[window_start:window_end], test_size)
            data = [torch.tensor(i, dtype=torch.float) for i in data]
            data_mark = train_test_split(X_mark[window_start:window_end], y_mark[window_start:window_end], test_size)
            data_mark = [torch.tensor(i, dtype=torch.float) for i in data_mark]
            save_dir = os.path.join(record_save_dir, task_name)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(record_save_dir, task_name, f'step {step} window start {window_start}')
            logger.info(f'start window {window_start}-{window_end}'.center(50, '='))
            transformer_model, best_test = transformer_predict(data, data_mark, save_file, log_file, transformer_model,
                                                               model_type)


def IAAS_seq_compare(save_dir, dataset_list, data_dir, target_name, timeLength, predictLength, train_size, test_size,
                     sliding_size):
    slide_total_num = 5
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'log.txt')
    logger = get_logger('IAAS seq', log_file, logging.INFO)
    transformer_model = None
    cfg = Config(config_file)
    for dataset in dataset_list:
        set_seed(seed)
        task_name = dataset.split('.')[0]
        x, y, _, _ = get_sequence_data(cache_dir, os.path.join(data_dir, dataset), target_name,
                                       timeLength, predictLength)
        y = y[:, :, 0]
        for step in range(slide_total_num):
            window_start = step * sliding_size
            window_end = window_start + train_size + test_size
            data = train_test_split(x[window_start:window_end], y[window_start:window_end], test_size)
            data = [torch.tensor(i, dtype=torch.float) for i in data]
            task_dir = os.path.join(save_dir, task_name, f'step {step} window start {window_start}')
            os.makedirs(task_dir, exist_ok=True)
            task_cfg = deepcopy(cfg)
            task_cfg.NASConfig['OUT_DIR'] = task_dir
            logger.info(f'start window {window_start}-{window_end}'.center(50, '='))
            if step > 0:
                # copy previous search result to current search task
                files_to_copy_list = [
                    'agent.pkl',
                    'env.pkl',
                    'model.db',
                    'replay_memory.pkl',
                    'rng_state.pkl',
                ]
                for f in files_to_copy_list:
                    src = os.path.join(save_dir,
                                       task_name,
                                       f'step {step - 1} window start {window_start - sliding_size}',
                                       f)
                    dst = os.path.join(task_dir, f)
                    shutil.copyfile(src, dst)
                env_ = NasEnv.try_load(task_cfg, logger)
                for net in env_.net_pool:
                    net.test_loss_best = None
                env_.save()
            search_net(task_cfg, data, logger)


if __name__ == '__main__':
    load_dataset_list = [
        'ME_spring.csv',
        'ME_summer.csv',
        'ME_autumn.csv',
        'ME_winter.csv',
        'NH_spring.csv',
        'NH_summer.csv',
        'NH_autumn.csv',
        'NH_winter.csv'
    ]

    trans_wind_datasets = [
        'WF1.csv',
    ]

    trans_load_datasets = [
        'ME.csv',
    ]

    wind_dataset_list = [
        'WF1_spring.csv',
        'WF1_summer.csv',
        'WF1_autumn.csv',
        'WF1_winter.csv',
        'WF2_spring.csv',
        'WF2_summer.csv',
        'WF2_autumn.csv',
        'WF2_winter.csv'
    ]
    seed = 42  # random seed
    cache_dir = 'cache'
    os.chdir('../data')  # change working dir
    os.makedirs(cache_dir, exist_ok=True)


    # grid search parameters
    layers = list(range(2,10+1))
    width = list(range(4, 128 + 1, 4))
    activations = [(torch.nn.ReLU, 'ReLU'), (torch.nn.SELU, 'SELU'), (torch.nn.LeakyReLU, 'LeakyReLU')]
    batch_sizes = list(range(64, 1024 + 1, 64))
    learn_rates = [i*0.01 + 0.01 for i in range(10)]
    dropout_rates = [0.2 + 0.1*i for i in range(7)]

    record_save_dir = 'compare_results'

    # grad search for comparison methods
    # adjust code in grid_search to run different comparison methods
    snas_cfg_file = '../src/compare/SNAS/config/wind_data.yaml'
    grid_search(record_save_dir, wind_dataset_list, 'wind dataset aligned', '实际功率', 72, 24, 24 * 5)
    snas_cfg_file = '../src/compare/SNAS/config/load_data.yaml'
    grid_search(record_save_dir, load_dataset_list, 'load_datasets', 'RT_Demand', 168, 24,
                int(24 * (30 + 31 + 30) * 0.2))

    # transformer comparison
    model_types = ['Informer', 'Transformer']
    timeLength = 168*2
    predictLength = 168
    test_size, sliding_size = 168, 168
    for model_type in model_types:
        transformer_save_dir = f'{model_type}_compare'
    transformer_compare(transformer_save_dir, trans_load_datasets, 'transformer_dataset', 'RT_Demand', timeLength, predictLength, int(24 * (30 + 31 + 30) * 0.8), test_size, sliding_size, model_type)
    transformer_compare(transformer_save_dir, trans_wind_datasets, 'transformer_dataset', '实际功率', timeLength, predictLength, int(24 * (30 + 31 + 30) - 5 * 24), test_size, sliding_size, model_type)
    IAAS_seq_dir = 'IAAS_seq_compare'
    config_file = 'NASConfig_load_seq.json'
    IAAS_seq_compare(IAAS_seq_dir, trans_load_datasets, 'transformer_dataset', 'RT_Demand', timeLength, predictLength, int(24 * (30 + 31 + 30) * 0.8), test_size, sliding_size)
    config_file = 'NASConfig_wind_seq.json'
    IAAS_seq_compare(IAAS_seq_dir, trans_wind_datasets, 'transformer_dataset', '实际功率', timeLength, predictLength, int(24 * (30 + 31 + 30) - 5 * 24), test_size, sliding_size)
