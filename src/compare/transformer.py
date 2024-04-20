import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyIAAS import get_logger, NasModel
from compare.LSTF.models.Autoformer import Model as Autoformer
from compare.LSTF.models.Informer import Model as Informer
from compare.LSTF.models.Transformer import Model as Transformer
import torchinfo
import argparse


def get_configs(model_name, pred_len, feature_number):

    args = f"--model {model_name} " \
           "--data custom " \
           "--features M " \
           "--seq_len 168 " \
           "--label_len 0 " \
           "--train_epochs 1000 " \
           f"--pred_len {pred_len} " \
           "--e_layers 2 " \
           "--d_layers 2 " \
           "--factor 3 " \
           f"--enc_in {feature_number} " \
           "--dec_in 1 " \
           "--d_model 512 " \
           "--n_heads 8 " \
           "--d_ff 512 " \
           "--c_out 1"
    args = args.split()


    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    # Formers
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utilsLocal/tools for usage')

    args = parser.parse_args(args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    return args

def transformer_predict(data, data_mark, save_file, log_file, prev_model, model_type):
    logger = get_logger('transformer', log_file, logging.INFO)
    model_dict = {
        'Informer': Informer,
        'Transformer': Transformer,
    }
    X_train, y_train, X_test, y_test = data
    X_train_mark, y_train_mark, X_test_mark, y_test_mark = data_mark
    configs = get_configs(model_type, y_train.shape[1], X_train.shape[-1])
    model = None
    if prev_model is None:
        model = model_dict[model_type](configs)
    else:
        model = prev_model

    if configs.use_gpu:
        model = model.cuda()
        X_train_mark, y_train_mark, X_test_mark, y_test_mark = X_train_mark.cuda(), y_train_mark.cuda(), X_test_mark.cuda(), y_test_mark.cuda()
        X_train, y_train, X_test, y_test = X_train.cuda(), y_train.cuda(), X_test.cuda(), y_test.cuda()

    epoch = configs.train_epochs
    loader = DataLoader(TensorDataset(X_train, y_train, X_train_mark, y_train_mark),
                        batch_size=256,
                        shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    loss_fn = NasModel.mix_rmse
    best_test = None
    best_model = None
    for i in range(epoch):
        losses = []
        model.train()
        for index, (X, y, X_mark, y_mark) in enumerate(loader):
            y_pred = model(X, X_mark, torch.zeros_like(y), y_mark)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'epoch {i} train loss {np.mean(losses)}')
        if i % 100 == 0 or i == epoch - 1:
            with torch.no_grad():
                model.eval()
                y_pred = model(X_test, X_test_mark, torch.zeros_like(y_test), y_test_mark)
                loss = loss_fn(y_pred, y_test)
                seq_loss = seq_RMSE(y_pred, y_test)
                if best_test is None or best_test > loss.item():
                    best_test = loss.item()
                    best_model = model
                    data = pd.DataFrame(
                        {'pred': y_pred.reshape(-1).detach().cpu(), 'truth': y_test.reshape(-1).detach().cpu()}, )
                    data.to_csv(f'{save_file}_pred.csv')
                    seq_loss_data = pd.DataFrame({'loss': seq_loss.view(-1).detach().cpu()})
                    seq_loss_data.to_csv(f'{save_file}_loss.csv')
                    torch.save(model, f'{save_file}_model.pt')
                    logger.info(
                        f'transformer {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()} new test best, save model')
                else:
                    logger.info(
                        f'transformer {save_file}  epoch {i}: train loss {np.average(losses)} test {loss.item()}')

    return best_model, best_test


def seq_RMSE(pred, truth):
    return ((pred - truth) ** 2).mean(0) ** 0.5
