import logging

import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm

from compare import record_data, reshape_one_dim
from pyIAAS import get_logger


def svr_predict(data, parameters, save_file, log_file):
    logger = get_logger('svr', log_file, logging.INFO)
    X_train, y_train, X_test, y_test = data
    X_train = reshape_one_dim(X_train)
    X_test = reshape_one_dim(X_test)
    if parameters[0] == 'poly':
        kernel = parameters[0]
        degree = parameters[1]
        C = parameters[2]
        epsilon = parameters[3]
        regr = sklearn.svm.SVR(kernel='poly', degree=degree, C=C, epsilon=epsilon)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
    else:
        kernel = parameters[0]
        C = parameters[1]
        epsilon = parameters[2]
        regr = sklearn.svm.SVR(kernel=kernel, C=C, epsilon=epsilon)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

    data = pd.DataFrame({'pred': y_pred, 'truth': y_test})
    data.to_csv(f'{save_file}.csv')
    loss = sklearn.metrics.mean_squared_error(y_test, y_pred) ** 0.5
    logger.info(f'svr {save_file}   test {loss}')
    return loss


def rf_predict(data, parameters, save_file, log_file):
    logger = get_logger('rf', log_file, logging.INFO)
    X_train, y_train, X_test, y_test = data
    X_train = reshape_one_dim(X_train)
    X_test = reshape_one_dim(X_test)
    criterion, n_estimators = parameters
    regr = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
    # regr = sklearn.ensemble.RandomForestRegressor(max_depth=10, criterion ='absolute_error')
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    data = pd.DataFrame({'pred': y_pred, 'truth': y_test})
    data.to_csv(f'{save_file}.csv')
    loss = sklearn.metrics.mean_squared_error(y_test, y_pred) ** 0.5
    logger.info(f'rf {save_file}   test {loss}')
    return loss

def rr_predict(data, parameters, save_file, log_file):
    logger = get_logger('rr', log_file, logging.INFO)
    X_train, y_train, X_test, y_test = data
    X_train = reshape_one_dim(X_train)
    X_test = reshape_one_dim(X_test)
    solver, alpha = parameters
    regr = sklearn.linear_model.Ridge(solver=solver, alpha=alpha)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    data = pd.DataFrame({'pred': y_pred, 'truth': y_test})
    data.to_csv(f'{save_file}.csv')
    loss = sklearn.metrics.mean_squared_error(y_test, y_pred) ** 0.5
    logger.info(f'{save_file}   test {loss}')
    return loss