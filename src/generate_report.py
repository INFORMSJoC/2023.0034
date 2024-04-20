import json
import os.path
import shutil
import sqlite3
from datetime import datetime

import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn
# Import seaborn
import seaborn as sns
import torch
import torchinfo

path_join = os.path.join


def RMSE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    return np.sqrt(np.mean(np.square(pred - truth)))


def RMSLE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    return np.sqrt(np.mean(np.square(np.log(pred + 1) - np.log(truth + 1))))


def MSE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    return np.mean(np.square(pred - truth))


def MAE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    return np.mean(np.abs(pred - truth))


def MAPE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    index = truth != 0
    return np.mean(np.abs((pred[index] - truth[index]) / truth[index]))


def plot_prediction(save_dir, search_name, avg_count):
    os.makedirs(save_dir, exist_ok=True)
    size = 10
    font_size = 20
    plt.figure(figsize=(2 * size, 1 * size))

    for place in places:
        plt.clf()
        record = pd.DataFrame()
        for i in range(len(seasons)):
            season = seasons[i]
            plt.subplot(411 + i)
            for method in [f'{search_name}{avg_count}']:
                predict_result_path = path_join(predict_result_dir, f'{method}_{place}_{season}.csv')
                if not os.path.exists(predict_result_path):
                    continue
                d = pd.read_csv(predict_result_path)
                record.loc[:, 'Real'] = d.loc[:, 'truth']
                record.loc[:, method] = d.loc[:, 'pred']
                plt.plot(record.loc[:, f'{method}'], label=f'{method}', ls='-.')
            plt.plot(record.loc[:, 'Real'], label='Real')
            plt.title(season, fontsize=font_size)
            if i == 3:
                plt.xlabel('Time(h)', fontsize=font_size)
            plt.ylabel('MW', fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
        plt.subplots_adjust(top=0.945,
                            bottom=0.095,
                            left=0.075,
                            right=0.98,
                            hspace=0.50,
                            wspace=0.2)
        # plt.tight_layout()
        plt.savefig(path_join(save_dir, f"{place}.pdf"))


def generate_report_from_predict_result(metrix_f, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    record = pd.DataFrame()
    for place in places:
        for season in seasons:
            min_max = None
            for method in methods:
                predict_result_path = path_join(predict_result_dir, f'{method}_{place}_{season}.csv')
                if not os.path.exists(predict_result_path):
                    continue
                d = pd.read_csv(predict_result_path)
                record.loc[f'{method}', f'{place}-{season}'] = metrix_f(d.loc[:, 'pred'], d.loc[:, 'truth'], min_max)
                record.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.3f')
    record.loc[:, 'mean'] = record.mean(axis=1)
    record.T.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.3f')
    print(f'{metrix_f} result ')


def restore_scaling(data, min_v, max_v):
    if 1 + 1e-2 >= (data.max() - data.min()):
        return data * (max_v - min_v) + min_v
    else:
        return data


def generate_report_from_predict_result_vix(metrix_f, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    record = pd.DataFrame()
    for method in methods:
        predict_result_path = path_join(predict_result_dir, f'{method}_vix_19.csv')
        if not os.path.exists(predict_result_path):
            continue
        d = pd.read_csv(predict_result_path)
        record.loc[f'{method}', f'vix-19'] = metrix_f(d.loc[:, 'pred'] * 1, d.loc[:, 'truth'] * 1)
        record.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.3f')
    record.loc[:, 'mean'] = record.mean(axis=1)
    # if metrix_f is MAPE:
    #     record *= 100
    record.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.4f')
    print(f'{metrix_f} result ')


def get_best_model_info(sql_file):
    conn = sqlite3.connect(sql_file)
    sql_quary = '''
    SELECT ID FROM MODEL  WHERE LOSS is not null  ORDER BY LOSS
    '''
    data = conn.execute(sql_quary).fetchone()[0]
    conn.close()
    return data


def get_top_n_model_info(sql_file, n):
    conn = sqlite3.connect(sql_file)
    sql_quary = f'''
    SELECT ID FROM MODEL  WHERE LOSS is not null  ORDER BY LOSS LIMIT {n}
    '''
    data = conn.execute(sql_quary).fetchall()
    conn.close()
    return [i[0] for i in data]


def get_all_model_info(sql_file):
    conn = sqlite3.connect(sql_file)
    sql_quary = f'''
    SELECT ID, LOSS, TRAIN_TIME FROM MODEL  WHERE LOSS is not null ORDER BY ID
    '''
    data = conn.execute(sql_quary).fetchall()
    conn.close()
    return data


def get_all_model_and_parent_info(sql_file):
    conn = sqlite3.connect(sql_file)
    sql_quary = f'''
    SELECT ID, PREV_INDEX, LOSS FROM MODEL  WHERE LOSS is not null ORDER BY ID
    '''
    data = conn.execute(sql_quary).fetchall()
    conn.close()
    return data


def get_top_n_CNN_model_info(sql_file, n):
    conn = sqlite3.connect(sql_file)
    sql_quary = f'''
    SELECT ID,structure FROM MODEL  WHERE LOSS is not null  ORDER BY LOSS
    '''
    data = conn.execute(sql_quary).fetchall()
    conn.close()
    result = []
    cnt, index = 0, 0
    while True:
        if 'rnn' in data[index][1]:
            index += 1
            continue
        result.append(data[index][0])
        cnt += 1
        index += 1
        if cnt == n:
            break
    return result


def copy_predict_result_IAAS(avg_count, search_name, search_type_dir, save_dir):
    global methods

    plot_model_dir = path_join(save_dir, 'plot_model')
    plot_loss_dir = path_join(save_dir, 'plot_loss')
    os.makedirs(plot_model_dir, exist_ok=True)
    os.makedirs(plot_loss_dir, exist_ok=True)
    model_structure = pd.DataFrame()
    for place in places:
        plt.clf()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(len(seasons)):
            plt.subplot(221 + i)
            season = seasons[i]
            # dot = graphviz.Digraph(f'{place}-{season}')
            search_result_dir = path_join(search_type_dir, f'{place}_{season}')
            sql_file = path_join(search_result_dir, 'model.db')
            if not os.path.exists(search_result_dir):
                continue
            best_model_dir = None
            for i in os.listdir(search_result_dir):
                if 'best' in i:
                    best_model_dir = i
                    break
            best_model_dir = path_join(search_result_dir, best_model_dir)

            predict_result_path = path_join(predict_result_dir, f'{search_name}{avg_count}_{place}_{season}.csv')
            pred_table = pd.read_csv(path_join(best_model_dir, 'best_pred.csv'), index_col=0)
            pred_table.to_csv(predict_result_path)

            model_ids = get_top_n_model_info(sql_file, avg_count)
            pred_tables = []
            loss_curve = None
            for model_id in model_ids:
                best_model_dir = path_join(search_result_dir, f'{model_id}')
                pred_table = pd.read_csv(path_join(best_model_dir, 'best_pred.csv'))
                pred_tables.append(pred_table)
                if loss_curve is None:  # plot on top one model
                    loss_curve = pd.read_csv(path_join(best_model_dir, 'loss.csv'))
                    plt.plot(loss_curve.iloc[:, 1])
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.title(season)
                    # plt.ylim(-0.005, y_max[place][i] * gap)
                    # draw model picture
                    transformation = pd.read_csv(path_join(best_model_dir, 'transformation.csv'))
                    id = transformation.loc[transformation.shape[0] - 1, 'current']
                    model_dir = path_join(search_result_dir, f'{id}')
                    with open(path_join(model_dir, 'NasModel.txt')) as f:
                        model_str = f.readline().replace('dense', 'fc')
                    components = model_str.split('->')
                    print(f'{search_name} {place} {season} components {len(components)}')
                    model_structure.loc[f'{place}-{season}', 'structure'] = model_str

                    # sub_graph = graphviz.Digraph(str(model_index))
                    #     for i in range(len(components)):
                    #         dot.node(str(model_index) + str(i), label=components[i], shape='box')
                    #         if i > 0:
                    #             dot.edge(str(model_index) + str(i - 1), str(model_index) + str(i))
                    #     dot.subgraph(sub_graph)
                    #     if model_index != transformation.shape[0]-1:
                    #         dot.edge(str(model_index)+str(0), str(model_index+1)+str(0))
                    # dot.render(path_join(plot_model_dir, f'{place}-{season}'), format='pdf')
            pred_avg = np.stack(pred_tables)[:, :, 2]
            pred_avg = np.average(pred_avg, axis=0)
            pred_table = pred_tables[0]
            pred_table.loc[:, 'pred'] = pred_avg
            pred_table.iloc[:, 1:].to_csv(predict_result_path)

            # later model pred
            # model_id = get_top_n_model_info(sql_file, avg_count + 10)[-1]
            # best_model_dir = path_join(search_result_dir, str(model_id))
            # pred_table = pd.read_csv(path_join(best_model_dir, 'pred.csv'))
            # predict_result_path = path_join(predict_result_dir, f'IAAS_{place}_{season}.csv')
            # pred_table.iloc[:, 1:].to_csv(predict_result_path)

        plt.savefig(path_join(plot_loss_dir, f'{place}_loss.pdf'))
    model_structure.to_csv(path_join(save_dir, 'structure.csv'))
    plot_prediction(path_join(save_dir, 'prediction'), search_name, avg_count)

def summary_search_progress(save_dir, search_type):
    os.makedirs(save_dir, exist_ok=True)
    param_table = pd.DataFrame(index=range(1))
    for place in places:
        for i in range(len(seasons)):
            season = seasons[i]
            search_result_dir = path_join(search_type, f'{place}_{season}')
            if not os.path.exists(search_result_dir):
                continue
            best_model_dir = None
            for i in os.listdir(search_result_dir):
                if 'best' in i:
                    best_model_dir = i
                    break
            best_model_dir = path_join(search_result_dir, best_model_dir)
            model_instance = torch.load(path_join(best_model_dir, 'NasModel.pth'))
            param_number = [torchinfo.summary(model_instance, verbose=False).total_params]
            param_table.loc[0, f'{place}_{season}'] = np.array(param_number)
            param_table.to_csv(path_join(save_dir, f'top_1_param.csv'))
            print(f'{search_type} {place} {season}:')
            print(param_table)



def generate_load_report():
    global places
    global methods
    methods = [
        'cnn_lstm',
        'cnn',
        'lstm',
        'svr',
        'rf',
        'rr',
        'res',
        'resplus',
        'dain',
        'snas',
        # 'SNAS-MTF'
    ]

    places = ['ME', 'NH']
    # places = ['ME']
    IAAS_no_randon_net = r'../no random net'
    IAAS = r'../IAAS'
    IAAS_n = r'../IAAS no pool'
    IAAS_sn = r'../IAAS no pool no selector'
    IAAS_s = r'../IAAS no selector'
    save_dir = 'load final'
    for i in [1]:  # avg count
        methods.append(f'no_randon_net{i}')
        # methods.append(f'IAAS{i}')
        # methods.append(f'IAAS_n{i}')
        # methods.append(f'IAAS_s{i}')
        # methods.append(f'IAAS_sn{i}')
        copy_predict_result_IAAS(i, f'no_randon_net', IAAS_no_randon_net,
                                 path_join(save_dir, 'no_randon_net'))
        # copy_predict_result_IAAS(i, f'IAAS', IAAS,
        #                          path_join(save_dir, 'IAAS'))
        # copy_predict_result_IAAS(i, f'IAAS_n', IAAS_n,
        #                          path_join(save_dir, 'IAAS_n'))
        # copy_predict_result_IAAS(i, f'IAAS_s', IAAS_s,
        #                          path_join(save_dir, 'IAAS_s'))
        # copy_predict_result_IAAS(i, f'IAAS_sn', IAAS_sn,
        #                          path_join(save_dir, 'IAAS_sn'))
    # summary_search_progress(path_join(save_dir, 'IAAS', 'search_progress'), IAAS)
    # summary_search_progress(path_join(save_dir, 'IAAS_n', 'search_progress'), IAAS_n)
    # summary_search_progress(path_join(save_dir, 'IAAS_s', 'search_progress'), IAAS_s)
    # summary_search_progress(path_join(save_dir, 'IAAS_sn', 'search_progress'), IAAS_sn)
    generate_report_from_predict_result(RMSLE, path_join(save_dir, 'RMSLE'))
    generate_report_from_predict_result(RMSE, path_join(save_dir, 'RMSE'))
    generate_report_from_predict_result(MAE, path_join(save_dir, 'MAE'))
    restructure_error_data(['RMSE', 'MAE', 'RMSLE'], save_dir)




def restructure_error_data(metric_list, save_dir):
    result = pd.DataFrame()
    data_list = []
    for metric in metric_list:
        file_path = path_join(save_dir, metric, 'method_compare.csv')
        data = pd.read_csv(file_path, index_col=0)
        data_list.append(data)

    for i in data.index:
        for j in data.columns:
            for k in range(len(metric_list)):
                result.loc[f'{i}-{metric_list[k]}', j] = data_list[k].loc[i, j]
    result.to_csv(path_join(save_dir, 'method_compare_summary.csv'))


def generate_wind_power_report():
    global places
    global methods
    methods = [
        'cnn_lstm',
        'cnn',
        'lstm',
        'svr',
        'rf',
        'rr',
        'res',
        'resplus',
        'dain',
        'snas',
        # 'SNAS-MTF'
    ]

    places = ['WF1', 'WF2']
    IAAS = r'../IAAS'
    IAAS_n = r'../IAAS no pool'
    IAAS_sn = r'../IAAS no pool no selector'
    IAAS_s = r'../IAAS no selector'
    save_dir = 'wind power final'
    for i in [1]:
        methods.append(f'IAAS{i}')
        methods.append(f'IAAS_n{i}')
        methods.append(f'IAAS_s{i}')
        methods.append(f'IAAS_sn{i}')
        # copy_predict_result_IAAS(i, f'IAAS', IAAS,
        #                          path_join(save_dir, 'IAAS'))
        # copy_predict_result_IAAS(i, f'IAAS_n', IAAS_n,
        #                          path_join(save_dir, 'IAAS_n'))
        # copy_predict_result_IAAS(i, f'IAAS_s', IAAS_s,
        #                          path_join(save_dir, 'IAAS_s'))
        # copy_predict_result_IAAS(i, f'IAAS_sn', IAAS_sn,
        #                          path_join(save_dir, 'IAAS_sn'))
    # summary_search_progress(path_join(save_dir, 'IAAS', 'search_progress'), IAAS)
    # summary_search_progress(path_join(save_dir, 'IAAS_n', 'search_progress'), IAAS_n)
    # summary_search_progress(path_join(save_dir, 'IAAS_s', 'search_progress'), IAAS_s)
    # summary_search_progress(path_join(save_dir, 'IAAS_sn', 'search_progress'), IAAS_sn)
    # generate_report_from_predict_result(RMSLE, path_join(save_dir, 'RMSLE'))
    # generate_report_from_predict_result(RMSE, path_join(save_dir, 'RMSE'))
    # generate_report_from_predict_result(MAE, path_join(save_dir, 'MAE'))
    restructure_error_data(['RMSE', 'MAE'], save_dir)

def data_pattern_plot_load():
    fontsize = 20
    ticksize = 20
    data_root = r'../data/used dataset'
    places = ['NH', 'ME']
    seasons = ['spring', 'summer', 'autumn', 'winter']
    data_name = 'RT_Demand'
    color_plate = sns.color_palette()
    for place in places:
        plt.figure(figsize=(16, 9))
        print(place)
        plt.title(place)
        row = 1
        origin_data = pd.DataFrame()
        for season in seasons:
            plt.subplot(4, 2, row)
            data = pd.read_csv(path_join(data_root, f'{place}_{season}.csv'))
            # data = data.loc[:, data_name]
            plt.plot(data.loc[:, 'RT_Demand'], color=color_plate[row // 2])
            plt.ylim(500, 2400)
            plt.ylabel('MW', fontsize=fontsize)
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
            plt.subplot(2, 2, 2)
            sns.distplot(data.loc[:, 'RT_Demand'], hist=True, bins=20, kde=True, vertical=False,
                         color=color_plate[row // 2], label=season)
            s = data.loc[:, ['RT_Demand']]
            s.columns = ['MW']
            s.loc[:, 'season'] = season
            origin_data = pd.concat([origin_data, s], axis=0)
            row += 2
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
        plt.ylabel("Density", fontsize=fontsize)
        plt.subplot(2, 2, 4)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        sns.boxplot(x='season', y='MW', data=origin_data, orient="v")
        plt.xlabel('')
        plt.ylabel('MW', fontsize=fontsize)
        plt.subplot(2, 2, 2)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.xlabel('')
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(path_join('data_description', f'data {place}.pdf'))


def data_pattern_plot_wind():
    fontsize = 20
    ticksize = 20
    data_root = r'../data/used dataset'
    places = ['大帽山', '忠门']
    places_code = {'大帽山': 'WF1', '忠门': 'WF2'}
    seasons = ['spring', 'summer', 'autumn', 'winter']
    data_name = '实际功率'
    color_plate = sns.color_palette()
    for place in places:
        print(place)
        plt.figure(figsize=(16, 9))
        plt.title(place)
        row = 1
        origin_data = pd.DataFrame()
        for season in seasons:
            plt.subplot(4, 2, row)
            data = pd.read_csv(path_join(data_root, f'{place}_{season}.csv'))
            # data = data.loc[:, data_name]
            plt.plot(data.loc[:, data_name], color=color_plate[row // 2])
            plt.ylabel('MW', fontsize=fontsize)
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
            plt.subplot(2, 2, 2)
            sns.distplot(data.loc[:, data_name], hist=True, bins=20, kde=True, vertical=False,
                         color=color_plate[row // 2], label=season)
            s = data.loc[:, [data_name]]
            s.columns = ['MW']
            s.loc[:, 'season'] = season
            origin_data = pd.concat([origin_data, s], axis=0)
            row += 2
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
        plt.ylabel("Density", fontsize=fontsize)
        plt.subplot(2, 2, 4)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        sns.boxplot(x='season', y='MW', data=origin_data, orient="v")
        plt.xlabel('')
        plt.ylabel('MW', fontsize=fontsize)
        plt.subplot(2, 2, 2)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.xlabel('')
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(path_join('data_description', f'data {places_code[place]}.pdf'))


def generate_performance_boxplot():
    plot_root_dir = 'performance_plot'
    os.makedirs(plot_root_dir, exist_ok=True)
    # sns.set_theme()

    # load 数据的结果
    data_root = 'load final'
    rmse_data = pd.read_csv(os.path.join(data_root, 'RMSE', 'method_compare.csv'))
    rmse_data = box_preprocess(rmse_data, 'IAAS_LoadNet')
    rmsle_data = pd.read_csv(os.path.join(data_root, 'RMSLE', 'method_compare.csv'))
    rmsle_data = box_preprocess(rmsle_data, 'IAAS_LoadNet')
    mae_data = pd.read_csv(os.path.join(data_root, 'MAE', 'method_compare.csv'))
    mae_data = box_preprocess(mae_data, 'IAAS_LoadNet')

    size = 8
    font_size = 20
    rotation = -90
    plt.figure(figsize=(2 * size, 1 * size))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=rmse_data, width=0.5, showfliers=True, showmeans=True, meanline=True,
                meanprops={'linestyle': '--', 'color': 'red'})
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('RMSE', fontsize=font_size)

    plt.subplot(1, 3, 3)
    sns.boxplot(data=rmsle_data, width=0.5, showfliers=True, showmeans=True, meanline=True,
                meanprops={'linestyle': '--', 'color': 'red'})
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('RMSLE', fontsize=font_size)

    plt.subplot(1, 3, 2)
    sns.boxplot(data=mae_data, width=0.5, showfliers=True, showmeans=True, meanline=True,
                meanprops={'linestyle': '--', 'color': 'red'})
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('MAE', fontsize=font_size)

    plt.subplots_adjust(top=0.92,
                        bottom=0.240,
                        left=0.045,
                        right=0.97,
                        hspace=0.195,
                        wspace=0.250)

    plt.savefig(path_join(plot_root_dir, f'load_boxplot.pdf'))
    # plt.clf()

    # wind power 数据的结果
    data_root = 'wind power final'
    rmse_data = pd.read_csv(os.path.join(data_root, 'RMSE', 'method_compare.csv'))
    rmse_data = box_preprocess(rmse_data, 'IAAS_WindNet')
    mae_data = pd.read_csv(os.path.join(data_root, 'MAE', 'method_compare.csv'))
    mae_data = box_preprocess(mae_data, 'IAAS_WindNet')
    rmsle_data = pd.read_csv(os.path.join(data_root, 'RMSLE', 'method_compare.csv'))
    rmsle_data = box_preprocess(rmsle_data, 'IAAS_WindNet')

    plt.figure(figsize=(2 * size * 2 / 3, 1 * size))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=rmse_data, width=0.5, showfliers=True, showmeans=True, meanline=True,
                meanprops={'linestyle': '--', 'color': 'red'})
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('RMSE', fontsize=font_size)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=mae_data, width=0.5, showfliers=True, showmeans=True, meanline=True,
                meanprops={'linestyle': '--', 'color': 'red'})
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('MAE', fontsize=font_size)

    plt.subplots_adjust(top=0.92,
                        bottom=0.240,
                        left=0.045,
                        right=0.97,
                        hspace=0.195,
                        wspace=0.25)
    plt.savefig(path_join(plot_root_dir, f'wind_boxplot.pdf'))
    plt.clf()


def box_preprocess(data, method_name):
    data = data.iloc[:-1, 1:]
    data = data.iloc[:, :11]
    col = ['CNN+LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'RR', 'ResNet', 'ResNetPlus', 'DAIN', 'SNAS', 'IAAS']
    col[-1] = method_name
    data.columns = col
    return data


def collect_grid_search_result():
    """
    搜集grid search的数据到两张表格中，一张保存性能，一张保存超参数。遇到不存在的数据则留空
    @return:
    """
    compare_result_dir = 'compare'
    compare_result_summary_file = 'compare/compare_summary_wind.xlsx'
    data_list = [
        'WF1_spring',
        'WF1_summer',
        'WF1_autumn',
        'WF1_winter',
        'WF2_spring',
        'WF2_summer',
        'WF2_autumn',
        'WF2_winter',
        'ME_spring',
        'ME_summer',
        'ME_autumn',
        'ME_winter',
        'NH_spring',
        'NH_summer',
        'NH_autumn',
        'NH_winter',
    ]
    method_list = [
        'cnn_lstm',
        'cnn',
        'lstm',
        'svr',
        'rf',
        'rr',
        'res',
        'resplus',
        'snas',
    ]
    performance_table = pd.DataFrame()
    parameters_table = pd.DataFrame()
    for method in method_list:
        for dataset in data_list:
            record_file = os.path.join(compare_result_dir, dataset, method, 'record.csv')
            # regenerate report
            regenerate = False
            if regenerate:
                file_list = os.listdir(os.path.join(compare_result_dir, dataset, method))
                for i in ['record.csv', 'log.txt', '.__log.txt.lock', 'log.txt.1']:
                    if i in file_list:
                        file_list.remove(i)
                record_item = pd.DataFrame()
                for f in file_list:
                    path = path_join(compare_result_dir, dataset, method, f)
                    # d = pd.read_csv(path).iloc[-120:]
                    d = pd.read_csv(path)
                    r = RMSE(d.truth.to_numpy(), d.pred.to_numpy())
                    record_item.loc[f.split('.cs')[0], 'loss'] = r
                    record_item.loc[f.split('.cs')[0], 'task'] = f.split('.cs')[0]
                record_item.to_csv(record_file)
            if not os.path.exists(record_file):
                print(f'file {record_file} not exist')
                continue
            record = pd.read_csv(record_file, index_col=0)
            record = record.sort_values(by='loss', ascending=True)
            best = record.iloc[0, :]
            task = best.task
            loss = best.loss
            performance_table.loc[dataset, method] = loss
            parameters_table.loc[dataset, method] = task
            # copy best model pred result to compare dir
            path = path_join(compare_result_dir, dataset, method, f'{task}')
            d = pd.read_csv(path)
            d.loc[:, ['pred', 'truth']].to_csv(path_join(predict_result_dir, f'{method}_{dataset}.csv'))

    with pd.ExcelWriter(compare_result_summary_file) as writer:
        performance_table.to_excel(writer, 'performance')
        parameters_table.to_excel(writer, 'parameters')


def collect_search_result():
    result_list = [
        'ME_spring',
        'ME_summer',
        'ME_autumn',
        'ME_winter',
        'NH_spring',
        'NH_summer',
        'NH_autumn',
        'NH_winter',
        'WF1_spring',
        'WF1_summer',
        'WF1_autumn',
        'WF1_winter',
        'WF2_spring',
        'WF2_summer',
        'WF2_autumn',
        'WF2_winter',
    ]
    collect_dir = 'collected results'
    os.makedirs(collect_dir, exist_ok=True)
    record = pd.DataFrame()
    structures = pd.DataFrame()
    for result_name in result_list:
        save_dir = os.path.join(collect_dir, result_name)
        # generate_single_start_summary(result_name, save_dir)
        # get best model id
        sql_file = path_join(result_name, 'model.db')
        # check if the file exist
        if not os.path.exists(sql_file):
            continue

        model_id = get_top_n_model_info(sql_file, 1)[0]
        os.makedirs(save_dir, exist_ok=True)
        best_model_dir = path_join(result_name, str(model_id))
        pred_table = pd.read_csv(path_join(best_model_dir, 'best_pred.csv'))
        record.loc[result_name, 'rmse'] = RMSE(pred_table['truth'], pred_table['pred'])
        record.loc[result_name, 'rmsle'] = RMSLE(pred_table['truth'], pred_table['pred'])
        record.loc[result_name, 'mae'] = MAE(pred_table['truth'], pred_table['pred'])
        shutil.copytree(best_model_dir, path_join(save_dir, f'best {model_id}'), dirs_exist_ok=True)
        shutil.copyfile(sql_file, path_join(save_dir, 'model.db'))
        transformation = pd.read_csv(path_join(best_model_dir, 'transformation.csv'))
        # dot = graphviz.Digraph(result_name)
        with open(path_join(best_model_dir, 'NasModel.txt')) as f:
            model_str = f.readline().replace('dense', 'fc')
            record.loc[result_name, 'structure'] = model_str

        for model_index in range(transformation.shape[0]):
            id = transformation.loc[model_index, 'current']
            print(f'copy model {id}')
            shutil.copytree(best_model_dir, path_join(save_dir, str(id)), dirs_exist_ok=True)
            # model_dir = path_join(result_name, str(id))
            # if not os.path.exists(model_dir):
            #     continue
            # with open(path_join(model_dir, 'NasModel.txt')) as f:
            #     model_str = f.readline().replace('dense', 'fc')
            # components = model_str.split('->')
            # sub_graph = graphviz.Digraph(str(model_index))
            # for i in range(len(components)):
            #     dot.node(str(model_index) + str(i), label=components[i], shape='box')
            #     if i > 0:
            #         dot.edge(str(model_index) + str(i - 1), str(model_index) + str(i))
            # dot.subgraph(sub_graph)
            # dot.render(path_join(save_dir, f'structure evolve'), format='pdf')

        record.to_csv(path_join(collect_dir, 'record.csv'))


def collect_seq_transformer_result():
    slide_total_num = 5
    # result_root_dir = '../seq2seq compare/Informer_compare'
    # save_dir = 'seq2seq compare/Informer_compare'
    result_root_dir = '../seq2seq compare/Transformer_compare'
    save_dir = 'seq2seq compare/Transformer_compare'
    task_name_list = [
        'ME',
        'WF2',
    ]
    sliding_size = 168
    test_size = 168
    predictLength = 168
    os.makedirs(save_dir, exist_ok=True)
    for task_name in task_name_list:
        record = pd.DataFrame()
        for step in range(slide_total_num):
            window_start = step * sliding_size
            task_dir = os.path.join(result_root_dir, task_name, f'step {step} window start {window_start}')
            pred_file = os.path.join(result_root_dir, task_name, f'step {step} window start {window_start}_pred.csv')
            pred_table = pd.read_csv(pred_file, index_col=0)
            pred_table = pred_table.to_numpy()
            pred_table.resize((test_size, predictLength, 2))
            # os.makedirs(os.path.join(save_dir, task_name), exist_ok=True)
            # for i in range(predictLength):
            #     record.loc[task_dir, f'RMSE {i + 1}'] = RMSE(pred_table[:, i, 0], pred_table[:, i, 1])
            # for i in range(predictLength):
            #     record.loc[task_dir, f'MAE {i + 1}'] = MAE(pred_table[:, i, 0], pred_table[:, i, 1])
            # for i in range(predictLength):
            #     record.loc[task_dir, f'RMSLE {i + 1}'] = RMSLE(pred_table[:, i, 0], pred_table[:, i, 1])
            record.loc[task_dir, 'RMSE'] = RMSE(pred_table[:, :, 0], pred_table[:, :, 1])
            record.loc[task_dir, 'MAE'] = MAE(pred_table[:, :, 0], pred_table[:, :, 1])
            record.loc[task_dir, 'RMSLE'] = RMSLE(pred_table[:, :, 0], pred_table[:, :, 1])

        record.loc['average', 'RMSE'] = np.mean(record.loc[:, 'RMSE'])
        record.loc['average', 'MAE'] = np.mean(record.loc[:, 'MAE'])
        record.loc['average', 'RMSLE'] = np.mean(record.loc[:, 'RMSLE'])
        record.to_csv(path_join(save_dir, f'{task_name} record.csv'))


def collect_seq_search_result():
    slide_total_num = 5
    result_root_dir = '../seq2seq compare/IAAS_seq_compare'
    save_dir = 'seq2seq compare/IAAS_seq_compare'
    task_name_list = [
        'ME',
        'WF2',
    ]
    sliding_size = 168
    test_size = 168
    predictLength = 168
    os.makedirs(save_dir, exist_ok=True)
    for task_name in task_name_list:
        record = pd.DataFrame()
        for step in range(slide_total_num):
            window_start = step * sliding_size
            task_dir = os.path.join(result_root_dir, task_name, f'step {step} window start {window_start}')
            if not os.path.exists(task_dir):
                continue
            save_dir_task = os.path.join(save_dir, task_name, f'step {step} window start {window_start}')
            # get best model id
            sql_file = path_join(task_dir, 'model.db')
            model_id = get_top_n_model_info(sql_file, 1)[0]
            os.makedirs(save_dir_task, exist_ok=True)
            best_model_dir = path_join(task_dir, f'best')
            shutil.copytree(best_model_dir, path_join(save_dir_task, f'best'), dirs_exist_ok=True)
            shutil.copyfile(sql_file, path_join(save_dir_task, 'model.db'))
            pred_table = pd.read_csv(path_join(best_model_dir, 'best_pred.csv'), index_col=0)
            pred_table = pred_table.to_numpy()
            pred_table.resize((test_size, predictLength, 2))
            # for i in range(predictLength):
            #     record.loc[task_dir, f'RMSE {i + 1}'] = RMSE(pred_table[:, i, 0], pred_table[:, i, 1])
            # for i in range(predictLength):
            #     record.loc[task_dir, f'MAE {i + 1}'] = MAE(pred_table[:, i, 0], pred_table[:, i, 1])
            # for i in range(predictLength):
            #     record.loc[task_dir, f'RMSLE {i + 1}'] = RMSLE(pred_table[:, i, 0], pred_table[:, i, 1])
            # with open(path_join(best_model_dir, 'NasModel.txt')) as f:
            #     model_str = f.readline().replace('dense', 'fc')
            #     record.loc[task_dir, 'structure'] = model_str
            record.loc[task_dir, 'RMSE'] = RMSE(pred_table[:, :, 0], pred_table[:, :, 1])
            record.loc[task_dir, 'MAE'] = MAE(pred_table[:, :, 0], pred_table[:, :, 1])
            record.loc[task_dir, 'RMSLE'] = RMSLE(pred_table[:, :, 0], pred_table[:, :, 1])

        record.loc['average', 'RMSE'] = np.mean(record.loc[:, 'RMSE'])
        record.loc['average', 'MAE'] = np.mean(record.loc[:, 'MAE'])
        record.loc['average', 'RMSLE'] = np.mean(record.loc[:, 'RMSLE'])
        record.to_csv(path_join(save_dir, f'{task_name} record.csv'))


def generate_random_search_result():
    search_process_dict = {
        'RL': 'IAAS',
        'Random Search': 'Random Search',
    }
    save_report_dir = 'random_search_compare'
    os.makedirs(save_report_dir, exist_ok=True)
    result_list = [
        'ME_spring',
        'ME_summer',
        'ME_autumn',
        'ME_winter',
        # 'NH_spring',
        # 'NH_summer',
        # 'NH_autumn',
        # 'NH_winter',
        'WF1_spring',
        'WF1_summer',
        'WF1_autumn',
        'WF1_winter',
        # 'WF2_spring',
        # 'WF2_summer',
        # 'WF2_autumn',
        # 'WF2_winter',
    ]
    result_dict = {
        'ME': [
            'ME_spring',
            'ME_summer',
            'ME_autumn',
            'ME_winter',
        ],
        # 'NH': [
        #     'NH_spring',
        #     'NH_summer',
        #     'NH_autumn',
        #     'NH_winter',
        # ],
        'WF1': [
            'WF1_spring',
            'WF1_summer',
            'WF1_autumn',
            'WF1_winter',
        ],
        # 'WF2': [
        #     'WF2_spring',
        #     'WF2_summer',
        #     'WF2_autumn',
        #     'WF2_winter',
        # ]
    }
    net_numbers = 600
    plt.figure(figsize=(10, 10))
    record = pd.DataFrame()
    # collect search process data
    for result_types in result_dict.keys():
        plt.clf()
        save_fig_name = os.path.join(save_report_dir, f'{result_types} random search')
        result_col = 0
        for result_name in result_dict[result_types]:
            result_col += 1
            plt.subplot(4, 1, result_col)
            plt.title(result_name)
            for search_process in search_process_dict.keys():
                sql_file = path_join(search_process_dict[search_process], result_name, 'model.db')
                search_info = get_all_model_info(sql_file)
                if len(search_info) < net_numbers:
                    to_add = net_numbers - len(search_info)
                    search_info += [search_info[1]] * to_add
                search_performance = np.array([i[1] for i in search_info])[:net_numbers]
                top_performance = [search_performance[:i + 1].min() for i in range(search_performance.shape[0])]
                record.loc[f'{result_name} {search_process}', :] = pd.DataFrame(top_performance)
                plt.plot(top_performance, label=search_process, alpha=0.9)
                if result_col == 1:
                    plt.legend(loc='upper right')
                if result_col != 4:
                    plt.xticks([])
        plt.subplots_adjust(top=0.949,
                            bottom=0.108,
                            left=0.064,
                            right=0.974,
                            hspace=0.27,
                            wspace=0.155)
        plt.savefig(save_fig_name)
        plt.savefig(save_fig_name + '.pdf')


def generate_seq2seq_barplot():
    record_file = 'seq2seq compare/seq2seq result.csv'
    save_dir = 'seq2seq compare/seq2seq result figure one day'
    os.makedirs(save_dir, exist_ok=True)
    size = 10
    plt.figure(figsize=(1 * size, 1 * size))
    data = pd.read_csv(record_file)
    filter_conditions = [
        ('RMSE', 'ME'),
        ('RMSE', 'WF1'),
        ('MAE', 'ME'),
        ('MAE', 'WF1'),
        ('RMSLE', 'ME'),
    ]
    for metric, place in filter_conditions:
        sns.barplot(data.loc[data.loc[:, 'metric'] == metric].loc[data.loc[:, 'place'] == place], x='method',
                    y='cost', )
        plt.ylabel(metric, fontsize=fontsize)
        plt.xlabel('')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(path_join(save_dir, f'seq2seq {metric} {place}.pdf'))
        plt.clf()


def generate_sensitive_analysis():
    # 生成敏感性分析结果
    figsize = (10, 10)
    result_dir_dict = {
        'no random net': '../no random net',
        'no random net no heuristic': '../no random net no heuristic',
        'no random net random drop': '../no random net random drop',
    }
    result_dict = {
        'ME': [
            'ME_spring',
            'ME_summer',
            'ME_autumn',
            'ME_winter',
        ],
        'WF1': [
            'WF1_spring',
            'WF1_summer',
            'WF1_autumn',
            'WF1_winter',
        ],
    }

    generate_type = 'sensitive analysis'  # 2 options
    # generate_type = 'episode sensitive analysis'
    save_dir = 'sensitive analysis'
    os.makedirs(save_dir, exist_ok=True)

    # 搜集模型参数数据
    collect_model_parameters_data(result_dict, result_dir_dict, save_dir)

    # 获取搜索过程中的pool信息

    if generate_type == 'episode sensitive analysis':
        episode_sensitive_table = pd.DataFrame()
    for dataset in result_dict.keys():
        for method in result_dir_dict.keys():
            plt.clf()
            fig, ax = plt.subplots(4, 1, constrained_layout=True, figsize=(12, 12))
            for dataset_number in range(len(result_dict[dataset])):
                record = pd.DataFrame()
                item = result_dict[dataset][dataset_number]
                result_dir = path_join(result_dir_dict[method], item)
                pool_statistic_file = path_join(result_dir, 'pool_statistic.json')
                if not os.path.exists(pool_statistic_file):
                    print(f'file {pool_statistic_file} not exist')
                    continue

                # 读取pool信息
                with open(pool_statistic_file, 'r') as f:
                    pool_statistic = json.load(f)

                # 预处理pool信息
                s = []
                performance_record = pd.DataFrame()
                for episode in range(len(pool_statistic)):
                    for model_id, model_performance in zip(pool_statistic[episode]['id'],
                                                           pool_statistic[episode]['performance']):
                        if model_id in performance_record.index:
                            performance_record.loc[model_id, 'performance'] = min(
                                performance_record.loc[model_id, 'performance'], model_performance)
                        else:
                            performance_record.loc[model_id, 'performance'] = model_performance
                    s.append({
                        'pool_size': pool_statistic[episode]['pool_size'],
                        'id': performance_record.sort_values(by='performance').iloc[
                              :pool_statistic[episode]['pool_size'], :].index.tolist(),
                        'performance':
                            performance_record.sort_values(by='performance').iloc[:pool_statistic[episode]['pool_size'],
                            :]['performance'].tolist(),
                    })
                pool_statistic = s

                # 统计搜索过程信息
                record = pd.DataFrame()
                for episode in range(200):
                    record_episode = episode
                    if episode >= len(pool_statistic):
                        episode = len(pool_statistic) - 1
                    for model_id, model_performance in zip(pool_statistic[episode]['id'],
                                                           pool_statistic[episode]['performance']):
                        record = pandas.concat([record, pd.DataFrame(index=[0], data={
                            'episode': record_episode,
                            'RMSE': model_performance,
                            'type': 'pool average',
                            'id': model_id,
                        })], ignore_index=True)
                    best_index = np.argmin(pool_statistic[episode]['performance'])
                    best_id = pool_statistic[episode]['id'][best_index]
                    record = pandas.concat([record, pd.DataFrame(index=[0], data={
                        'episode': record_episode,
                        'RMSE': pool_statistic[episode]['performance'][best_index],
                        'type': 'best',
                        'id': best_id,
                    })], ignore_index=True)

                record.to_csv(path_join(save_dir, f'{method} {item} pool_statistic.csv'))

                if generate_type == 'sensitive analysis':
                    # 绘制搜索过程信息 sensitive analysis
                    sns.lineplot(ax=ax[dataset_number], data=record, x='episode', y='RMSE', hue='type', errorbar='sd',
                                 color='red')
                    plt.subplots_adjust(top=0.964,
                                        bottom=0.088,
                                        left=0.074,
                                        right=0.974,
                                        hspace=0.43,
                                        wspace=0.13)
                elif generate_type == 'episode sensitive analysis':
                    table_index = [40, 80, 120, 160, 200]
                    for i in table_index:
                        best_model_info = record.where(record.loc[:, 'episode'] == i - 1).dropna().where(
                            record.loc[:, 'type'] == 'best').dropna()
                        model_id = int(best_model_info.iloc[0]['id'])
                        best_pred = pd.read_csv(path_join(result_dir, str(model_id), 'best_pred.csv'))
                        episode_sensitive_table.loc[f'{item} RMSE', i] = RMSE(best_pred['truth'], best_pred['pred'])
                        episode_sensitive_table.loc[f'{item} MAE', i] = MAE(best_pred['truth'], best_pred['pred'])
                        episode_sensitive_table.loc[f'{item} RMSLE', i] = RMSLE(best_pred['truth'], best_pred['pred'])
            if generate_type == 'sensitive analysis':
                plt.savefig(path_join(save_dir, f'{method} {dataset} pool_statistic.pdf'))
                plt.savefig(path_join(save_dir, f'{method} {dataset} pool_statistic.png'))
            elif generate_type == 'episode sensitive analysis':
                plt.savefig(path_join(save_dir, f'{method} {dataset} episode sensitive analysis.pdf'))
                plt.savefig(path_join(save_dir, f'{method} {dataset} episode sensitive analysis.png'))

            # plt.show()
    if generate_type == 'episode sensitive analysis':
        episode_sensitive_table.to_csv(path_join(save_dir, 'episode sensitive analysis.csv'), float_format='%.3f')


def generate_time_analysis():
    # 生成时间消耗分析结果
    figsize = (10, 10)
    result_dir_dict = {
        'Random Search': 'Random Search',
        'RL': 'IAAS',
    }
    result_dict = {
        'ME': [
            'ME_spring',
            'ME_summer',
            'ME_autumn',
            'ME_winter',
        ],
        'WF1': [
            'WF1_spring',
            'WF1_summer',
            'WF1_autumn',
            'WF1_winter',
        ],
    }
    save_dir = 'time analysis'
    os.makedirs(save_dir, exist_ok=True)
    generate_average_epoch_time(figsize, result_dict, result_dir_dict, save_dir)

    # 搜集模型参数数据
    collect_model_parameters_data(result_dict, result_dir_dict, save_dir)
    generate_parameters_data_report(result_dict, result_dir_dict, save_dir, figsize)


def generate_parameters_data_report(result_dict, result_dir_dict, save_dir, figsize):
    parameter_number = pd.DataFrame()
    parameter_search_process = pd.DataFrame()
    net_numbers = 600
    for dataset in result_dict.keys():
        for method in result_dir_dict.keys():
            method_total_params = 0
            method_total_models = 0
            for item in result_dict[dataset]:
                data = pd.read_csv(path_join(save_dir, f'{method} {item} param numbers.csv'))
                data.columns = ['id', 'parameter number']
                data.loc[:, 'method'] = method
                data.loc[:, 'dataset'] = dataset
                method_total_params += data.loc[:, 'parameter number'].sum()
                method_total_models += data.shape[0]
                for i in data.index:
                    data.loc[i, 'average parameter number'] = data.loc[:i, 'parameter number'].mean()
                parameter_search_process = pd.concat([parameter_search_process, data.iloc[:net_numbers]])
            parameter_number = parameter_number.append({
                'dataset': dataset,
                'Number of parameters': method_total_params / method_total_models,
                'method': method,
            }, ignore_index=True)

    # 生成平均参数量信息
    plt.figure(figsize=figsize)
    sns.barplot(parameter_number, x='dataset', y='Number of parameters', hue='method')
    plt.ylim(0, 70000)
    plt.savefig(path_join(save_dir, 'parameter number.pdf'))
    plt.savefig(path_join(save_dir, 'parameter number.png'))
    parameter_number.to_csv(path_join(save_dir, 'parameter number.csv'))

    # 生成搜索过程信息
    # for dataset, d in parameter_search_process.groupby('dataset'):
    #     plt.figure(figsize=figsize)
    #     sns.lineplot(d, x='id', y='average parameter number', hue='method')
    #     plt.savefig(path_join(save_dir, f'search process {dataset}.pdf'))
    #     plt.savefig(path_join(save_dir, f'search process {dataset}.png'))
    #     plt.title(dataset)


def collect_model_parameters_data(result_dict, result_dir_dict, save_dir):
    for dataset in result_dict.keys():
        for method in result_dir_dict.keys():
            for item in result_dict[dataset]:
                if os.path.exists(path_join(save_dir, f'{method} {item} param numbers.csv')):
                    print(f'file {path_join(save_dir, f"{method} {item} param numbers.csv")} exist')
                    continue
                print(f'collect {method} {item} param number count')
                record = pd.DataFrame()
                result_dir = path_join(result_dir_dict[method], item)
                sql_file = path_join(result_dir, 'model.db')
                model_info = get_all_model_info(sql_file)
                for i in model_info:
                    model_id = i[0]
                    model_dir = path_join(result_dir, str(model_id))
                    if not os.path.exists(model_dir):
                        continue
                    model_instance = torch.load(path_join(model_dir, 'NasModel.pth'))
                    param_number = torchinfo.summary(model_instance, verbose=False).total_params
                    record.loc[model_id, 'param'] = param_number
                record.to_csv(path_join(save_dir, f'{method} {item} param numbers.csv'))
                print(f'finish {method} {item} param number count')


def generate_average_epoch_time(figsize, result_dict, result_dir_dict, save_dir):
    # 生成平均每个模型每个epoch时间消耗结果
    record = pd.DataFrame()
    for dataset in result_dict.keys():
        for method in result_dir_dict.keys():
            method_total_time = 0
            method_total_epoch = 0
            for item in result_dict[dataset]:
                result_dir = path_join(result_dir_dict[method], item)
                log_file = path_join(result_dir, 'log.txt')
                sql_file = path_join(result_dir, 'model.db')
                with open(log_file, 'r') as f:
                    line = f.readlines()[-1]
                    search_time_total = float(line.split('Search time :')[-1].split()[0])
                model_info = get_all_model_info(sql_file)
                total_train_epochs = sum([i[2] for i in model_info])
                average_epoch_time = search_time_total / total_train_epochs
                # record.loc[dataset, f'{item.replace(dataset, method)} epoch time'] = average_epoch_time
                method_total_time += search_time_total
                method_total_epoch += total_train_epochs
            record = record.append({
                'dataset': dataset,
                'epoch time (s)': method_total_time / method_total_epoch,
                'method': method,
            }, ignore_index=True)
    plt.figure(figsize=figsize)
    sns.barplot(record, x='dataset', y='epoch time (s)', hue='method')
    plt.savefig(path_join(save_dir, 'epoch time.pdf'))
    plt.savefig(path_join(save_dir, 'epoch time.png'))
    record.to_csv(path_join(save_dir, 'epoch time.csv'))


def generate_RL_performance():
    """
    生成RL训练效果分析的结果
    :return:
    """
    result_dir = path_join('experiment analysis result', 'RL_performance')
    os.makedirs(result_dir, exist_ok=True)
    data_dir = 'IAAS'
    data_list = [
        'WF1_spring',
        'WF1_summer',
        'WF1_autumn',
        'WF1_winter',
        'WF2_spring',
        'WF2_summer',
        'WF2_autumn',
        'WF2_winter',
        'ME_spring',
        'ME_summer',
        'ME_autumn',
        'ME_winter',
        'NH_spring',
        'NH_summer',
        'NH_autumn',
        'NH_winter',
    ]

    for data_name in data_list:
        data_item_dir = path_join(data_dir, data_name)
        RL_performance = pd.DataFrame()

        # read database
        sql_file = path_join(data_item_dir, 'model.db')
        assert os.path.exists(sql_file)
        model_info = get_all_model_and_parent_info(sql_file)
        model_info = pd.DataFrame(model_info, columns=['ID', 'PREV_INDEX', 'LOSS'])
        model_info = model_info.set_index(model_info.loc[:, 'ID'])

        # RL performance
        RL_trans_model_info = model_info.loc[model_info.loc[:, 'PREV_INDEX'] != -1]
        RL_trans_model_info.loc[:, 'PREV_LOSS'] = model_info.loc[
            RL_trans_model_info.loc[:, 'PREV_INDEX'].values, 'LOSS'].tolist()
        improve_index = (RL_trans_model_info.loc[:, 'PREV_LOSS'] > RL_trans_model_info.loc[:, 'LOSS']).astype(
            int).tolist()
        RL_improve_cnt = np.cumsum(RL_trans_model_info.loc[:, 'PREV_LOSS'] > RL_trans_model_info.loc[:, 'LOSS'])

        window_size = 100
        RL_improve_rate = []
        for i in range(0, len(improve_index)):
            if i < window_size:
                continue
            RL_improve_rate.append(sum(improve_index[:i]) / i)

        plt.clf()
        plt.plot(RL_improve_rate)
        plt.savefig(path_join(result_dir, f'RL performance {data_name}.png'))

        print('developing')


if __name__ == '__main__':
    predict_result_dir = r'compare_pred'
    seasons = ['spring', 'summer', 'autumn', 'winter']

    sns.set_theme()
    sns.set(font_scale=1.5)
    sns.set_style("white")
    fontsize = 25
    matplotlib.rcParams.update({'font.size': fontsize})  # 改变所有字体大小，改变其他性质类似
    plt.rc('font', family='Times New Roman')
    plt.ion()

    # uncomment corresponding function to generate tables or figures.
    # note that the result directory should be set to the correct path.
    # some experiment result need manually copy to the result directory.

    # collect_grid_search_result()
    # collect_search_result()
    # generate_sensitive_analysis()
    # generate_time_analysis()
    # generate_random_search_result()
    # collect_seq_transformer_result()
    # collect_seq_search_result()
    # generate_seq2seq_barplot()
    # generate_load_report()
    # generate_wind_power_report()
    # generate_performance_boxplot()
    # generate_bubble_parameters_plot()
    # data_pattern_plot_wind()
    # data_pattern_plot_load()
    # generate_RL_performance()
