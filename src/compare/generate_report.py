import os.path
import sqlite3
from datetime import datetime

import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import seaborn
import seaborn as sns
import torch

from model import NasModel

path_join = os.path.join


def RMSE(pred, truth, min_max=None):
    if min_max is not None:
        min_v, max_v = min_max
        pred = restore_scaling(pred, min_v, max_v)
        truth = restore_scaling(truth, min_v, max_v)
    return np.sqrt(np.mean(np.square(pred - truth)))


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
        # min_v, max_v = get_min_max_value_load(place)
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
                # record.loc[:, 'Real'] = restore_scaling(d.loc[:, 'truth'], min_v, max_v)
                # record.loc[:, method] = restore_scaling(d.loc[:, 'pred'], min_v, max_v)
                record.loc[:, 'Real'] = d.loc[:, 'truth']
                record.loc[:, method] = d.loc[:, 'pred']
                plt.plot(record.loc[:, f'{method}'], label=f'{method}', ls='-.')
            plt.plot(record.loc[:, 'Real'], label='Real')
            # plt.legend(loc= 'upper left')
            # plt.ylim(0, 1)
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


def plot_prediction_vix(save_dir, search_name):
    os.makedirs(save_dir, exist_ok=True)
    # plt.figure(figsize=(20,10))
    plt.clf()
    record = pd.DataFrame()
    predict_result_path = path_join(predict_result_dir, f'{search_name}_vix_19.csv')
    d = pd.read_csv(predict_result_path)
    record.loc[:, 'Real'] = d.loc[:, 'truth'] * 1
    record.loc[:, 'vix_19'] = d.loc[:, 'pred'] * 1
    plt.plot(record.loc[:, 'Real'], label='Real')
    plt.plot(record.loc[:, f'vix_19'], label=f'Prediction', ls='-.')
    plt.legend(loc='upper left')
    # plt.ylim(0, 1)
    plt.title('vix_19')
    plt.xlabel('Time(h)')
    plt.ylabel('MW')
    plt.savefig(path_join(save_dir, search_name))


def get_min_max_value_load(place):
    csv_result_path = path_join('../../data', f'load_{place}_{168}_scaled.csv')
    if not os.path.exists(csv_result_path):
        return None
    data = pd.read_csv(csv_result_path)
    data.loc[:, 'RT_Demand'] = data.loc[:, 'RT_Demand'].apply(
        lambda x: float(x.replace(',', '') if type(x) == str else x))
    min = data.loc[:, 'RT_Demand'].min()
    max = data.loc[:, 'RT_Demand'].max()
    return min, max


def generate_report_from_predict_result(metrix_f, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    record = pd.DataFrame()
    for place in places:
        min_max = get_min_max_value_load(place)
        for season in seasons:
            for method in methods:
                predict_result_path = path_join(predict_result_dir, f'{method}_{place}_{season}.csv')
                if not os.path.exists(predict_result_path):
                    continue
                d = pd.read_csv(predict_result_path)
                record.loc[f'{method}', f'{place}-{season}'] = metrix_f(d.loc[:, 'pred'], d.loc[:, 'truth'], min_max)
                record.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.3f')
    record.loc[:, 'mean'] = record.mean(axis=1)
    record.to_csv(path_join(save_dir, 'method_compare.csv'), float_format='%.3f')
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
    SELECT ID FROM MODEL  WHERE LOSS is not null ORDER BY ID
    '''
    data = conn.execute(sql_quary).fetchall()
    conn.close()
    return [i[0] for i in data]


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


def copy_predict_result_NAS_RNN(avg_count, result_type, search_name, search_type_dir, save_dir):
    global methods
    search_name_single = search_name + 'single'
    methods.append(search_name_single)

    plot_model_dir = path_join(save_dir, 'plot_model')
    plot_loss_dir = path_join(save_dir, 'plot_loss')
    os.makedirs(plot_model_dir, exist_ok=True)
    os.makedirs(plot_loss_dir, exist_ok=True)
    y_max = {'VT': [0.05, 0.4, 0.05, 0.05], 'NH': [0.05, 0.1, 0.05, 0.05], 'ME': [0.05, 0.05, 0.05, 0.05], }
    model_structure = pd.DataFrame()
    for place in places:
        min_v, max_v = get_min_max_value_load(place)
        gap = max_v - min_v
        plt.clf()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(len(seasons)):
            plt.subplot(221 + i)
            season = seasons[i]
            dot = graphviz.Digraph(f'{place}-{season}')
            search_result_dir = path_join(search_type_dir, f'{place}_{result_type}-{season}')
            sql_file = path_join(search_result_dir, 'model.db')
            predict_result_path = path_join(predict_result_dir, f'{search_name}{avg_count}_{place}_{season}.csv')
            model_ids = get_top_n_model_info(sql_file, avg_count)
            pred_tables = []
            loss_curve = None
            for model_id in model_ids:
                best_model_dir = path_join(search_result_dir, str(model_id))
                pred_table = pd.read_csv(path_join(best_model_dir, 'pred.csv'))
                pred_tables.append(pred_table)
                if loss_curve is None:  # plot on top one model
                    loss_curve = pd.read_csv(path_join(best_model_dir, 'loss.csv'))
                    plt.plot(loss_curve.iloc[:, 1] * gap)
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.title(season)
                    # plt.ylim(-0.005, y_max[place][i] * gap)
                    # draw model picture
                    transformation = pd.read_csv(path_join(best_model_dir, 'transformation.csv'))
                    for model_index in range(transformation.shape[0]):
                        id = transformation.loc[model_index, 'current']
                        model_dir = path_join(search_result_dir, str(id))
                        if not os.path.exists(model_dir):
                            continue
                        with open(path_join(model_dir, 'NasModel.txt')) as f:
                            model_str = f.readline().replace('dense', 'fc')
                        components = model_str.split('->')
                        if model_index == transformation.shape[0] - 1:
                            print(f'{search_name} {place} {season} components {len(components)}')
                            model_structure.loc[f'{place}-{season}', 'structure'] = model_str
                        sub_graph = graphviz.Digraph(str(model_index))
                        for i in range(len(components)):
                            dot.node(str(model_index) + str(i), label=components[i], shape='box')
                            if i > 0:
                                dot.edge(str(model_index) + str(i - 1), str(model_index) + str(i))
                        dot.subgraph(sub_graph)
                        # if model_index != transformation.shape[0]-1:
                        #     dot.edge(str(model_index)+str(0), str(model_index+1)+str(0))
                    # dot.render(path_join(plot_model_dir, f'{place}-{season}'), format='pdf')
            pred_avg = np.stack(pred_tables)[:, :, 2]
            pred_avg = np.average(pred_avg, axis=0)
            pred_table = pred_tables[0]
            pred_table.loc[:, 'pred'] = pred_avg
            pred_table.iloc[:, 1:].to_csv(predict_result_path)

            # later model pred
            model_id = get_top_n_model_info(sql_file, avg_count + 10)[-1]
            best_model_dir = path_join(search_result_dir, str(model_id))
            pred_table = pd.read_csv(path_join(best_model_dir, 'pred.csv'))
            predict_result_path = path_join(predict_result_dir, f'{search_name_single}_{place}_{season}.csv')
            pred_table.iloc[:, 1:].to_csv(predict_result_path)

        plt.savefig(path_join(plot_loss_dir, f'{place}_loss.pdf'))
    model_structure.to_csv(path_join(save_dir, 'structure.csv'))
    plot_prediction(path_join(save_dir, 'prediction'), search_name, avg_count)


def copy_predict_result_NAS_RNN_wind_power(avg_count, result_type, search_name, search_type_dir, save_dir):
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
            dot = graphviz.Digraph(f'{place}-{season}')
            search_result_dir = path_join(search_type_dir, f'{place}_{result_type}-{season}')
            sql_file = path_join(search_result_dir, 'model.db')
            predict_result_path = path_join(predict_result_dir, f'{search_name}{avg_count}_{place}_{season}.csv')
            model_ids = get_top_n_model_info(sql_file, avg_count)
            pred_tables = []
            loss_curve = None
            for model_id in model_ids:
                best_model_dir = path_join(search_result_dir, str(model_id))
                pred_table = pd.read_csv(path_join(best_model_dir, 'pred.csv'))
                pred_tables.append(pred_table)
                if loss_curve is None:  # plot on top one model
                    loss_curve = pd.read_csv(path_join(best_model_dir, 'loss.csv'))
                    plt.plot(loss_curve.iloc[:, 1])
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.title(season)
                    # draw model picture
                    transformation = pd.read_csv(path_join(best_model_dir, 'transformation.csv'))
                    for model_index in range(transformation.shape[0]):
                        id = transformation.loc[model_index, 'current']
                        model_dir = path_join(search_result_dir, str(id))
                        if not os.path.exists(model_dir):
                            continue
                        with open(path_join(model_dir, 'NasModel.txt')) as f:
                            model_str = f.readline().replace('dense', 'fc')
                        components = model_str.split('->')
                        if model_index == transformation.shape[0] - 1:
                            print(f'{search_name} {place} {season} components {len(components)}')
                            model_structure.loc[f'{place}-{season}', 'structure'] = model_str
                        sub_graph = graphviz.Digraph(str(model_index))
                        for i in range(len(components)):
                            dot.node(str(model_index) + str(i), label=components[i], shape='box')
                            if i > 0:
                                dot.edge(str(model_index) + str(i - 1), str(model_index) + str(i))
                        dot.subgraph(sub_graph)
                        # if model_index != transformation.shape[0]-1:
                        #     dot.edge(str(model_index)+str(0), str(model_index+1)+str(0))
                    dot.render(path_join(plot_model_dir, f'{place}-{season}'), format='pdf')
            pred_avg = np.stack(pred_tables)[:, :, 2]
            pred_avg = np.average(pred_avg, axis=0)
            pred_table = pred_tables[0]
            pred_table.loc[:, 'pred'] = pred_avg
            pred_table.iloc[:, 1:].to_csv(predict_result_path)
        plt.savefig(path_join(plot_loss_dir, f'{place}_loss.pdf'))
    model_structure.to_csv(path_join(save_dir, 'structure.csv'))
    plot_prediction(path_join(save_dir, 'prediction'), search_name, avg_count)


def summary_search_progress(save_dir, result_type, search_type):
    os.makedirs(save_dir, exist_ok=True)
    param_table = pd.DataFrame(index=range(10))
    for place in places:
        min_max = get_min_max_value_load(place)
        for i in range(len(seasons)):
            season = seasons[i]
            search_result_dir = path_join(search_type, f'{place}_{result_type}-{season}')
            sql_file = path_join(search_result_dir, 'model.db')
            model_ids = get_all_model_info(sql_file)
            metrix_list = []
            metrix_top1_list = []
            metrix_top10_avg_list = []
            param_number = []
            for model_id in model_ids:
                model_dir = path_join(search_result_dir, str(model_id))
                pred_table = pd.read_csv(path_join(model_dir, 'pred.csv'))
                loss = RMSE(pred_table.loc[:, 'pred'], pred_table.loc[:, 'truth'], min_max)
                if np.isnan(loss):
                    continue
                metrix_list.append(loss)
                metrix_top10_avg_list.append(np.average(np.sort(metrix_list)[:10]))
                metrix_top1_list.append(np.sort(metrix_list)[0])
                model_instance = torch.load(path_join(model_dir, 'NasModel.pth'))
                param_number.append(NasModel.get_param_number(model_instance))
            top_10_ids = np.argsort(metrix_list)[:10]
            param_table.loc[range(top_10_ids.shape[0]), f'{place}_{season}'] = np.array(param_number)[top_10_ids]
            param_table.to_csv(path_join(save_dir, f'top_10_param.csv'))
            print(f'{search_type} {place} {season} top 10:')
            print(param_table)
            plt.clf()
            # plt.plot(mape_list, label = 'mape')
            plt.plot(metrix_top1_list, label='top 1 rmse')
            plt.plot(metrix_top10_avg_list, label='pool rmse')
            plt.legend()
            plt.savefig(path_join(save_dir, f'{place}_{season}_progress.pdf'))


def copy_predict_result_NAS_RNN_vix(save_dir, vix_data_dir, search_name, avg_count):
    plot_model_dir = path_join(save_dir, 'plot_model')
    plot_loss_dir = path_join(save_dir, 'plot_loss')
    os.makedirs(plot_loss_dir, exist_ok=True)
    os.makedirs(plot_model_dir, exist_ok=True)
    plt.clf()
    dot = graphviz.Digraph(f'vix')
    model_structure = pd.DataFrame()

    search_result_dir = vix_data_dir
    sql_file = path_join(search_result_dir, 'model.db')
    predict_result_path = path_join(predict_result_dir, f'{search_name}_vix_19.csv')
    model_ids = get_top_n_model_info(sql_file, avg_count)
    pred_tables = []
    loss_curve = None
    for model_id in model_ids:
        best_model_dir = path_join(search_result_dir, str(model_id))
        pred_table = pd.read_csv(path_join(best_model_dir, 'pred.csv'))
        pred_tables.append(pred_table)
        if loss_curve is None:  # plot on top one model
            loss_curve = pd.read_csv(path_join(best_model_dir, 'loss.csv'))
            plt.plot(loss_curve.iloc[:, 1])
            plt.ylabel('loss')
            plt.xlabel('iteration')
            plt.title('vix_19')
            # draw model picture
            transformation = pd.read_csv(path_join(best_model_dir, 'transformation.csv'))
            for model_index in range(transformation.shape[0]):
                id = transformation.loc[model_index, 'current']
                model_dir = path_join(search_result_dir, str(id))
                if not os.path.exists(model_dir):
                    continue
                with open(path_join(model_dir, 'NasModel.txt')) as f:
                    model_str = f.readline().replace('dense', 'fc')
                components = model_str.split('->')
                if model_index == transformation.shape[0] - 1:
                    model_structure.loc[f'vix_19', 'structure'] = model_str
                sub_graph = graphviz.Digraph(str(model_index))
                for i in range(len(components)):
                    dot.node(str(model_index) + str(i), label=components[i], shape='box')
                    if i > 0:
                        dot.edge(str(model_index) + str(i - 1), str(model_index) + str(i))
                dot.subgraph(sub_graph)
                # if model_index != transformation.shape[0]-1:
                #     dot.edge(str(model_index)+str(0), str(model_index+1)+str(0))

            dot.render(path_join(plot_model_dir, f'vix_19'), format='pdf')
    pred_avg = np.stack(pred_tables)[:, :, 2]
    pred_avg = np.average(pred_avg, axis=0)
    pred_table = pred_tables[0]
    pred_table.loc[:, 'pred'] = pred_avg
    pred_table.iloc[:, 1:].to_csv(predict_result_path)
    plt.savefig(path_join(plot_loss_dir, f'vix_19_loss.pdf'))
    model_structure.to_csv(path_join(save_dir, 'structure.csv'))
    plot_prediction_vix(path_join(save_dir, 'plot_prediction'), search_name)


def generate_one_cluster_report(root, cluster_table, cluster_dir, search_result_dir, min_max, draw_tree=True):
    cur_cluster_dir = os.path.join(cluster_dir, str(root))
    os.makedirs(cur_cluster_dir, exist_ok=True)

    best_loss = 1e5
    best_model = None
    # calculate accuracy (RMSE)
    for model_index in cluster_table.index:
        data = pd.read_csv(os.path.join(search_result_dir, str(model_index), 'pred.csv'))
        loss = RMSE(data.loc[:, 'pred'], data.loc[:, 'truth'], min_max)
        cluster_table.loc[model_index, 'loss'] = loss
        if best_loss is None or best_loss > loss:
            best_loss = loss
            best_model = model_index

    if draw_tree:
        # draw transformation processes
        dot = graphviz.Digraph(f'transformation processes')
        for model_index in cluster_table.index:
            if model_index == best_model:
                dot.node(str(model_index), f'{model_index}({round(cluster_table.loc[model_index, "loss"], 4)})',
                         shape='box',
                         fillcolor='turquoise', style='filled')
            else:
                dot.node(str(model_index), f'{model_index}({round(cluster_table.loc[model_index, "loss"], 4)})',
                         shape='box')
        for model_index in cluster_table.index:
            prev_index = cluster_table.loc[model_index, 'prev']
            if prev_index != -1:
                dot.edge(str(prev_index), str(model_index))
        dot.render(path_join(cur_cluster_dir, 'transformation processes'), format='pdf')
    return {'best_loss': best_loss, 'best_model': best_model, 'cluster size': cluster_table.shape[0]}


def generate_single_start_summary(save_dir, result_type, search_type, search_name):
    """
    提取单个结构开始搜索的过程。按照初始网络进行聚类。然后对于每个类提取出最高性能
    的网络编号以及性能。并且画出此类转换的具体过程（表示变化过程的树）
    @param save_dir:
    @param result_type:
    @param search_type:
    @return:
    """
    global methods
    search_name_single = search_name + 'single'
    methods.append(search_name_single)
    os.makedirs(save_dir, exist_ok=True)
    cluster = pd.DataFrame()
    for place in places:
        min_max = get_min_max_value_load(place)
        for i in range(len(seasons)):
            season = seasons[i]
            cluster_dir = os.path.join(save_dir, 'clusters', f'{place}_{season}')
            os.makedirs(cluster_dir, exist_ok=True)
            search_result_dir = path_join(search_type, f'{place}_{result_type}-{season}')
            sql_file = path_join(search_result_dir, 'model.db')
            model_info = get_all_model_and_parent_info(sql_file)
            model_info = pd.DataFrame(model_info, columns=['ID', 'PREV_INDEX', 'LOSS'])
            model_info = model_info.set_index(model_info.loc[:, 'ID'])
            # cluster all models based on the transformation
            cluster_info = pd.DataFrame(columns=['root', 'id', 'prev', 'loss'])
            for id in model_info.index:
                cur_id = id
                loss = model_info.loc[id, 'LOSS']
                cur_prev_index = model_info.loc[id, 'PREV_INDEX']
                prev_index = cur_prev_index
                while cur_prev_index != -1:  # find root
                    cur_id = cur_prev_index
                    cur_prev_index = model_info.loc[cur_id, 'PREV_INDEX']
                cluster_info.loc[id] = {'root': cur_id, 'id': id, 'loss': loss, 'prev': prev_index}
            cluster_info = list(cluster_info.groupby('root'))
            cluster_report = []
            for root, cluster_table in cluster_info:
                report = generate_one_cluster_report(root, cluster_table, cluster_dir,
                                                     search_result_dir, min_max, draw_tree=False)
                cluster_report.append(report)
            cluster_report = pd.DataFrame(cluster_report)
            cluster_report = cluster_report.sort_values(['best_loss'])
            cluster_report.to_csv(path_join(save_dir, f'{place}_{season}.csv'))
            # save the sub optimal result
            if cluster_report.shape[0] > 1:
                suboptimal = cluster_report.sort_values(['best_loss']).iloc[1]['best_model']
            else:
                suboptimal = cluster_report.iloc[0]['best_model']
            predict_result_path = path_join(predict_result_dir, f'{search_name_single}_{place}_{season}.csv')
            data = pd.read_csv(os.path.join(search_result_dir, str(int(suboptimal)), 'pred.csv'))
            data.to_csv(predict_result_path)
            print(f'{search_name_single}\t{place}{season}\n{cluster_report}')


def generate_load_report():
    global places
    global methods
    methods = ['CNN-LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'RR', 'RES', 'RESPLUS', 'DAIN', 'SNAS-MTF']

    places = ['NH', 'ME']
    NAS_RNN = r'bin_rnn2'
    NAS_RNN_single = r'../bin_rnn_single2'
    NAS_RNN2 = r'bin_rnn_200-50'
    EAS_RNN_dir = r'bin_eas'
    EAS_no_pool_dir = r'../bin_eas_single'
    save_dir = 'load'
    for i in [1]:  # avg count
        # methods.append(f'NAS-RNN-200-20AVG{i}')
        # methods.append(f'EAS-RNN-AVG{i}')
        methods.append(f'NAS-RNN-200-50AVG{i}')
        methods.append(f'EAS-pool{i}')
        methods.append(f'NAS-RNN-200-50AVG' + 'single')
        methods.append(f'EAS-pool' + 'single')
        # methods.append(f'NAS-no-pool{i}')
        # methods.append(f'EAS_no_pool{i}')
        # copy_predict_result_NAS_RNN(i, 'scale_rnn_episode-200-20','NAS-RNN-200-20AVG', NAS_RNN, path_join(save_dir,'NAS-RNN-200-20AVG'))
        copy_predict_result_NAS_RNN(i, 'scale_rnn_noise_episode-200-50', 'NAS-RNN-200-50AVG', NAS_RNN2,
                                    path_join(save_dir, 'NAS-RNN-200-50AVG'))
        # copy_predict_result_NAS_RNN(i, 'scale_rnn_episode-50-20','EAS-pool', EAS_RNN_dir,path_join(save_dir,'EAS'))
        # copy_predict_result_NAS_RNN(i, 'rnn_episode-200-50','NAS-no-pool', NAS_RNN_single, path_join(save_dir,'NAS-no-pool'))
        # copy_predict_result_NAS_RNN(i, 'rnn_episode-100-50','EAS_no_pool', EAS_no_pool_dir, path_join(save_dir,'EAS_no_pool'))
    # generate_single_start_summary(path_join(save_dir, 'NAS-RNN-200-20AVG', 'single_start_summary'),
    #                               'scale_rnn_episode-200-20', NAS_RNN,'NAS-RNN-200-20AVG')
    # generate_single_start_summary(path_join(save_dir, 'NAS-RNN-200-50AVG', 'single_start_summary'),
    #                               'scale_rnn_noise_episode-200-50', NAS_RNN2,'NAS-RNN-200-50AVG')
    # generate_single_start_summary(path_join(save_dir, 'NAS-no-pool', 'single_start_summary'),
    #                               'scale_rnn_episode-200-50', NAS_RNN_single,'NAS-no-pool')
    # generate_single_start_summary(path_join(save_dir, 'NAS-no-pool2', 'single_start_summary'),
    #                               'scale_rnn_episode-200-50', NAS_RNN_single,'NAS-no-pool2')
    # summary_search_progress(path_join(save_dir, 'NAS-RNN-200-20AVG', 'search_progress'),
    #                         'scale_rnn_episode-200-20', NAS_RNN)
    # summary_search_progress(path_join(save_dir, 'NAS-RNN-200-50AVG','search_progress'),
    #                         'scale_rnn_noise_episode-200-50', NAS_RNN2)
    # summary_search_progress(path_join(save_dir, 'EAS','search_progress'),
    #                         'scale_rnn_episode-50-20', EAS_RNN_dir)
    # summary_search_progress(path_join(save_dir, 'NAS-no-pool', 'search_progress'),
    #                         'rnn_episode-200-50', NAS_RNN_single)
    # summary_search_progress(path_join(save_dir, 'NAS-no-pool2','search_progress'),
    #                         'scale_rnn_episode-200-50', NAS_RNN_single)
    # summary_search_progress(path_join(save_dir, 'EAS_no_pool','search_progress'),
    #                         'rnn_episode-100-50', EAS_no_pool_dir)
    generate_report_from_predict_result(RMSE, path_join(save_dir, 'RMSE'))
    generate_report_from_predict_result(MAE, path_join(save_dir, 'MAE'))
    # generate_report_from_predict_result(MAPE, path_join(save_dir, 'MAPE'))


def generate_vix_report():
    vix_report_dir = 'vix_report'
    vix_data_dir = r'../bin_vix_mape'
    methods.append(f'NAS-RNN')
    copy_predict_result_NAS_RNN_vix(vix_report_dir, vix_data_dir, 'NAS-RNN', 1)
    generate_report_from_predict_result_vix(RMSE, f'{vix_report_dir}/RMSE')
    generate_report_from_predict_result_vix(MAE, f'{vix_report_dir}/MAE')
    generate_report_from_predict_result_vix(MAPE, f'{vix_report_dir}/MAPE')


def generate_wind_power_report():
    global places
    global methods
    methods = ['CNN-LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'RR', 'RES', 'RESPLUS', 'DAIN', 'SNAS-MTF']

    places = ['大帽山', '忠门']
    wind_result_dir = '../bin_wind'
    NAS_RNN_single = r'../bin_wind_single'
    wind_eas_result_dir = '../bin_wind_eas'
    wind_eas_no_pool_dir = '../bin_wind_eas_single'
    wind_save_dir = 'wind-power'
    for i in [1]:
        methods.append(f'NAS-RNN-AVG{i}')
        methods.append(f'EAS-pool{i}')
        methods.append(f'NAS-no-pool{i}')
        methods.append(f'EAS-no-pool{i}')
        copy_predict_result_NAS_RNN_wind_power(i, 'rnn_episode-100-50', f'NAS-RNN-AVG',
                                               wind_result_dir, path_join(wind_save_dir, 'nas-rnn'))
        # copy_predict_result_NAS_RNN_wind_power(i, 'rnn_episode-100-50', f'EAS-pool',
        #                                        wind_eas_result_dir, path_join(wind_save_dir, 'eas'))
        # copy_predict_result_NAS_RNN_wind_power(i, 'rnn_episode-100-50', f'NAS-no-pool',
        #                                        NAS_RNN_single, path_join(wind_save_dir, 'NAS-no-pool'))
        # copy_predict_result_NAS_RNN_wind_power(i, 'rnn_episode-100-50', f'EAS-no-pool',
        #                                        wind_eas_no_pool_dir, path_join(wind_save_dir, 'EAS-no-pool'))
    # plot_prediction(path_join(wind_save_dir, 'nas-rnn', 'plot_prediction'), 'NAS-RNN-AVG', 1)
    # plot_prediction(path_join(wind_save_dir, 'eas', 'plot_prediction'), 'EAS-pool', 1)
    # plot_prediction(path_join(wind_eas_no_pool_dir, 'EAS-no-pool', 'plot_prediction'), 'EAS-no-pool', 1)
    # generate_single_start_summary(path_join(wind_save_dir, 'nas-rnn', 'single_start_summary'),
    #                         'rnn_episode-100-50', wind_result_dir,'NAS-RNN-AVG')
    # generate_single_start_summary(path_join(wind_save_dir, 'NAS-no-pool', 'single_start_summary'),
    #                         'rnn_episode-100-50', NAS_RNN_single,'NAS-no-pool')
    # summary_search_progress(path_join(wind_save_dir, 'NAS-no-pool', 'search_progress'),
    #                         'rnn_episode-100-50', NAS_RNN_single)
    # summary_search_progress(path_join(wind_save_dir, 'nas-rnn', 'search_progress'),
    #                         'rnn_episode-100-50', wind_result_dir)
    # summary_search_progress(path_join(wind_save_dir, 'eas', 'search_progress'),
    #                         'rnn_episode-100-50', wind_eas_result_dir)
    # summary_search_progress(path_join(wind_save_dir, 'EAS-no-pool', 'search_progress'),
    #                         'rnn_episode-100-50', wind_eas_no_pool_dir)
    generate_report_from_predict_result(RMSE, path_join(wind_save_dir, 'RMSE'))
    generate_report_from_predict_result(MAE, path_join(wind_save_dir, 'MAE'))
    # generate_report_from_predict_result(MAPE, path_join(wind_save_dir, 'MAPE'))


def generate_used_datasets():
    data_root = '../data'
    save_dir = path_join(data_root, '../../data/used dataset')
    os.makedirs(save_dir, exist_ok=True)
    place_codes = ['VT', 'NH', 'ME']
    seasons = ['spring', 'summer', 'autumn', 'winter']
    # for place in place_codes:
    #     time_length = 168
    #     csv_result_path = path_join(data_root, f'load_{place}_{time_length}_scaled.csv')
    #     origin_data = pd.read_csv(csv_result_path)
    #     origin_data.loc[:, 'HourEnding'] = origin_data.loc[:, 'HourEnding'].apply(
    #         lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    #     origin_data.loc[:, 'RT_Demand'] = origin_data.loc[:, 'RT_Demand'].apply(
    #         lambda x: float(x.replace(',', '') if type(x) == str else x))
    #     for season_idx in range(1, 5):
    #         data = origin_data.loc[origin_data.loc[:, 'HourEnding'].apply(lambda x: x.quarter) == season_idx]
    #         data = data.reset_index()
    #         data = data.loc[:, ['HourEnding', 'RT_Demand','DayOfWeek','HourOfDay']]
    #         data.to_csv(path_join(save_dir, f'{place}_{seasons[season_idx-1]}.csv'))
    #         data.describe().to_csv(path_join(save_dir, f'{place}_{seasons[season_idx-1]}_statistic.csv'))
    places = ['大帽山', '忠门']
    for place in places:
        time_length = 72
        csv_result_path = path_join(data_root, f'{place}_{time_length}.csv')
        origin_data = pd.read_csv(csv_result_path)
        origin_data.loc[:, '时间'] = origin_data.loc[:, '时间'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        for season_idx in range(1, 5):
            data = origin_data.loc[origin_data.loc[:, '时间'].apply(lambda x: x.quarter) == season_idx]
            data = data.reset_index()
            data = data.iloc[:, 2:]
            data.to_csv(path_join(save_dir, f'{place}_{seasons[season_idx - 1]}.csv'))
            data.describe().to_csv(path_join(save_dir, f'{place}_{seasons[season_idx - 1]}_statistic.csv'))


def plot_bubble_graph(save_file, acc_s_p, acc_s_np, acc_ns_p, acc_ns_np, param_s_p, param_s_np, param_ns_p,
                      param_ns_np, type='load'):
    division = 1000
    coor_plate = sns.color_palette()
    origin_param_s_p = round(param_s_p / division)
    origin_param_s_np = round(param_s_np / division)
    origin_param_ns_p = round(param_ns_p / division)
    origin_param_ns_np = round(param_ns_np / division)
    param_s_p = np.sqrt(param_s_p / division)
    param_s_np = np.sqrt(param_s_np / division)
    param_ns_p = np.sqrt(param_ns_p / division)
    param_ns_np = np.sqrt(param_ns_np / division)
    max_count = max(param_s_p, param_s_np, param_ns_p,
                    param_ns_np) * 2
    max_y = max(acc_s_p, acc_s_np, acc_ns_p, acc_ns_np)
    min_y = min(acc_s_p, acc_s_np, acc_ns_p, acc_ns_np)
    x_interval = (max_y - min_y) / 2
    x_tick_labels = ['ns_np', 'ns_p', 's_np', 's_p']
    param_s_p = x_interval * param_s_p / max_count
    param_s_np = x_interval * param_s_np / max_count
    param_ns_p = x_interval * param_ns_p / max_count
    param_ns_np = x_interval * param_ns_np / max_count
    circle4 = plt.Circle((x_interval * 1, acc_ns_np), radius=param_ns_np, label=f'IAAS_sn_{type}Net',
                         color=coor_plate[0])
    circle3 = plt.Circle((x_interval * 2, acc_ns_p), radius=param_ns_p, label=f'IAAS_s_{type}Net', color=coor_plate[1])
    circle2 = plt.Circle((x_interval * 3, acc_s_np), radius=param_s_np, label=f'IAAS_n_{type}Net', color=coor_plate[2])
    circle1 = plt.Circle((x_interval * 4, acc_s_p), radius=param_s_p, label=f'IAAS_{type}Net', color=coor_plate[3])
    ax = plt.gca()
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.annotate(f'{origin_param_ns_np}k', xy=(x_interval * 1, acc_ns_np),
                xytext=(x_interval * 1 - 0.5 * x_interval, acc_ns_np + param_ns_np * 1.5))
    ax.annotate(f'{origin_param_ns_p}k', xy=(x_interval * 2, acc_ns_p),
                xytext=(x_interval * 2 - 0.5 * x_interval, acc_ns_p + param_ns_p * 1.5))
    ax.annotate(f'{origin_param_s_np}k', xy=(x_interval * 3, acc_s_np),
                xytext=(x_interval * 3 - 0.5 * x_interval, acc_s_np + param_s_np * 1.5))
    ax.annotate(f'{origin_param_s_p}k', xy=(x_interval * 4, acc_s_p),
                xytext=(x_interval * 4 - 0.5 * x_interval, acc_s_p + param_s_p * 1.5))
    ax.set_ylim(min_y - x_interval, max_y + x_interval)
    ax.set_xlim(x_interval * 0, x_interval * 5)
    plt.legend()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel("MAE")
    # plt.ylabel("RMSE")
    # plt.title('Param Size & RMSE loss')
    plt.savefig(save_file)
    # plt.show()


def generate_bubble_parameters_plot():
    """生成表示参数量大小关系的泡泡图"""
    plot_root_dir = 'bubble_plot_rmse'
    metrix = 'RMSE'
    # plot_root_dir = 'bubble_plot_mae'
    # metrix = 'MAE'
    os.makedirs(plot_root_dir, exist_ok=True)

    # load数据的生成
    s_p = 'load/NAS-RNN-200-50AVG'
    s_np = 'load/NAS-no-pool'
    ns_p = 'load/EAS'
    ns_np = 'load/EAS_no_pool'
    # 参数量读取
    param_table_s_p = pd.read_csv(path_join(s_p, 'search_progress', 'top_10_param.csv'))
    param_table_s_np = pd.read_csv(path_join(s_np, 'search_progress', 'top_10_param.csv'))
    param_table_ns_p = pd.read_csv(path_join(ns_p, 'search_progress', 'top_10_param.csv'))
    param_table_ns_np = pd.read_csv(path_join(ns_np, 'search_progress', 'top_10_param.csv'))
    accuracy_table = pd.read_csv(path_join('load', metrix, 'method_compare.csv'))
    acc_s_p = accuracy_table.iloc[-4, -1]
    acc_ns_p = accuracy_table.iloc[-3, -1]
    acc_ns_np = accuracy_table.iloc[-2, -1]
    acc_s_np = accuracy_table.iloc[-1, -1]
    param_s_p = param_table_s_p.iloc[:, -8:].mean(axis=1).iloc[0]
    param_s_np = param_table_s_np.iloc[:, -8:].mean(axis=1).iloc[0]
    param_ns_p = param_table_ns_p.iloc[:, -8:].mean(axis=1).iloc[0]
    param_ns_np = param_table_ns_np.iloc[:, -8:].mean(axis=1).iloc[0]
    plot_bubble_graph(path_join(plot_root_dir, 'load_bubble.pdf'), acc_s_p, acc_s_np, acc_ns_p, acc_ns_np, param_s_p,
                      param_s_np, param_ns_p, param_ns_np, 'Load')
    plt.clf()
    division = 1000
    type = 'Load'
    df = pd.DataFrame({
        f'IAAS_sn_{type}Net': param_table_ns_np.iloc[0, -8:] / division,
        f'IAAS_s_{type}Net': param_table_ns_p.iloc[0, -8:] / division,
        f'IAAS_n_{type}Net': param_table_s_np.iloc[0, -8:] / division,
        f'IAAS_{type}Net': param_table_s_p.iloc[0, -8:] / division,
    })
    sns.violinplot(data=df, scale='width', width=0.8, cut=0, alpha=0.5)
    plt.ylabel('Number of Parameters (k)')
    plt.savefig(path_join(plot_root_dir, 'load_param_violin.pdf'))
    plt.clf()

    # wind power数据的生成
    s_p = 'wind-power/nas-rnn'
    s_np = 'wind-power/NAS-no-pool'
    ns_p = 'wind-power/eas'
    ns_np = 'wind-power/EAS-no-pool'
    # 参数量读取
    param_table_s_p = pd.read_csv(path_join(s_p, 'search_progress', 'top_10_param.csv'))
    param_table_s_np = pd.read_csv(path_join(s_np, 'search_progress', 'top_10_param.csv'))
    param_table_ns_p = pd.read_csv(path_join(ns_p, 'search_progress', 'top_10_param.csv'))
    param_table_ns_np = pd.read_csv(path_join(ns_np, 'search_progress', 'top_10_param.csv'))
    accuracy_table = pd.read_csv(path_join('wind-power', metrix, 'method_compare.csv'))
    acc_s_p = accuracy_table.iloc[-4, -1]
    acc_ns_p = accuracy_table.iloc[-3, -1]
    acc_s_np = accuracy_table.iloc[-2, -1]
    acc_ns_np = accuracy_table.iloc[-1, -1]
    param_s_p = param_table_s_p.iloc[:, -8:].mean(axis=1).iloc[0]
    param_s_np = param_table_s_np.iloc[:, -8:].mean(axis=1).iloc[0]
    param_ns_p = param_table_ns_p.iloc[:, -8:].mean(axis=1).iloc[0]
    param_ns_np = param_table_ns_np.iloc[:, -8:].mean(axis=1).iloc[0]
    plot_bubble_graph(path_join(plot_root_dir, 'wind_bubble.pdf'), acc_s_p, acc_s_np, acc_ns_p, acc_ns_np, param_s_p,
                      param_s_np, param_ns_p, param_ns_np, 'Wind')
    plt.clf()
    division = 1000
    type = 'Wind'
    df = pd.DataFrame({
        f'IAAS_sn_{type}Net': param_table_ns_np.iloc[0, -8:] / division,
        f'IAAS_s_{type}Net': param_table_ns_p.iloc[0, -8:] / division,
        f'IAAS_n_{type}Net': param_table_s_np.iloc[0, -8:] / division,
        f'IAAS_{type}Net': param_table_s_p.iloc[0, -8:] / division,
    })
    sns.violinplot(data=df, scale='width', width=0.8, cut=0)
    plt.ylabel('Number of Parameters (k)')
    plt.savefig(path_join(plot_root_dir, 'wind_param_violin.pdf'))
    plt.clf()


def data_pattern_plot_load():
    data_root = r'C:\Data\BaiduNetdiskWorkspace\2021年\NAS-load forecasting\load_NAS\data\used dataset'
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
            plt.ylabel('MW')
            plt.subplot(2, 2, 2)
            sns.distplot(data.loc[:, 'RT_Demand'], hist=True, bins=20, kde=True, vertical=False,
                         color=color_plate[row // 2], label=season)
            s = data.loc[:, ['RT_Demand']]
            s.columns = ['MW']
            s.loc[:, 'season'] = season
            origin_data = pd.concat([origin_data, s], axis=0)
            row += 2
        plt.subplot(2, 2, 4)
        sns.boxplot(x='season', y='MW', data=origin_data, orient="v")
        plt.xlabel('')
        plt.subplot(2, 2, 2)
        plt.xlabel('')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_join('data_description', f'data {place}.pdf'))


def data_pattern_plot_wind():
    data_root = r'C:\Data\BaiduNetdiskWorkspace\2021年\NAS-load forecasting\load_NAS\data\used dataset'
    places = ['大帽山', '忠门']
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
            plt.ylabel('MW')
            plt.subplot(2, 2, 2)
            sns.distplot(data.loc[:, data_name], hist=True, bins=20, kde=True, vertical=False,
                         color=color_plate[row // 2], label=season)
            s = data.loc[:, [data_name]]
            s.columns = ['MW']
            s.loc[:, 'season'] = season
            origin_data = pd.concat([origin_data, s], axis=0)
            row += 2
        plt.subplot(2, 2, 4)
        sns.boxplot(x='season', y='MW', data=origin_data, orient="v")
        plt.xlabel('')
        plt.subplot(2, 2, 2)
        plt.xlabel('')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_join('data_description', f'{place}.pdf'))


def generate_performance_boxplot():
    plot_root_dir = 'performance_plot'
    os.makedirs(plot_root_dir, exist_ok=True)
    # sns.set_theme()

    # load 数据的结果
    data_root = 'load'
    rmse_data = pd.read_csv(os.path.join(data_root, 'RMSE', 'method_compare.csv')).T
    rmse_data = box_preprocess(rmse_data, 'IAAS_LoadNet')
    mae_data = pd.read_csv(os.path.join(data_root, 'MAE', 'method_compare.csv')).T
    mae_data = box_preprocess(mae_data, 'IAAS_LoadNet')

    size = 8
    font_size = 20
    plt.figure(figsize=(2 * size, 1 * size))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=rmse_data, width=0.5)
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('RMSE', fontsize=font_size)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=mae_data, width=0.5)
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.title('MAE', fontsize=font_size)
    plt.subplots_adjust(top=0.92,
                        bottom=0.215,
                        left=0.045,
                        right=0.97,
                        hspace=0.195,
                        wspace=0.155)

    plt.savefig(path_join(plot_root_dir, f'load_boxplot.pdf'))
    plt.clf()

    # wind power 数据的结果
    data_root = 'wind-power'
    rmse_data = pd.read_csv(os.path.join(data_root, 'RMSE', 'method_compare.csv')).T
    rmse_data = box_preprocess(rmse_data, 'IAAS_WindNet')
    mae_data = pd.read_csv(os.path.join(data_root, 'MAE', 'method_compare.csv')).T
    mae_data = box_preprocess(mae_data, 'IAAS_WindNet')

    plt.figure(figsize=(2 * size, 1 * size))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=rmse_data, width=0.5)
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('RMSE', fontsize=font_size)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=mae_data, width=0.5)
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.title('MAE', fontsize=font_size)
    plt.subplots_adjust(top=0.92,
                        bottom=0.215,
                        left=0.045,
                        right=0.97,
                        hspace=0.195,
                        wspace=0.155)
    plt.savefig(path_join(plot_root_dir, f'wind_boxplot.pdf'))
    plt.clf()


def box_preprocess(data, method_name):
    data.columns = data.iloc[0, :]
    data = data.iloc[1:, :-3]
    col = list(data.columns)
    col[-1] = method_name
    data.columns = col
    return data


predict_result_dir = r'compare'
# places = ['大帽山','忠门']
places = ['VT', 'NH', 'ME']

seasons = ['spring', 'summer', 'autumn', 'winter']

methods = ['CNN-LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'RR', 'RES', 'RESPLUS', 'DAIN']

sns.set_theme()
sns.set_style("white")
matplotlib.rcParams.update({'font.size': 18})  # 改变所有字体大小，改变其他性质类似
plt.rc('font', family='Times New Roman')

if __name__ == '__main__':
    # generate_used_datasets()
    # generate_load_report()
    # generate_wind_power_report()
    # generate_performance_boxplot()
    generate_bubble_parameters_plot()
    # data_pattern_plot_wind()
    # data_pattern_plot_load()
