import pandas as pd


def record_data(pred, truth, save_file):
    data = pd.DataFrame({'pred': pred, 'truth': truth})
    data.to_csv(save_file)


def reshape_one_dim(data):
    return data.reshape(data.shape[0], -1, )
