import pandas as pd
import sklearn.svm


def svr_predict(data, record: pd.DataFrame, place, season):
    X_train, y_train, X_test, y_test = data
    regr = sklearn.svm.SVR()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    column_name = f'{place}_{season}_mse'
    record.loc['SVR',column_name] = mse
    print(f'SVR recording: {column_name}: {mse}')
