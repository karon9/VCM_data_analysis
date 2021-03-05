import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from lightgbm_learn import lightgbm
from SVR_learn import SVR
from deeplearning import deep_learning
from create_dataset import modify_dataset_AUC, modify_dataset_CL
from linearregression import Linear_Regression
from xgboost_learn import xgboost


# def C01_to_C02_log(df):
#     df_log = pd.Series(np.log10(df['C02'] - df['C12']), name='logC01toC02')
#     df = pd.concat([df, df_log], axis=1)
#     return df


def main():
    df = pd.read_csv('data.csv')
    X, Y = modify_dataset_CL(df)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=4)

    if args.method == 'lightgbm':
        y_pred, model = lightgbm(train_X, test_X, train_Y, args)
    elif args.method == 'svr':
        y_pred, model = SVR(train_X, test_X, train_Y)
    elif args.method == 'deep':
        y_pred, model = deep_learning(train_X, test_X, train_Y)
    elif args.method == 'linear':
        y_pred, model = Linear_Regression(train_X, test_X, train_Y)
    elif args.method == 'xgboost':
        y_pred, model = xgboost(train_X, test_X, train_Y)

    # 結果の出力
    df_test = pd.DataFrame([test_Y.values, y_pred, abs(test_Y.values - y_pred)], index=['実験値', '予想値', '実験値と予想値の差'])
    df_test.to_csv('result_CL.csv', encoding='utf-8')

    r2 = r2_score(test_Y.values, y_pred)

    test_difference = abs(test_Y.values - y_pred)
    df_test = pd.DataFrame([test_Y.values, y_pred, test_difference], index=['実験値', '予想値', '実験値と予想値の差'])
    df_test.to_csv('result_CL.csv', encoding='utf-8_sig')
    print('Output csv file')

    print(f'決定係数 : {r2}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna", help="Using optuna", action="store_true")
    parser.add_argument("-m", '--method', help="Using what of method",
                        choices=['lightgbm', 'svr', 'deep', 'linear', 'xgboost'],
                        required=True)
    args = parser.parse_args()
    main()
