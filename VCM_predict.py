import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from lightgbm_learn import lightgbm
from SVR_learn import SVR
from deeplearning import deep_learning


def C01_to_C02_log(df):
    df_log = pd.Series(np.log10(df['C02'] - df['C12']), name='logC01toC02')
    df = pd.concat([df, df_log], axis=1)
    return df


def sum_D1_Dss(df):
    df_sum = pd.Series(df['D1'] + df['Dss'], name='sum_D1_Dss')

    df = pd.concat([df, df_sum], axis=1)
    return df


def main():
    df = pd.read_csv('data(bind_9000).csv')

    # 外れ値を削除

    df = df[df['AUCss'] < 3000]

    Y = df['AUCss']
    # カラムの消したり消さなかったり、dropは消すって意味だよ
    X = df.drop(['ID', 'AUCss'], axis=1)

    # append_log
    # X = C01_to_C02_log(X)
    # X = sum_D1_Dss(X)
    # 不必要なカラムを削除
    drop_list = []
    drop_list_C = ['C03', 'C04', 'C05', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10']
    # drop_list = drop_list + drop_list_C
    X = X.drop(drop_list, axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=4)

    if args.method == 'lightgbm':
        y_pred = lightgbm(train_X, test_X, train_Y, args)
    elif args.method == 'svr':
        y_pred = SVR(train_X, test_X, train_Y)
    elif args.method == 'deep':
        y_pred = deep_learning(train_X, test_X, train_Y)

    # 結果の出力
    df_test = pd.DataFrame([test_Y.values, y_pred, abs(test_Y.values - y_pred)], index=['実験値', '予想値', '実験値と予想値の差'])
    df_test.to_csv('result.csv', encoding='utf-8')

    r2 = r2_score(test_Y.values, y_pred)

    print(f'決定係数 : {r2}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna", help="Using optuna", action="store_true")
    parser.add_argument("-m", '--method', help="Using what of method", choices=['lightgbm', 'svr', 'deep'],
                        required=True)
    args = parser.parse_args()
    main()
