import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from lightgbm_learn import lightgbm
from SVR_learn import SVR
from deeplearning import deep_learning
from create_dataset import modify_dataset_AUC


def main():
    df = pd.read_csv('data.csv')

    X, Y = modify_dataset_AUC(df)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=4)

    if args.method == 'lightgbm':
        y_pred, model = lightgbm(train_X, test_X, train_Y, args)
    elif args.method == 'svr':
        y_pred, model = SVR(train_X, test_X, train_Y)
    elif args.method == 'deep':
        y_pred, model = deep_learning(train_X, test_X, train_Y)

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
