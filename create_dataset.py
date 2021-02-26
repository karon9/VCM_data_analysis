import pandas as pd
import numpy as np


def C01_to_C02_log(df):
    df_log = pd.Series(np.log10(df['C02'] - df['C12']), name='logC01toC02')
    df = pd.concat([df, df_log], axis=1)
    return df


def sum_D1_Dss(df):
    df_sum = pd.Series(df['D1'] + df['Dss'], name='sum_D1_Dss')

    df = pd.concat([df, df_sum], axis=1)
    return df


def create_dataset() -> pd.DataFrame:
    df = pd.read_csv('data(bind_9000).csv')

    # 外れ値を削除
    df = df[df['AUCss'] < 3000]
    df = df[df['AUCss'] > 50]
    df = df.set_index('ID')
    df.to_csv('data.csv')


def modify_dataset_AUC(df):
    Y = df['AUCss']
    # カラムの消したり消さなかったり、dropは消すって意味だよ
    X = df.drop(['AUCss', 'ID', 'CL'], axis=1)
    # df_CL = pd.read_csv('data_CL.csv')
    # X = pd.concat([X, df_CL], axis=1)

    # append_log
    # X = C01_to_C02_log(X)
    # X = sum_D1_Dss(X)
    # 不必要なカラムを削除
    drop_list = []
    drop_list_C = ['C01', 'C02', 'C03', 'C04', 'C05', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10']
    drop_list = drop_list + drop_list_C
    X = X.drop(drop_list, axis=1)
    return X, Y


def modify_dataset_CL(df):
    Y = df['CL']
    # カラムの消したり消さなかったり、dropは消すって意味だよ
    X = df.drop(['CL', 'ID'], axis=1)

    # append_log
    # X = C01_to_C02_log(X)
    # X = sum_D1_Dss(X)
    # 不必要なカラムを削除
    drop_list = ['AUCss', 'Dss']
    drop_list_C = ['C01', 'C02', 'C03', 'C04', 'C05', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10']
    drop_list = drop_list + drop_list_C
    X = X.drop(drop_list, axis=1)
    return X, Y


if __name__ == '__main__':
    create_dataset()
