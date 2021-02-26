import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np


def read_csv_dir(target: str):
    csv_dir = os.path.join(os.getcwd(), 'data', 'csv')
    target_csv = f'{target}.csv'
    if target_csv in glob.glob(os.path.join(csv_dir, '*')):
        df = pd.read_csv(f'{os.path.join(csv_dir, target_csv)}')
        return df


def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    _df = pd.DataFrame()
    _df['feature_importance'] = models.feature_importance()
    _df['column'] = feat_train_df.columns
    feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column') \
                .sum()[['feature_importance']] \
                .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(len(order) * .4, 7))
    sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    fig.tight_layout()
    plt.show()
    return fig, ax


def scatterplot(df, column):
    plt.figure(figsize=(10, 5))
    df = df[df[' shares'] < 150000]
    ax = sns.scatterplot(y=' shares', x=column, data=df)
    # ax.set_xticks([0, 250, 500, 1000, 2000, 3000, 4000])

    plt.show()


def count_plot(df, column):
    def ex_create_to(df, step):
        probability_list = []
        df_ex = df.query(f'{i - step} <= {column} < {i}')
        df_group = df_ex.groupby('popularity').count()
        if len(df_group) == 1:
            if list(df_group.index) == 1:
                print(f'理論値(0.25)を超える : 全体数({df_group.values[0]}) : {list(df_group.index)} が {df_group.values[0]}個')
            else:
                print(f'理論値(0.25)を超えない : 全体数({df_group.values[0]}) : {list(df_group.index)} が {df_group.values[0]}個')
        elif len(df_group) == 0:
            pass
        elif len(df_group) == 2:
            probability = df_group.values[1] / (df_group.values[0] + df_group.values[1])
            if probability > 0.25:
                print(f'理論値(0.25)を超える : 全体数({df_group.values[0] + df_group.values[1]}) : {round(i - step, 4)} <= {probability} < {round(i, 4)}')
            else:
                print(f'理論値(0.25)を超えない : 全体数({df_group.values[0] + df_group.values[1]}) : {round(i - step, 4)} <= {probability} < {round(i, 4)}')

    df['popularity'] = df['shares'].apply(lambda x: 0 if x < 2800 else 1)
    df = pd.concat([df[column], df['popularity']], axis=1)
    if column == 'global_subjectivity' or column == 'n_unique_tokens':
        for i in np.arange(0.001, 1.01, 0.05):
            ex_create_to(df, 0.05)
    if column == 'n_tokens_content' or column == 'self_reference_min_shares':
        for i in np.arange(0, 8000, 200):
            ex_create_to(df, 400)
    if column == 'average_token_length':
        for i in np.arange(0, 8, 0.25):
            ex_create_to(df, 1)


if __name__ == '__main__':
    df = pd.read_csv('OnlineNewsPopularity.csv')
    df.columns = df.columns.str.replace(" ", "")
    df = df.drop(df.index[[31037]])
    df = df[df['n_tokens_content'] != 0]

    target = 'average_token_length'
    df['popularity'] = df['shares'].apply(lambda x: 0 if x < 2800 else 1)
    count_plot(df, target)

