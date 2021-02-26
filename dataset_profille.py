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


