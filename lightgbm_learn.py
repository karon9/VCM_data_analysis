import lightgbm as lgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


def lightgbm(train_X, test_X, train_Y, args):
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=4)

    # データセットを生成する
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_eval = lgb.Dataset(valid_X, valid_Y, reference=lgb_train)

    if args.optuna:
        # optunaを使用
        print("Using optuna!!")
        import optuna.integration.lightgbm as lgb_optuna

        # LightGBM のハイパーパラメータ
        lgbm_params = {
            # 回帰分析
            'objective': 'regression',
            # AUC の最大化を目指す
            'metric': 'rmse',
            # Fatal の場合出力
            'verbosity': -1,
            "feature_pre_filter": False
        }

        best_params, history = {}, []

        # 上記のパラメータでモデルを学習する
        model = lgb_optuna.train(lgbm_params, lgb_train, valid_sets=lgb_eval,
                                 verbose_eval=100,  # 100イテレーション毎に学習結果出力
                                 num_boost_round=1000,  # 最大イテレーション回数指定
                                 early_stopping_rounds=100,
                                 best_params=best_params,
                                 tuning_history=history,
                                 )

        print(f'best_params : {best_params}')
        with open('optuna.txt', 'w') as f:
            print(best_params, file=f)
    else:
        best_params = {'lambda_l1': 3.89081415861961e-06, 'lambda_l2': 0.02666349731287391, 'num_leaves': 6,
                       'max_depth': -1,
                       'feature_fraction': 0.8999999999999999, 'bagging_fraction': 1.0, 'bagging_freq': 0,
                       'min_child_samples': 20,
                       'objective': 'regression', 'metric': 'rmse'}
    model = lgb.train(best_params, lgb_train, valid_sets=lgb_eval,
                      verbose_eval=50,  # 50イテレーション毎に学習結果出力
                      num_boost_round=1000,  # 最大イテレーション回数指定
                      early_stopping_rounds=100)

    # テストデータを予測する
    y_pred = model.predict(test_X, num_iteration=model.best_iteration)

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_X)
    shap.summary_plot(shap_values, train_X)

    return y_pred, model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna", help="using optuna", action="store_true")
    args = parser.parse_args()
    lightgbm()
