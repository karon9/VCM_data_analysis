from sklearn import linear_model
import pandas as pd


def Linear_Regression(train_X, test_X, train_Y):
    model = linear_model.LinearRegression(normalize=True, )
    model.fit(train_X, train_Y)
    # 変数coefficientに係数の値を格納
    coefficient = model.coef_
    # データフレームに変換し、カラム名とインデックス名を指定
    df_coefficient = pd.DataFrame(coefficient,
                                  columns=["係数"],
                                  index=[train_X.columns])
    print(df_coefficient)
    # print(f"決定係数 : {model.score(train_X, train_Y)}")

    y_pred = model.predict(test_X)
    return y_pred, model
