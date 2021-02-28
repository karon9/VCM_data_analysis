import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def SVR(train_X, test_X, train_Y):
    sc = preprocessing.StandardScaler()
    sc.fit(train_X)
    train_X = sc.transform(train_X)
    test_X = sc.transform(test_X)

    clf_svr = svm.SVR(kernel='linear', C=10, epsilon=2.0, verbose=True)
    clf_svr.fit(train_X, train_Y)

    y_pred = clf_svr.predict(test_X)
    return y_pred, clf_svr


if __name__ == '__main__':
    from sklearn.datasets import load_boston

    boston = load_boston()

    # 訓練データ、テストデータに分割
    X, Xtest, y, ytest = train_test_split(boston['data'], boston['target'], test_size=0.2, random_state=114514)
    y_pred = SVR(X, Xtest, y)
    print(r2_score(ytest, y_pred))
