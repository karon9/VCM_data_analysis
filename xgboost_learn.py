import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
import shap


def xgboost(train_X, test_X, train_Y):
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=4)

    clf = xgb.XGBRegressor()

    # ハイパーパラメータ探索
    clf_cv = model_selection.GridSearchCV(clf, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
    clf_cv.fit(train_X, train_Y.values)
    print(clf_cv.best_params_, clf_cv.best_score_)

    clf = xgb.XGBRegressor(**clf_cv.best_params_)
    clf.fit(train_X, train_Y)

    shap.initjs()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(train_X)
    shap.summary_plot(shap_values, train_X)

    # テストデータを予測する
    y_pred = clf.predict(test_X)

    return y_pred, clf
