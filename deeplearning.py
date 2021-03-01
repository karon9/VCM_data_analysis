import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing
from sklearn.metrics import r2_score

import shap


def deep_learning(train_X, test_X, train_Y):
    sc = preprocessing.StandardScaler()
    sc.fit(train_X)
    train_X = sc.transform(train_X)
    test_X = sc.transform(test_X)

    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[train_X.shape[1]]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()

    # エポックが終わるごとにドットを一つ出力することで進捗を表示
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 1000

    history = model.fit(
        train_X, train_Y,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    plot_history(history)
    model = build_model()

    # patience は改善が見られるかを監視するエポック数を表すパラメーター
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(train_X, train_Y, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    plot_history(history)
    y_pred = model.predict(test_X).flatten()

    return y_pred, model


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    boston = load_boston()

    # 訓練データ、テストデータに分割
    X, Xtest, y, ytest = train_test_split(boston['data'], boston['target'], test_size=0.2, random_state=114514)
    y_pred = deep_learning(X, Xtest, y)
    print(r2_score(ytest, y_pred))
