#MNIST ディープ ラーニング プログラム

#以下のようにライブラリのインポートが必要
#pip install tensorflow
#pip install keras
#pip install keras2onnx

#Python のバージョン
#3.7

#各ライブラリのバージョン:
#TensorFlow 2.1.0
#Keras 2.2.4-tf
#keras2onnx 1.6.1

#ライブラリのインポート
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plot
import time
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx
import onnx

#主要ライブラリのバージョンの確認
tensorflow.__version__

keras.__version__

keras2onnx.__version__

#手書き文字データ (学習用、テスト用) の読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#手書き文字データの確認
len(x_train)

x_train[0:10]

x_train[0]

len(x_test)

len(y_test)

#学習用データを10だけ描画
for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_train[index].reshape(28, 28), cmap = None)

#0～255のデータを0.0～1.0の範囲に変換
max_value = 255
x_train, x_test = x_train / max_value, x_test / max_value

#変換後のデータを確認
x_train[0]

#ディープ ラーニングの各レイヤーを設定したモデルを作成
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512),
    keras.layers.Activation('relu'),
    keras.layers.Dense(512),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax')
])

#モデルをコンパイル
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

#モデルを訓練
model.fit(x_train, y_train, epochs = 20)

#モデルの評価
model.evaluate(x_test, y_test)

#テスト データの確認
y_test[0:10]

x_test[0:10]

#テスト データを10だけ描画
for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_test[index].reshape(28, 28), cmap = None)

#予想の結果
predict_results = model.predict(x_test[0:10])
predict_results

predict_result_list = list(map(lambda result: result.argmax(), predict_results))
predict_result_list

#モデルの概要の確認
model.summary()

#モデル保存用のフォルダーのパス
saved_model_path = "./saved_models/{}".format(int(time.time()))
saved_model_path

#モデルを TensorFlow のフォーマットで保存する
tensorflow.keras.models.save_model(model, saved_model_path, save_format="tf")

#モデルを Keras のフォーマットで保存する
model.save(saved_model_path + '/mnist_model.h5')

#ONNXフォーマットに変換
onnx_model = keras2onnx.convert_keras(model, model.name)

#モデルをONNXフォーマットで保存する
onnx_model_file = saved_model_path + '/mnist_model.onnx'
onnx.save_model(onnx_model, onnx_model_file)
