# decode2020
Azure Machine Learning Studio (Preview) と Python と C#/.NET によるディープ ラーニングのサンプル/チュートリアル (de:code 2020 用)

### ■ 概要:

Microsoft [de:code 2020](https://www.microsoft.com/ja-jp/events/decode/2020/) で提供されるサンプル コードです。

ローカル コンピューターまたは [Microsoft Azure](https://azure.microsoft.com) 上で [Python](https://www.python.org) による機械学習/ディープラーニングのモデルを作成し、それを [.NET](https://docs.microsoft.com/ja-jp/dotnet/) の中の [ML.NET](https://docs.microsoft.com/ja-jp/dotnet/machine-learning/) を [C#](https://docs.microsoft.com/ja-jp/dotnet/csharp/) から利用してみましょう。

チュートリアル形式で説明します。

1. 話題の Keras と TensorFlow を使って機械学習/ディープラーニングをやってみよう
2. Azure Machine Learning を使ってクラウド上で機械学習してみよう
3. .NET の中の ML.NET を使って学習済みのモデルを使ったアプリケーションを作ってみよう

機械学習/ディープラーニングのモデルを作成するところから、.NET アプリケーションで使うまでの手順を、実際に手を動かしながら学ぶことができます。

### ■ プロジェクト:

本サンプルは、以下の2つのプロジェクトからなっています。

#### 1. [mnist.python](/mnist.python)

Python による手書き文字のディープラーニングのサンプル プログラムです。
[Visual Studio](https://visualstudio.microsoft.com) からコンソールで実行できます。
また、後述する [Azure Machine Learning Studio (Preview)](https://ml.azure.com) を使って Azure 上の [Jupyter notebook](https://jupyter.org) で実行することもできます。

学習済みのモデルを [ONNX (Open Neural Network Exchange)](https://onnx.ai) 形式などでファイル出力します。

ライブラリとして、[TensorFlow](https://www.tensorflow.org) や [Keras](https://keras.io)、[keras2onnx](https://pypi.org/project/keras2onnx/) を利用しています。

学習用のデータとしては、MNIST (*) を利用しています。

(*) MNIST (Mixed National Institute of Standards and Technology database) は、手書き数字「0〜9」の画像60,000枚と、テスト画像10,000枚を集めた、画像分類問題で人気の高い画像のデータセットです。機械学習の入門のデータセットとしてもよく使われています。

| ソース コード | 説明 |
| --- | --- |
| [mnist.py](/mnist.python/mnist.py) | Python によるディープ ラーニング プログラム |

#### 2. [Mnist.CShart](/Mnist.CSharp)

上記 mnist.python　で作成した ONNX のファイルを C#/.NET で読み込んで、手書き文字データを認識してみるサンプルです。

| ソース コード/ファイル | クラス | 説明 |
| --- | --- | --- |
| [Program.cs](/Mnist.CSharp/Program.cs) | Program | Main ルーチン |
| [MnistInferer.cs](/Mnist.CSharp/MnistInferer.cs) | MnistInferer | 手書き数字推論器 |
| [EnumerableExtension.cs](/Mnist.CSharp/EnumerableExtension.cs) | EnumerableExtension | 汎用拡張メソッド |
| [assets/mnist_model.onnx](/Mnist.CSharp/assets/mnist_model.onnx) | --- | 学習済みモデル (ONNX 形式ファイル) |


### ■ 開発:

開発環境は、Visual Studio と Azure Machine Learning Studio です。

### 1. Visual Studio での Python 開発

まずは、Visual Studio で Python を使えるようにしましょう。
スタート メニューなどから、Visual Studio Installer を起動します。

1.1 「Python 開発」にチェックを入れて、「変更」します。
![Visual Studio Installer で Python 開発をインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0001.png)

1.2 Visual Studio を起動し、新しいプロジェクトとして「Python アプリケーション」を作成します。
![Visual Studio で Python アプリケーションを作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0002.png)

1.3 プロジェクトが出来上がったら、Python のライブラリーを追加します。
「ソリューション エクスプローラー」でプロジェクトの「Python 環境」の中を右クリックし、「Python パッケージの管理」を選択します。
![Python パッケージの管理](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0003.png)

先ほど右クリックした Python 環境に Python のライブラリーを3つ追加していきます。
先ずは、TensorFlow です。

1.4 検索窓に tensorflow と入力し、「次のコマンドを実行する: pip install tensorflow」をクリックします。
![TensorFlow のインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0004.png)

1.5 同様に、Keras をインストールします。
![Keras のインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0005.png)

1.6 最後に、keras2onnx をインストールします。
![keras2onnx のインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0006.png)

開発環境は以上で整いました。
コードを書いていきましょう。

1.7 プロジェクトの中にある Python のソース コード ファイル (拡張子が .py のファイル) を開け、 [mnist.py](/mnist.python/mnist.py) の中のコードに置き換えます。
処理の内容については、ソース コードをご参照ください。

```python:mnist.py
#mnist.py
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
```

1.8 Visual Studio で実行してみましょう。

コンソール画面が立ち上がります。
![mnist.python の開始画面](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0001.png)

Keras と TensorFlow によって、MNIST の学習を行います。これには、時間がかかります。
![MNIST の学習中](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0002.png)

最後に、学習済みのモデルを ONNX 形式でファイルに出力して、プログラムが終了します。
![プログラムの終了](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0003.png)

ファイル エクスプローラーで、プロジェクトのフォルダーを確認すると、学習済みモデルの ONNX 形式ファイルが出来ているのが分かります。 
![学習済みモデルの ONNX 形式ファイルが出来ている](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0004.png)


### 2. Azure Machine Learning Studio を利用する

次は、このプログラムを Azure 上で実行してみましょう。

Web ブラウザーで、[Azure Machine Learning Studio (Preview)](https://ml.azure.com) を開きましょう。

![Azure Portal](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0000.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0001.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0002.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0003.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0004.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0005.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0006.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0007.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0008.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0009.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0010.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0011.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0012.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0013.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0014.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0015.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0016.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0017.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0018.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0019.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0020.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0021.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0022.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0023.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0024.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0025.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0026.png)

### 3. Visual Studio で ML.NET を利用する

![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0007.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0008.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0009.png)
![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0010.png)

![](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0005.png)
