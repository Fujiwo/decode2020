# decode2020
Azure Machine Learning Studio (Preview) と Python と C#/.NET によるディープ ラーニングのサンプル/チュートリアル (de:code 2020 用)

## ■ 概要:

Microsoft [de:code 2020](https://www.microsoft.com/ja-jp/events/decode/2020/) で提供されるサンプル コードです。

ローカル コンピューターまたは [Microsoft Azure](https://azure.microsoft.com) 上で [Python](https://www.python.org) による機械学習/ディープラーニングのモデルを作成し、それを [.NET](https://docs.microsoft.com/ja-jp/dotnet/) の中の [ML.NET](https://docs.microsoft.com/ja-jp/dotnet/machine-learning/) を [C#](https://docs.microsoft.com/ja-jp/dotnet/csharp/) から利用してみましょう。

チュートリアル形式で説明します。

1. Visual Studio で話題の Keras と TensorFlow を使って Python で機械学習/ディープラーニングをやってみよう
2. Azure Machine Learning を使ってクラウド上で機械学習してみよう
3. ML.NET を使って学習済みのモデルを使ったアプリケーションを作ってみよう

機械学習/ディープラーニングのモデルを作成するところから、.NET アプリケーションで使うまでの手順を、実際に手を動かしながら学ぶことができます。

※ この内容は、2020年6月9日時点のものです。

## ■ プロジェクト:

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


## ■ 開発:

それでは、開発してみましょう。
開発環境は、Visual Studio と Azure Machine Learning Studio です。

### 1. Visual Studio で話題の Keras と TensorFlow を使って Python で機械学習/ディープラーニングをやってみよう

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

※ 本サンプル コードでの、Python とライブラリーのバージョンは以下のようになっています:
* Python のバージョン: 3.7
* 各ライブラリのバージョン:
** TensorFlow 2.1.0
** Keras 2.2.4-tf
** keras2onnx 1.6.1

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


### 2. Azure Machine Learning を使ってクラウド上で機械学習してみよう

次は、このプログラムを Azure 上で実行してみましょう。

2.1 Web ブラウザーで、[Microsoft Azure Portal](https://portal.azure.com) を開き、サインインします。
(Azure の使用には、費用が掛かる場合があります)

2.2 「リソースの作成」を選択します。
![Azure Portal でリソースの作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0000.png)

2.3 検索窓に「Machine Learning」と入力して検索し、「Machine Learning」を選びます。
![Machine Learning の選択](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0001.png)

2.4 「Machine Learning」を「作成」します。
![Machine Learning の作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0002.png)

2.5 必要事項を入力します。「リソース グループ」は新規に作成しても良いでしょう。
![Machine Learning の設定](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0003.png)

2.6 Web ブラウザーで、[Azure Machine Learning Studio (Preview)](https://ml.azure.com) を開きましょう。
![Azure Machine Learning Studio](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0004.png)

2.7 「ノートブック」 を「今すぐ開始」します。
![ノートブックの開始](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0005.png)

2.8 ノートブックが開いたら、「新しいフォルダーの作成」を行います。
![新しいフォルダーの作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0006.png)

2.9 フォルダー名は好きなもので結構です。ここでは「decode2020」とします。
![フォルダー名の設定](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0007.png)

2.10 「新しいフォルダーの作成」アイコンの隣の「新しいファイルの作成」アイコンをクリックして、「mnist.ipynb」という名前で新しい Python Notebook を作成します。
![新しい Python Notebook の作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0008.png)

2.11 ノートブックが開いたら、次に「新しいコンピューティングの作成」を行います。機械学習に使う CPU や GPU を持った仮想マシンの割り当てです。
![新しいコンピューティングの作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0009.png)

2.12 新しいコンピューティングを設定して作成します。ここでは、1コアの CPU のみの小さな仮想マシンとしました。
![新しいコンピューティングの設定と作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0010.png)

2.12 3 つの方法の中からノートブックを編集する方法が選べます。ここでは、「インラインで編集 (プレビュー)」を選択しました。
![ノートブックを編集する](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0011.png)

2.13 ノートブックでは、セルと呼ばれる個々の編集領域に少しずつコードを入れていくことができます。
セルの左側にある三角のアイコンをクリックすると、セルの中のコードだけをすぐに実行してみることができます。
先ずは、必要となる Python ライブラリーの「keras2onnx」をインストールしましょう。
セルの中に、「pip install keras2onnx」と書いて実行することでインストールすることができます。
![keras2onnx のインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0012.png)

2.14 ここからは、先ほど Visual Studio 上で使用した [mnist.py](/mnist.python/mnist.py) のコードを少しずつセルに入力していきます。
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0013.png)

次々と入力しては、実行していきます。
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0014.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0015.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0016.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0017.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0018.png)

モデルを訓練するところは時間が掛かります。
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0019.png)

![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0020.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0021.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0022.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0023.png)
![Python のコードの入力](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0024.png)

全部のセルを一気に実行することもできます。
![全部のセルを一気に実行](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0025.png)

全ての実行が終わると、Visual Studio で実行したときと同様に、学習済みモデルの ONNX 形式ファイルができています。
選択してダウンロードすることも可能です。
![学習済みモデルの ONNX 形式ファイル](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0026.png)

### 3. ML.NET を使って学習済みのモデルを使ったアプリケーションを作ってみよう

最後に、学習済みモデルの ONNX 形式ファイルを .NET から利用してみましょう。

3.1 Visual Studio で「コンソール アプリ (.NET Core)」を作成します。
※ この時点での .NET Core のバージョンは 3.1 です。
![Visual Studio でコンソール アプリ (.NET Core) を作成](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0007.png)

3.2 プロジェクトに NuGet で「Microsoft.ML.OnnxRuntime」という Onnx ファイルを扱うためのライブラリーをインストールします。
![NuGet で Microsoft.ML.OnnxRuntime をインストール](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0008.png)

3.2 ソリューション エクスプローラーで、プロジェクトに「assets」というフォルダーを作成し、その中に学習済みモデル (ONNX 形式ファイル) である [assets/mnist_model.onnx](/Mnist.CSharp/assets/mnist_model.onnx) をコピーしておきます。
![学習済みモデルをコピー](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0009.png)

3.3 ソリューション エクスプローラーで ONNX ファイルを右クリックし、プロパティを設定します。
「出力ディレクトリにコピー」を「新しい場合はコピーする」に設定しておきます。
![学習済みモデルをコピー](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/vs0010.png)

3.4 プロジェクトに C# のソース コードをコピーします。

プロジェクト内のファイル構成を再掲します:

| ソース コード/ファイル | クラス | 説明 |
| --- | --- | --- |
| [Program.cs](/Mnist.CSharp/Program.cs) | Program | Main ルーチン |
| [MnistInferer.cs](/Mnist.CSharp/MnistInferer.cs) | MnistInferer | 手書き数字推論器 |
| [EnumerableExtension.cs](/Mnist.CSharp/EnumerableExtension.cs) | EnumerableExtension | 汎用拡張メソッド |
| [assets/mnist_model.onnx](/Mnist.CSharp/assets/mnist_model.onnx) | --- | 学習済みモデル (ONNX 形式ファイル) |

```cs:Program.cs
// 機械学習で作成した ONNX モデル ファイルを利用して、手書き文字の認識を行う
using System;
using System.Linq;

namespace Mnist.CSharp
{
    class Program
    {
        static void Main()
        {
            // 機械学習で作成した ONNX モデル ファイル
            var dataFileName = "mnist_model.onnx";
            // 推論器
            var mnistInference = new MnistInferer(@$"assets\{dataFileName}");
            // 数字の予想
            var result = mnistInference.Infer(Program.data);
            Console.WriteLine($"The digit is probably {result}.");
        }

        // 手書き文字データ (0～255の byte の配列を0.0～1.0の float に変換)
        static float[] data = new byte[] {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x18, 0x80, 0x8B, 0x8A, 0xBF, 0xB4, 0xFD, 0xBF, 0x8A, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2B, 0x2F, 0x78, 0xBA, 0xFC, 0xFC, 0xFD, 0xFC, 0xFC, 0xFC, 0xFC, 0xFD, 0xFC, 0xE3, 0x1D, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0xA4, 0xF6, 0xFD, 0xFC, 0xFC, 0xE3, 0xB7, 0xB8, 0xA2, 0x45, 0x45, 0x45, 0x4F, 0xE3, 0xFC, 0x2D, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46, 0xFC, 0xFC, 0xFD, 0xEB, 0x4D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0xC2, 0xE3, 0x1D, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x41, 0x89, 0xC9, 0xAE, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xB5, 0xFC, 0xB7, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x80, 0xFF, 0xF9, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x72, 0xFC, 0xFD, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86, 0xF0, 0xFC, 0x7A, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0xA6, 0xF0, 0xFC, 0xA8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x35, 0xB5, 0xFC, 0xFC, 0x74, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x42, 0xF3, 0xFF, 0xF9, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x33, 0xBA, 0xFC, 0xFC, 0xDA, 0x4B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x31, 0xE4, 0xFC, 0xFC, 0xDD, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0xE9, 0xFC, 0xE3, 0x77, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xD2, 0xFA, 0xFD, 0xE7, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0xDE, 0xFD, 0xFD, 0x9E, 0x00, 0x00, 0x00, 0x0B, 0x22, 0x76, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA1, 0xFC, 0xFC, 0x6A, 0x00, 0x09, 0x2F, 0x59, 0xCB, 0xFD, 0xF4, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA1, 0xFC, 0xFC, 0xBE, 0xB9, 0xC5, 0xFC, 0xFC, 0xDD, 0xAD, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78, 0xFC, 0xFC, 0xFC, 0xFD, 0xFC, 0xFC, 0xFC, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x36, 0x89, 0x89, 0xBE, 0x89, 0x36, 0x16, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        }.Select(x => x / (float)byte.MaxValue).ToArray();
    }
}
```

```cs:MnistInferer.cs
// 機械学習で作成した ONNX モデル ファイルを利用して、手書き文字の認識を行う
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Mnist.CSharp
{
    /// <summary>手書き数字推論器</summary>
    class MnistInferer
    {
        readonly InferenceSession session;

        public MnistInferer(string onnxModelPath) => session = new InferenceSession(onnxModelPath);

        /// <summary>数字の予想</summary>
        /// <param name="input">0～1の値の配列 (28x28、左上原点)</param>
        /// <returns>0～9 (数字)</returns>
        public int Infer(float[] input) => GetInference(input).MaximumIndex();

        IEnumerable<float> GetInference(float[] input)
            => Infer(new[] {
                         NamedOnnxValue.CreateFromTensor(
                             session.InputMetadata.First().Key,
                             new DenseTensor<float>(new Memory<float>(input), session.InputMetadata.First().Value.Dimensions)
                         )
                     });

        IEnumerable<float> Infer(IReadOnlyCollection<NamedOnnxValue> inputOnnxValues)
            => session.Run(inputOnnxValues).First().AsTensor<float>();
    }
}
```

```cs:EnumerableExtension.cs
// 汎用拡張メソッド
using System.Collections.Generic;

namespace Mnist.CSharp
{
    static class EnumerableExtension
    {
        public static int MaximumIndex(this IEnumerable<float> @this)
        {
            var maximum      = float.MinValue;
            var maximumIndex = -1;
            var index        =  0;
            foreach (var element in @this) {
                if (element > maximum) {
                    maximum      = element;
                    maximumIndex = index;
                }
                index++;
            }
            return maximumIndex;
        }
    }
}
```

3.5 実行すると、次のように表示されます。手書き数字のデータを 2 と認識することができました。
![実行結果](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/console0005.png)


## Author Info

Fujio Kojima: a software developer in Japan
* Microsoft MVP for Development Tools - Visual C# (Jul. 2005 - Dec. 2014)
* Microsoft MVP for .NET (Jan. 2015 - Oct. 2015)
* Microsoft MVP for Visual Studio and Development Technologies (Nov. 2015 - Jun. 2018)
* Microsoft MVP for Developer Technologies (Nov. 2018 - Jun. 2020)
* [MVP Profile](https://mvp.microsoft.com/en-us/PublicProfile/21482 "MVP Profile")
* [Blog (Japanese)](http://wp.shos.info "Blog (Japanese)")
* [Web Site (Japanese)](http://www.shos.info "Web Site (Japanese)")

## License

This sample is under the MIT License.




