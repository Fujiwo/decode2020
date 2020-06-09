# decode2020
Azure Machine Learning Studio (Preview) と Python と C#/.NET によるディープ ラーニングのサンプル/チュートリアル (de:code 2020 用)


[mnist.py](/mnist.python/mnist.py)
[mnist.py](https://github.com/Fujiwo/decode2020/blob/master/mnist.python/mnist.py)
![Azure Portal](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0001.png)

### 概要:

Microsoft [de:code 2020](https://www.microsoft.com/ja-jp/events/decode/2020/) で提供されるサンプル コードです。

ローカル コンピューターまたは [Microsoft Azure](https://azure.microsoft.com) 上で [Python](https://www.python.org) による機械学習/ディープラーニングのモデルを作成し、それを [.NET](https://docs.microsoft.com/ja-jp/dotnet/) の中の [ML.NET](https://docs.microsoft.com/ja-jp/dotnet/machine-learning/) を [C#](https://docs.microsoft.com/ja-jp/dotnet/csharp/) で利用して使ってみましょう。

チュートリアル形式で説明します。

1. 話題の Keras と TensorFlow を使って機械学習/ディープラーニングをやってみよう
2. Azure Machine Learning を使ってクラウド上で機械学習してみよう
3. .NET の中の ML.NET を使って学習済みのモデルを使ったアプリケーションを作ってみよう

機械学習/ディープラーニングのモデルを作成するところから、.NET アプリケーションで使うまでの手順を、実際に手を動かしながら学ぶことができます。

### プロジェクト:

本

#### [mnist.python](/mnist.python)

Python による手書き文字のディープラーニングのサンプル プログラムです。
[Visual Studio](https://visualstudio.microsoft.com) からコンソールで実行できます。
また、後述する [Azure Machine Learning Studio (Preview)](https://ml.azure.com) を使って Azure 上の [Jupyter notebook](https://jupyter.org) で実行することもできます。

学習済みのモデルを [ONNX (Open Neural Network Exchange)](https://onnx.ai) 形式などでファイル出力します。

ライブラリとして、(TensorFlow][https://www.tensorflow.org) や [Keras](https://keras.io)、[keras2onnx](https://pypi.org/project/keras2onnx/) を利用しています。

学習用のデータとしては、MNIST (*) を利用しています。

(*) MNIST (Mixed National Institute of Standards and Technology database) は、手書き数字「0〜9」の画像60,000枚と、テスト画像10,000枚を集めた、画像分類問題で人気の高い画像のデータセットです。機械学習の入門のデータセットとしてもよく使われています。

| ソース コード | 説明 |
| --- | --- |
| mnist.py | Python によるディープ ラーニング プログラム |

#### [Mnist.CShart](/Mnist.CSharp)

上記 mnist.python　で作成した ONNX のファイルを C#/.NET で読み込んで、手書き文字データを認識してみるサンプルです。

| ソース コード/ファイル | クラス | 説明 |
| --- | --- | --- |
| Program.cs | Program | Main ルーチン |
| MnistInferer.cs | MnistInferer | 手書き数字推論器 |
| EnumerableExtension.cs | EnumerableExtension | 汎用拡張メソッド |
| assets/mnist_model.onnx | --- | 学習済みモデル (ONNX 形式ファイル) |


