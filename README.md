# decode2020
Azure Machine Learning Studio (Preview) と Python と C#/.NET によるディープ ラーニングのサンプル/チュートリアル (de:code 2020 用)


[mnist.py](/mnist.python/mnist.py)
[mnist.py](https://github.com/Fujiwo/decode2020/blob/master/mnist.python/mnist.py)
![Azure Portal](https://raw.githubusercontent.com/Fujiwo/decode2020/master/images/azure0001.png)



### プロジェクト:

#### [mnist.python](/mnist.python)

[Python](https://www.python.org) による手書き文字のディープラーニングのサンプル プログラムです。
[Visual Studio](https://visualstudio.microsoft.com) からコンソールで実行できます。
また、後述する [Azure Machine Learning Studio (Preview)](https://ml.azure.com) を使って Azure 上の [Jupyter notebook](https://jupyter.org) で実行することもできます。

学習済みのモデルを [ONNX (Open Neural Network Exchange)](https://onnx.ai) 形式などでファイル出力します。

ライブラリとして、(TensorFlow][https://www.tensorflow.org) や [Keras](https://keras.io)、[](https://pypi.org/project/keras2onnx/) を利用しています。

学習用のデータとしては、MNIST (*) を利用しています。

(*) MNIST (Mixed National Institute of Standards and Technology database) は、手書き数字「0〜9」の画像60,000枚と、テスト画像10,000枚を集めた、画像分類問題で人気の高い画像のデータセットです。機械学習の入門のデータセットとしてもよく使われています。

| ソース コード | 説明 |
| --- | --- |
| mnist.py | Python プログラム |

#### [Mnist.CShart](/Mnist.CSharp)

上記 mnist.python　で作成した ONNX のファイルを [C#](https://docs.microsoft.com/ja-jp/dotnet/csharp/)/[.NET](https://docs.microsoft.com/ja-jp/dotnet/) で読み込んで、手書き文字データを認識してみるサンプルです。

| ソース コード/ファイル | クラス | 説明 |
| --- | --- | --- |
| Program.cs | Program | Main ルーチン |
| MnistInferer.cs | MnistInferer | 手書き数字推論器 |
| EnumerableExtension.cs | EnumerableExtension | 汎用拡張メソッド |
| assets/mnist_model.onnx | --- | 学習済みモデル (ONNX 形式ファイル) |


