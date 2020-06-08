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
