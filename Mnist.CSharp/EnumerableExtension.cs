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
