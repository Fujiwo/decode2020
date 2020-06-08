#pip install tensorflow
#pip install keras
#pip install keras2onnx

import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plot
import time
#import os
#os.environ['TF_KERAS'] = '1'
#import keras2onnx
#import onnx
#from winmltools import convert_coreml


tensorflow.__version__

keras.__version__

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

len(x_train)

x_train[0:10]

x_train[0]

len(x_test)

len(y_test)

for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_train[index].reshape(28, 28), cmap = None)

max_value = 255
x_train, x_test = x_train / max_value, x_test / max_value

x_train[0]

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 20)

model.evaluate(x_test, y_test)

y_test[0:10]

x_test[0:10]

for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_test[index].reshape(28, 28), cmap = None)

predict_results = model.predict(x_test[0:10])
predict_results

predict_result_list = list(map(lambda result: result.argmax(), predict_results))
predict_result_list

model.summary()

saved_model_path = "./saved_models/{}".format(int(time.time()))
saved_model_path

tensorflow.keras.models.save_model(model, saved_model_path, save_format="tf")

model.save(saved_model_path + '/mnist_model.h5')

#convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

# モデルをONNXフォーマットのバイナリモデルとテキストで保存する
#save_model(onnx_model, saved_model_path + '/mnist_model.onnx')
#save_text(onnx_model, saved_model_path + '/mnist_model.txt')
model_file = saved_model_path + '/mnist_model.onnx'
onnx.save_model(onnx_model, model_file)
#keras2onnx.save_model(onnx_model, model_file)
#sess = onnxruntime.InferenceSession(model_file)

#from winmltools import convert_coreml
#model_onnx = convert_coreml(model, 7, name='mnist_model')
#from winmltools.utils import save_model
# Save the produced ONNX model in binary format
#save_model(model_onnx, model_file)
# Save the produced ONNX model in text format
#from winmltools.utils import save_text
#save_text(model_onnx, saved_model_path + '/mnist_model.txt')
