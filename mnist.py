import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plot
import time

print(tensorflow.__version__)

print(keras.__version__)

batch_size = 128
class_number = 10
epoch_number = 20

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(len(x_train))
print(x_train[0:10])
print(x_train[0])
print(len(x_test))
print(len(y_test))

for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_train[index].reshape(28, 28), cmap = None)

max_value = 255
x_train, x_test = x_train / max_value, x_test / max_value

print(x_train[0])

model = keras.models.Sequential([
    keras.layers.Flatten(),
    #keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(512, activation =tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 20)
model.evaluate(x_test, y_test)

print(y_test[0:10])
print(x_test[0:10])

for index in range(10):
    plot.subplot(2, 5, index + 1)
    plot.title("Label: " + str(index))
    plot.imshow(x_test[index].reshape(28, 28), cmap = None)

predict_results = model.predict(x_test[0:10])

print(predict_results)
predict_result_list = list(map(lambda result: result.argmax(), predict_results))
print(predict_result_list)

print(model.summary())

model.save('mnist_model.h5')

import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
tensorflow.keras.models.save_model(model, saved_model_path, True, True, save_format="tf")
print(saved_model_path)

