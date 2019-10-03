import tensorflow
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.nn import relu as RELU, softmax as SOFTMAX

print(device_lib.list_local_devices())

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

sequential = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation=RELU),
    Dropout(0.2),
    Dense(10, activation=SOFTMAX)
])
sequential.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

sequential.fit(x_train, y_train, epochs=6)
sequential.evaluate(x_test, y_test)

sequential.save("tenserflow_mnist.tenserflow")
