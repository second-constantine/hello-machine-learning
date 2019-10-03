import tensorflow as tf
import numpy as np


def debugg_result(model, item):
    item.shape = (1, 28, 28)
    result = model.predict(item)
    print(np.nanargmax(result))
    for x in range(0, 28):
        final = ""
        for y in range(0, 28):
            val = item[0][x][y]
            if val == 0:
                final += "0"
            else:
                final += "1"
        print(final)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model("tenserflow_mnist.tenserflow")
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
for i in range(0, 10):
    debugg_result(model, x_test[i])
