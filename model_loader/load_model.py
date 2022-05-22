import keras
from keras.datasets import mnist

import numpy as np

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

if __name__ == '__main__':
    (_, _), (x_test, y_test) = mnist.load_data()

    path = "./model/lenet5.h5"
    model = keras.models.load_model(path)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    x_test_new = np.expand_dims(x_test, 3)
    y_test_new = keras.utils.to_categorical(y_test, 10)

    a = np.argmax(model.predict(mnist_preprocessing(x_test_new)), axis=1)
    print(len(a))

    score = model.evaluate(x_test_new, y_test_new, verbose=0)
    print(score)
    print(model.metrics_names)
