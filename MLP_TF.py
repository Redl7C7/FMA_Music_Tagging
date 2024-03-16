import numpy as np
import tensorflow as tf
from random import random
from sklearn.model_selection import train_test_split


# Traings- und Testdaten für Training und Test
def generate_dataset(num_samples, test_size=0.33):
    """Generates train/test data for sum operation

    :param num_samples (int): Num of total samples in dataset
    :param test_size (int): Ratio of num_samples used as test set
    :return x_train (ndarray): 2d array with input data for training
    :return x_test (ndarray): 2d array with input data for testing
    :return y_train (ndarray): 2d array with target data for training
    :return y_test (ndarray): 2d array with target data for testing

    """

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":

    # create a dataset with 2000 samples
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    # print("x_test: \n {}".format(x_test))
    # print("y_test: \n {}".format(y_test))

    # Modell bauen
    model = tf.keras.Sequential()
# Modell kompilieren

# Modell trainieren

# Modell evaluieren

# Vorhersagen treffen