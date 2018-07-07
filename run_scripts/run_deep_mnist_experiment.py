import numpy as np
import datetime
import os
from cake.mnist_classifier import MNISTLinearKernelEmbeddingClassifier
import sys


def create_mnist_data():

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    # Load the training data
    x_train = mnist.train.images
    y_train = mnist.train.labels

    # Load the validation data
    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    # Add the validation data to the training data
    x = np.concatenate((x_train, x_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)

    # Load the testing data
    x_test = mnist.test.images
    y_test = mnist.test.labels

    return x, y[:, np.newaxis], x_test, y_test[:, np.newaxis]


def parse(key, currentvalue, arg=1):
    """
    Parses the command line arguments
    1. Obtains the expected data type
    2. Checks if key is present
    3. If it is, check if the expected data type is a boolean
    4. If so, set the flag away from default and return
    5. Otherwise, obtain the 'arg'-th parameter after the key
    6. Cast the resulting string into the correct type and return
    (*) This will return the default value if key is not present
    """
    cast = type(currentvalue)
    if key in sys.argv:
        if cast == bool:
            return not currentvalue
        currentvalue = sys.argv[sys.argv.index(key) + arg]
    return cast(currentvalue)

x_train, y_train, x_test, y_test = create_mnist_data()

n_train = parse('-n_train', 60000)

seed = parse('-seed', 0)

zeta = parse('-zeta', 1.)
learning_rate = parse('-learning_rate', 0.01)
dropout = parse('-dropout', 0.5)
grad_tol = parse('-grad_tol', 0.0)
max_iter = parse('-max_iter', 10000)
n_sgd_batch = parse('-n_sgd_batch', 1000)
objective = parse('-objective', 'full')
sequential_batch = parse('-sequential_batch', False)
save_step = parse('-save_step', 1)
n_block = parse('-n_block', 10000)
memory_fraction = parse('-memory', 1.0)

x_train = x_train[:n_train]
y_train = y_train[:n_train]
n_sgd_batch = None if n_sgd_batch == 0 else n_sgd_batch

name = 'deep_mnist_%s_%d_sgd_%d' % (objective, n_train, n_sgd_batch)

now = datetime.datetime.now()
now_string = '_%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                     now.hour, now.minute, now.second)

directory = './%s%s/' % (name, now_string)
os.mkdir(directory)

print('Configurations:')
print('n_train: ', n_train)
print('seed: ', seed)
print('learning_rate: ', learning_rate)
print('dropout: ', dropout)
print('grad_tol: ', grad_tol)
print('max_iter: ', max_iter)
print('n_sgd_batch: ', n_sgd_batch)
print('objective: ', objective)
print('sequential_batch: ', sequential_batch)
print('save_step: ', save_step)
print('n_block: ', n_block)
print('name: ', name)
print('memory_fraction:', memory_fraction)
print('directory: ', directory)

config = None

if memory_fraction < 1.0:

    import tensorflow as tf

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)

kec = MNISTLinearKernelEmbeddingClassifier()
kec.initialise_deep_parameters(zeta=zeta, seed=0)
kec.fit(x_train, y_train, x_test, y_test,
        learning_rate=learning_rate,
        dropout=dropout,
        grad_tol=grad_tol,
        max_iter=max_iter,
        n_sgd_batch=n_sgd_batch,
        objective=objective,
        sequential_batch=sequential_batch,
        save_step=save_step,
        n_block=n_block,
        config=config,
        directory=directory)
kec.sess.close()
