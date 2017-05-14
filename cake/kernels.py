"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import tensorflow as tf
from .data_type_def import *


def sqdist(x_p, x_q, theta):
    """
    Pairwise squared Euclidean distanced between datasets under length scaling.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The pairwise squared Euclidean distance under length scaling (n_p, n_q)
    """
    with tf.name_scope('sqdist'):
        z_p = tf.divide(x_p, theta)  # (n_p, d)
        z_q = tf.divide(x_q, theta)  # (n_q, d)
        d_pq = tf.matmul(z_p, tf.transpose(z_q))  # (n_p, n_q)
        d_p = tf.reduce_sum(tf.square(z_p), axis=1)  # (n_p,)
        d_q = tf.reduce_sum(tf.square(z_q), axis=1)  # (n_q,)
        return d_p[:, tf.newaxis] - 2 * d_pq + d_q  # (n_p, n_q)


def linear(x_p, x_q, theta, name=None):
    """
    Define the linear kernel.

    The linear kernel does not need any hyperparameters.
    Passing hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('linear_kernel'):
        return tf.multiply(tf.square(theta), tf.matmul(x_p, tf.transpose(x_q)), name=name)


def gaussian(x_p, x_q, theta, name=None):
    """
    Define the Gaussian or squared exponential kernel.

    The hyperparameters are the length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('gaussian_kernel'):
        return tf.exp(tf.multiply(-0.5, sqdist(x_p, x_q, theta)), name=name)


def s_gaussian(x_p, x_q, theta, name=None):
    """
    Define the sensitised Gaussian or squared exponential kernel.

    The hyperparameters are the sensitivity and length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(1 + 1,), (1 + d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('s_gaussian_kernel'):
        s = theta[0]
        l = theta[1:]
        return tf.multiply(tf.square(s), gaussian(x_p, x_q, l), name=name)


def matern32(x_p, x_q, theta, name=None):
    """
    Define the Matern 3/2 kernel.

    The hyperparameters are the length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('matern32_kernel'):
        r = tf.sqrt(sqdist(x_p, x_q, theta))
        return tf.multiply(1.0 + r, tf.exp(-r), name=name)


def s_matern32(x_p, x_q, theta, name=None):
    """
    Define the Matern 3/2 kernel.

    The hyperparameters are the sensitivity and length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1 + 1,), (1 + d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('s_matern32_kernel'):
        s = theta[0]
        l = theta[1:]
        return tf.multiply(tf.square(s), matern32(x_p, x_q, l), name=name)


def kronecker_delta(y_p, y_q, *args, name=None):
    """
    Define the Kronecker delta kernel.

    The Kronecker delta kernel does not need any hyperparameters.
    Passing hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('kronecker_delta_kernel'):
        y_q_flat = tf.reshape(y_q, [-1])
        return tf.cast(tf.equal(y_p, y_q_flat), tf_float_type, name=name)
