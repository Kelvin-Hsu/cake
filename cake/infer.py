"""
Inference Module.

These are the core but simple inference algorithms used by kernel embeddings.
"""
import tensorflow as tf
import numpy as np
from .data_type_def import *


def expectance(y, w, name=None):
    """
    Obtain the expectance from an empirical embedding.

    Parameters
    ----------
    y : tensorflow.Tensor
        The training outputs (n, d_y)
    w : tensorflow.Tensor
        The conditional or posterior weight matrix (n, n_q)
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        The conditional expected value of the output (n_q, d_y)
    """
    with tf.name_scope('expectance'):
        return tf.transpose(tf.matmul(tf.transpose(y), w), name=name)


def variance(y, w, name=None):
    """
    Obtain the variance from an empirical embedding.

    Parameters
    ----------
    y : tensorflow.Tensor
        The training outputs (n, d_y)
    w : tensorflow.Tensor
        The conditional or posterior weight matrix (n, n_q)
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        The conditional covariance value of the output (n_q, d_y)
    """
    with tf.name_scope('variance'):
        # Compute the expectance (d_y, n_q)
        y_q_exp = expectance(y, w)

        # Compute the expectance of squares (d_y, n_q)
        y_q_exp_sq = expectance(tf.square(y), w)

        # Compute the variance (n_q, d_y)
        return tf.subtract(y_q_exp_sq, tf.square(y_q_exp), name=name)


def clip_normalize(w, name=None):
    """
    Clip-normalise over the first axis of a tensor.

    Parameters
    ----------
    w : tensorflow.Tensor
        Any tensor
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        The clip-normalised tensor of the same size as the input
    """
    with tf.name_scope('clip_normalize'):
        w_clip = tf.clip_by_value(w, 0, np.inf)
        return tf.divide(w_clip, tf.reduce_sum(w_clip, axis=0), name=name)


def classify(p, classes=None, name=None):
    """
    Classify or predict based on a discrete probability distribution.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)
    classes : tensorflow.Tensor, optional
        The unique class labels of size (m,); the default is [0, ..., m - 1]
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        The classification predictions of size (n,)
    """
    with tf.name_scope('classify'):
        if classes is None:
            classes = tf.range(tf.shape(p)[1])
        class_index_predictions = tf.cast(tf.argmax(p, axis=1), tf_int_type)
        return tf.gather(classes, class_index_predictions, name=name)


def adjust_prob(p, name=None):
    """
    Adjust invalid probabilities for entropy computations.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)
    name : str
        Name of the output tensor

    Returns
    -------
        Discrete probability distribution of size (n, m)
    """
    with tf.name_scope('adjust_prob'):
        invalid = tf.less_equal(p, 0)
        ones = tf.cast(tf.ones(tf.shape(p)), tf_float_type)
        return tf.where(invalid, ones, p, name=name)


def entropy(p, name=None):
    """
    Compute the entropy of a discrete probability distribution.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        The entropy of size (n,)
    """
    with tf.name_scope('entropy'):
        p_adjust = adjust_prob(p)
        entropy_terms = -tf.multiply(p_adjust, tf.log(p_adjust))
        return tf.reduce_sum(entropy_terms, axis=1, name=name)


def decode_one_hot(y_one_hot, name=None):
    """
    Decode one hot encoding formats back to the original label format.

    Parameters
    ----------
    y_one_hot : tensorflow.Tensor
        The one hot encoding form of an array of labels (n, m)
    name : str
        Name of the output tensor

    Returns
    -------
    tensorflow.Tensor
        A vector of label indices (n,)
    """
    with tf.name_scope('decode_one_hot'):
        indices_1d = tf.cast(tf.range(tf.shape(y_one_hot)[1]), tf_float_type)
        indices_2d = tf.ones(tf.shape(y_one_hot)) * indices_1d
        zeros = tf.zeros(tf.shape(y_one_hot))
        y_indices = tf.reduce_sum(tf.where(y_one_hot, indices_2d, zeros), axis=1)
        return tf.cast(y_indices, tf_int_type, name=name)