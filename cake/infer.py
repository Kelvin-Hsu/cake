"""
Inference Module.

These are the core but simple inference algorithms used by kernel embeddings.
"""
import tensorflow as tf
import numpy as np
from .data_type_def import *


def clip_normalize(w, eps=1e-15, name=None):
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
        w_clip = tf.clip_by_value(w, eps, np.inf)
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