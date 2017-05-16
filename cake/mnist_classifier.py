"""
Define the mnist linear kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from cake.infer import clip_normalize as _clip_normalize
from cake.infer import classify as _classify
from cake.infer import decode_one_hot as _decode_one_hot
from cake.data_type_def import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class MNISTLinearKernelEmbeddingClassifier():

    def __init__(self):
        """Initialise the classifier."""
        pass

    def initialise_deep_parameters(self, zeta=1., seed=0):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('parameters'):

            tf.set_random_seed(seed)
            np.random.seed(seed)

            with tf.name_scope('conditional_embedding_regularisation'):
                self.zeta_init = zeta
                self.log_zeta = tf.Variable(np.log(np.atleast_1d(self.zeta_init)).astype(np_float_type), name="log_zeta")
                self.zeta = tf.exp(self.log_zeta, name="zeta")

            with tf.name_scope('convolutional_layer_1'):
                self.w_conv_1 = weight_variable([5, 5, 1, 32])
                self.b_conv_1 = bias_variable([32])

            with tf.name_scope('convolutional_layer_2'):
                self.w_conv_2 = weight_variable([5, 5, 32, 64])
                self.b_conv_2 = bias_variable([64])

            with tf.name_scope('fully_connected_layer'):
                self.w_fc_1 = weight_variable([7 * 7 * 64, 1024])
                self.b_fc_1 = bias_variable([1024])
                self.n_features = 1024

            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf_float_type)

            self.var_list = [self.log_zeta, self.w_conv_1, self.b_conv_1, self.w_conv_2, self.b_conv_2, self.w_fc_1, self.b_fc_1]

    def features(self, x):
        """Define the features of the network."""
        with tf.name_scope('features'):

            with tf.name_scope('input_layer'):
                # (n, 28, 28, 1)
                x_image = tf.reshape(x, [-1, 28, 28, 1])

            with tf.name_scope('convolutional_layer_1'):
                # (n, 28, 28, 32)
                h_conv_1 = tf.nn.relu(conv2d(x_image, self.w_conv_1) + self.b_conv_1)
                # (n, 14, 14, 32)
                h_pool_1 = max_pool_2x2(h_conv_1)

            with tf.name_scope('convolutional_layer_2'):
                # (n, 14, 14, 64)
                h_conv_2 = tf.nn.relu(conv2d(h_pool_1, self.w_conv_2) + self.b_conv_2)
                # (n, 7, 7, 64)
                h_pool_2 = max_pool_2x2(h_conv_2)

            with tf.name_scope('fully_connected_layer'):
                # (n, 7 * 7 * 64)
                h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
                # (n, 1024)
                h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, self.w_fc_1) + self.b_fc_1)

            with tf.name_scope('dropout'):
                # (n, 1024)
                h_fc_1_dropout = tf.nn.dropout(h_fc_1, self.dropout)

            return h_fc_1_dropout

    def fit(self, x_train, y_train, x_test, y_test,
            learning_rate=0.1,
            dropout=0.5,
            grad_tol=0.0,
            max_iter=60000,
            n_sgd_batch=1000,
            objective='full',
            sequential_batch=True,
            save_step=1,
            n_block=6000,
            config=None,
            directory='./'):

        with tf.name_scope('metadata'):

            classes = np.unique(y_train)
            class_indices = np.arange(classes.shape[0])
            self.classes = tf.cast(tf.constant(classes), tf_float_type, name='classes')
            self.class_indices = tf.cast(tf.constant(class_indices), tf_int_type, name='class_indices')
            self.n_classes = classes.shape[0]
            self.n = x_train.shape[0]
            self.d = x_train.shape[1]

        with tf.name_scope('core_graph'):

            self._setup_core_graph()

        with tf.name_scope('query_graph'):

            self._setup_query_graph()

        with tf.name_scope('optimisation'):

            const = tf.cast(tf.constant(4. * np.exp(1.)), tf_float_type)

            if objective == 'full':
                self.lagrangian = self.query_cross_entropy_loss + const * self.complexity
            elif objective == 'cross_entropy_loss':
                self.lagrangian = self.query_cross_entropy_loss
            elif objective == 'complexity':
                self.lagrangian = self.complexity
            elif objective == 'cross_entropy_loss_valid':
                self.lagrangian = self.query_cross_entropy_loss_valid
            elif objective == 'full_valid':
                self.lagrangian = self.query_cross_entropy_loss_valid + const * self.complexity
            else:
                raise ValueError('No such objective named %s' % objective)

            self.grad = tf.gradients(self.lagrangian, self.var_list)

            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train = opt.minimize(self.lagrangian, var_list=self.var_list)

        # Run the optimisation
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.training_iterations = 0
        batch_grad_norm = grad_tol + 1
        np.set_printoptions(precision=2)
        test_feed_dict = {self.x_train: x_train, self.y_train: y_train, self.x_query: x_test, self.y_query: y_test, self.dropout: 1.0}
        batch_train_feed_dict = {self.x_train: x_train, self.y_train: y_train, self.x_query: x_train, self.y_query: y_train, self.dropout: dropout}
        batch_test_feed_dict = {self.x_train: x_train, self.y_train: y_train, self.x_query: x_test, self.y_query: y_test, self.dropout: 1.0}
        print('Starting Training')
        print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')

        _n_block = n_block
        assert self.n % _n_block == 0
        _n_batch = int(self.n / _n_block)
        _x_train_batches = np.reshape(x_train, (_n_batch, _n_block, 28 * 28))

        if sequential_batch:

            epoch = 0
            perm_indices = np.random.permutation(np.arange(self.n))
            print('Epoch: %d' % epoch)

        while batch_grad_norm > grad_tol and self.training_iterations < max_iter:

            # Sample the data batch for this training iteration
            if n_sgd_batch:

                if sequential_batch:

                    if int((self.training_iterations * n_sgd_batch) / self.n) > epoch:

                        epoch += 1
                        perm_indices = np.random.permutation(np.arange(self.n))
                        print('Epoch: %d' % epoch)

                    sgd_indices = np.arange(self.training_iterations * n_sgd_batch, (self.training_iterations + 1) * n_sgd_batch) % self.n
                    x_batch = x_train[perm_indices][sgd_indices]
                    y_batch = y_train[perm_indices][sgd_indices]

                else:

                    sgd_indices = np.random.choice(self.n, n_sgd_batch, replace=False)
                    x_batch = x_train[sgd_indices]
                    y_batch = y_train[sgd_indices]

                batch_train_feed_dict = {self.x_train: x_batch, self.y_train: y_batch, self.x_query: x_batch, self.y_query: y_batch, self.dropout: dropout}
                batch_test_feed_dict = {self.x_train: x_batch, self.y_train: y_batch, self.x_query: x_test, self.y_query: y_test, self.dropout: 1.0}

            # Log and save the progress every so iterations
            if self.training_iterations % save_step == 0:

                # Save the parameters
                zeta = self.sess.run(self.zeta)
                w_conv_1 = self.sess.run(self.w_conv_1)
                b_conv_1 = self.sess.run(self.b_conv_1)
                w_conv_2 = self.sess.run(self.w_conv_2)
                b_conv_2 = self.sess.run(self.b_conv_2)
                w_fc_1 = self.sess.run(self.w_fc_1)
                b_fc_1 = self.sess.run(self.b_fc_1)
                np.savez('%sparameter_info_%d.npz' % (
                directory, self.training_iterations),
                         zeta=zeta,
                         w_conv_1=w_conv_1,
                         b_conv_1=b_conv_1,
                         w_conv_2=w_conv_2,
                         b_conv_2=b_conv_2,
                         w_fc_1=w_fc_1,
                         b_fc_1=b_fc_1)

                # Save the batch training performance
                batch_train_acc = self.sess.run(self.query_accuracy, feed_dict=batch_train_feed_dict)
                batch_train_cel = self.sess.run(self.query_cross_entropy_loss, feed_dict=batch_train_feed_dict)
                batch_train_cel_valid = self.sess.run(self.query_cross_entropy_loss_valid, feed_dict=batch_train_feed_dict)
                batch_train_msp = self.sess.run(self.query_msp, feed_dict=batch_train_feed_dict)
                batch_complexity = self.sess.run(self.complexity, feed_dict=batch_train_feed_dict)
                batch_grad = self.sess.run(self.grad, feed_dict=batch_train_feed_dict)
                batch_grad_norms = np.array([np.max(np.abs(grad_i)) for grad_i in batch_grad])
                batch_grad_norm = np.max(batch_grad_norms)
                print('Step %d' % self.training_iterations,
                      '|REG:', zeta[0],
                      '|BC:', batch_complexity,
                      '|BACC:', batch_train_acc,
                      '|BCEL:', batch_train_cel,
                      '|BCELV:', batch_train_cel_valid,
                      '|BMSP:', batch_train_msp,
                      '|Batch Gradient Norms:', batch_grad_norms)

                # Save the batch testing performance
                batch_test_acc = self.sess.run(self.query_accuracy, feed_dict=batch_test_feed_dict)
                batch_test_cel = self.sess.run(self.query_cross_entropy_loss, feed_dict=batch_test_feed_dict)
                batch_test_cel_valid = self.sess.run(self.query_cross_entropy_loss_valid, feed_dict=batch_test_feed_dict)
                batch_test_msp = self.sess.run(self.query_msp, feed_dict=batch_test_feed_dict)
                print('Step %d' % self.training_iterations,
                      '|BTACC:', batch_test_acc,
                      '|BTCEL:', batch_test_cel,
                      '|BTCELV:', batch_test_cel_valid,
                      '|BTMSP:', batch_test_msp)

                _z = np.concatenate(tuple([self.sess.run(self.z_query, feed_dict={self.x_query: _x_train_batches[i], self.dropout: 1.0}) for i in range(_n_batch)]), axis=0)
                _zeta = self.sess.run(self.zeta)
                _b = self.sess.run(self.y_train_one_hot, feed_dict={self.y_train: y_train})
                _w = weights(_z, _b, _zeta)

                _z_test = self.sess.run(self.z_query, feed_dict={self.x_query: x_test, self.dropout: 1.0})
                _b_test = self.sess.run(self.y_query_one_hot, feed_dict={self.y_query: y_test})
                _p_test = np.dot(_z_test, _w)
                _p_test_valid = clip_normalize(_p_test.T).T
                _p_y_test = np.sum(_b_test * _p_test, axis=1)
                _p_y_test_valid = np.sum(_b_test * _p_test_valid, axis=1)
                _classes = self.sess.run(self.classes)
                _y_test = classify(_p_test, classes=_classes)

                test_acc = np.mean(_y_test == y_test.ravel())
                test_cel = np.mean(- np.log(np.clip(_p_y_test, 1e-15, np.inf)))
                test_cel_valid = np.mean(- np.log(np.clip(_p_y_test_valid, 1e-15, np.inf)))
                test_msp = np.mean(np.sum(_p_test, axis=1))

                print('Step %d' % self.training_iterations,
                      '|TACC:', test_acc,
                      '|TCEL:', test_cel,
                      '|TCELV:', test_cel_valid,
                      '|TMSP:', test_msp)

                np.savez('%strain_test_info_%d.npz' % (directory, self.training_iterations),
                         batch_train_acc=batch_train_acc,
                         batch_train_cel=batch_train_cel,
                         batch_train_cel_valid=batch_train_cel_valid,
                         batch_train_msp=batch_train_msp,
                         batch_complexity=batch_complexity,
                         batch_grad_norms=batch_grad_norms,
                         batch_grad_norm=batch_grad_norm,
                         batch_test_acc=batch_test_acc,
                         batch_test_cel=batch_test_cel,
                         batch_test_cel_valid=batch_test_cel_valid,
                         batch_test_msp=batch_test_msp,
                         test_acc=test_acc,
                         test_cel=test_cel,
                         test_cel_valid=test_cel_valid,
                         test_msp=test_msp)

            # Run a training step
            self.sess.run(train, feed_dict=batch_train_feed_dict)
            self.training_iterations += 1

        return self

    def _setup_core_graph(self):
        """Setup the core computational graph."""
        with tf.name_scope('train_data'):

            self.x_train = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_train')
            self.y_train = tf.placeholder(tf_float_type, shape=[None, 1], name='y_train')
            self.n_train = tf.shape(self.x_train)[0]
            self.y_train_one_hot = tf.equal(self.y_train, self.classes, name='y_train_one_hot')
            self.y_train_indices = _decode_one_hot(self.y_train_one_hot, name='y_train_indices')

        with tf.name_scope('train_features'):

            self.z_train = self.features(self.x_train)

        with tf.name_scope('regularisation_matrix'):

            i = tf.cast(tf.eye(tf.shape(self.z_train)[1]), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('weights'):

            zt = tf.transpose(self.z_train)
            z = self.z_train
            ztz_reg = tf.matmul(zt, z) + reg
            self.chol_ztz_reg = tf.cholesky(ztz_reg, name='chol_ztz_reg')
            self.weights = tf.cholesky_solve(self.chol_ztz_reg, tf.matmul(zt, tf.cast(self.y_train_one_hot, tf_float_type)), name='weights')

        with tf.name_scope('complexity'):

            self.complexity = self._define_complexity(name='complexity')

    def _setup_query_graph(self):
        """Setup the querying computational graph."""
        with tf.name_scope('query_input'):

            self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_query')
            self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')
            self.n_query = tf.shape(self.x_query)[0]
            self.y_query_one_hot = tf.equal(self.y_query, self.classes, name='y_query_one_hot')
            self.y_query_indices = _decode_one_hot(self.y_query_one_hot, name='y_query_indices')

        with tf.name_scope('query_features'):

            self.z_query = self.features(self.x_query)

        with tf.name_scope('query_decision_probabilities'):

            self.query_p = tf.matmul(self.z_query, self.weights, name='query_p')

            self.query_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.query_p)), name='query_p_valid')

        with tf.name_scope('query_predictions'):

            self.query_y = _classify(self.query_p, classes=self.classes, name='query_y')

        with tf.name_scope('query_accuracy'):

            self.query_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.query_y, tf.reshape(self.y_query, [-1])), tf_float_type), name='query_accuracy')

        with tf.name_scope('query_cross_entropy_loss'):

            y_query_one_hot = tf.cast(self.y_query_one_hot, tf_float_type)

            self.query_p_y = tf.reduce_sum(tf.multiply(y_query_one_hot, self.query_p), axis=1, name='query_p_y')

            self.query_p_y_valid = tf.reduce_sum(tf.multiply(y_query_one_hot, self.query_p_valid), axis=1, name='query_p_y_valid')

            self.query_cross_entropy_loss = tf.reduce_mean(tf_info(self.query_p_y), name='query_cross_entropy_loss')

            self.query_cross_entropy_loss_valid = tf.reduce_mean(tf_info(self.query_p_y_valid), name='query_cross_entropy_loss_valid')

        with tf.name_scope('other'):

            self.query_msp = tf.reduce_mean(tf.reduce_sum(self.query_p, axis=1), name='query_msp')

    def _define_complexity(self, name='complexity_definition'):
        """
        Define the kernel embedding classifier model complexity.

        Returns
        -------
        float
            The model complexity
        """
        with tf.name_scope(name):
            return tf.sqrt(tf.reduce_sum(tf.square(self.weights)))


def tf_info(p, eps=1e-15):
    """
    Compute information.

    Parameters
    ----------
    p : tensorflow.Tensor
        A tensor of probabilities of any shape

    Returns
    -------
    tensorflow.Tensor
        A tensor of information of the same shape as the input probabilities
    """
    return - tf.log(tf.clip_by_value(p, eps, np.inf))


from scipy.linalg import cholesky, cho_solve


def weights(z, b, zeta):

    a = np.dot(z.T, z) + z.shape[0] * zeta * np.eye(z.shape[1])

    lower = True
    l = cholesky(a, lower=lower, check_finite=True)
    w = cho_solve((l, lower), np.dot(z.T, b), check_finite=True)

    return w


def clip_normalize(w):
    """
    Use the clipping method to normalize weights.

    Parameters
    ----------
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The clip-normalized conditional or posterior weight matrix (n, n_q)
    """
    w_clip = np.clip(w, 0, np.inf)
    return w_clip / np.sum(w_clip, axis=0)


def classify(p, classes=None):
    """
    Classify or predict based on a discrete probability distribution.

    Parameters
    ----------
    p : numpy.ndarray
        Discrete probability distribution of size (n, m)
    classes : numpy.ndarray
        The unique class labels of size (m,) where the default is [0, 1, ..., m]

    Returns
    -------
    numpy.ndarray
        The classification predictions of size (n,)
    """
    if classes is None:
        classes = np.arange(p.shape[1])
    return classes[np.argmax(p, axis=1)]