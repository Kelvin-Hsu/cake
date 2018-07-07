"""
Define the stationary kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from .infer import clip_normalize as _clip_normalize
from .infer import classify as _classify
from .infer import decode_one_hot as _decode_one_hot
from .kernels import s_gaussian as _s_gaussian
from .data_type_def import *


class StationaryKernelEmbeddingClassifier():

    def __init__(self, kernel=_s_gaussian, learning_objective='er+rcb', learning_rate=0.1):

        self.out_kernel = kernel
        self.learning_objective = learning_objective
        self.learning_rate = learning_rate
        self.setup = False
        self.has_test_data = False
        self.directory = None

    def initialise_parameters(self, theta, zeta):

        with tf.name_scope('parameters'):

            self.zeta_init = zeta
            log_zeta = np.log(np.atleast_1d(self.zeta_init)).astype(np_float_type)
            self.log_zeta = tf.Variable(log_zeta, name="log_zeta")
            self.zeta = tf.exp(self.log_zeta, name="zeta")

            self.theta_init = theta
            log_theta = np.log(np.atleast_1d(self.theta_init)).astype(np_float_type)
            self.log_theta = tf.Variable(log_theta, name="log_theta")
            self.theta = tf.exp(self.log_theta, name="theta")

            self.var_list = [self.log_theta, self.log_zeta]

    def features(self, x, name='features'):

        with tf.name_scope(name):
            return x

    def kernel(self, x_p, x_q, name=None):

        with tf.name_scope(name):
            return self.out_kernel(self.features(x_p), self.features(x_q), self.theta)

    def fit(self, x_train, y_train,
            max_iter=1000,
            n_sgd_batch=None,
            sequential_batch=False,
            save_step=10,
            log_all=False):
        if not self.setup:
            self._setup_graph(x_train, y_train)

        # Run the optimisation
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.step = 0
        np.set_printoptions(precision=2)
        self.batch_train_feed_dict = {self.x_train: x_train, self.y_train: y_train}
        print('Starting Training')
        print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')

        epoch = 0
        perm_indices = np.random.permutation(np.arange(self.n))
        print('Epoch: %d' % epoch)

        while self.step < max_iter:

            # Sample batch data
            if n_sgd_batch:
                if sequential_batch:
                    if int((self.step * n_sgd_batch) / self.n) > epoch:
                        epoch += 1
                        perm_indices = np.random.permutation(np.arange(self.n))
                        print('Epoch: %d' % epoch)
                    sgd_indices = np.arange(self.step * n_sgd_batch, (self.step + 1) * n_sgd_batch) % self.n
                    x_batch = x_train[perm_indices][sgd_indices]
                    y_batch = y_train[perm_indices][sgd_indices]
                else:
                    sgd_indices = np.random.choice(self.n, n_sgd_batch, replace=False)
                    x_batch = x_train[sgd_indices]
                    y_batch = y_train[sgd_indices]
                self.batch_train_feed_dict = {self.x_train: x_batch, self.y_train: y_batch}
            else:
                x_batch = x_train
                y_batch = y_train

            # Save status
            if self.step % save_step == 0:
                if log_all:
                    self.save_status(x_train, y_train)
                else:
                    self.save_status(x_batch, y_batch)

            # Run a training step
            self.sess.run(self.train_step, feed_dict=self.batch_train_feed_dict)
            self.step += 1

        return self

    def log_test_data(self, x_test, y_test):

        self.x_test_data = x_test
        self.y_test_data = y_test
        self.has_test_data = True

    def log_directory(self, directory):

        self.directory = directory

    def save_status(self, x_batch, y_batch):

        theta = self.sess.run(self.theta)
        zeta = self.sess.run(self.zeta)

        batch_feed_dict = {self.x_train: x_batch, self.y_train: y_batch}

        train_acc = self.sess.run(self.accuracy, feed_dict=batch_feed_dict)
        train_cel = self.sess.run(self.cross_entropy_loss, feed_dict=batch_feed_dict)
        train_cel_valid = self.sess.run(self.cross_entropy_loss_valid, feed_dict=batch_feed_dict)
        train_msp = self.sess.run(self.msp, feed_dict=batch_feed_dict)
        complexity = self.sess.run(self.complexity, feed_dict=batch_feed_dict)

        grad = self.sess.run(self.grad, feed_dict=batch_feed_dict)
        grad_norms = compute_grad_norms(grad)

        result = {'step': self.step,
                  'theta': theta,
                  'zeta': zeta,
                  'train_acc': train_acc,
                  'train_cel': train_cel,
                  'train_cel_valid': train_cel_valid,
                  'train_msp': train_msp,
                  'complexity': complexity,
                  'grad_norms': grad_norms}

        print('Step %d (n=%d)' % (self.step, x_batch.shape[0]),
              '|REG:', zeta[0],
              '|THETA: ', theta,
              '|BC:', complexity,
              '|BACC:', train_acc,
              '|BCEL:', train_cel,
              '|BCELV:', train_cel_valid,
              '|BMSP:', train_msp,
              '|Batch Gradient Norms:', grad_norms)

        if self.has_test_data:

            batch_test_feed_dict = {self.x_train: x_batch, self.y_train: y_batch, self.x_query: self.x_test_data, self.y_query: self.y_test_data}
            test_acc = self.sess.run(self.query_accuracy, feed_dict=batch_test_feed_dict)
            test_cel = self.sess.run(self.query_cross_entropy_loss, feed_dict=batch_test_feed_dict)
            test_cel_valid = self.sess.run(self.query_cross_entropy_loss_valid, feed_dict=batch_test_feed_dict)
            test_msp = self.sess.run(self.query_msp, feed_dict=batch_test_feed_dict)

            result.update({'test_acc': test_acc,
                           'test_cel': test_cel,
                           'test_cel_valid': test_cel_valid,
                           'test_msp': test_msp})

            print('Step %d (n=%d)' % (self.step, x_batch.shape[0]),
                  '|BTACC:', test_acc,
                  '|BTCEL:', test_cel,
                  '|BTCELV:', test_cel_valid,
                  '|BTMSP:', test_msp)

        if self.directory is not None:
            np.savez('%sresults_%d.npz' % (self.directory, self.step), **result)
            self.save_step = self.step

    def _setup_graph(self, x_train, y_train):

        with tf.name_scope('metadata'):

            classes = np.unique(y_train)
            class_indices = np.arange(classes.shape[0])
            self.classes = tf.cast(tf.constant(classes), tf_float_type,
                                   name='classes')
            self.class_indices = tf.cast(tf.constant(class_indices),
                                         tf_int_type, name='class_indices')
            self.n_classes = classes.shape[0]
            self.n = x_train.shape[0]
            self.d = x_train.shape[1]

        with tf.name_scope('core_graph'):

            self._setup_core_graph()

        with tf.name_scope('query_graph'):

            self._setup_query_graph()

        with tf.name_scope('optimisation'):

            if self.learning_objective == 'er+rcb':
                self.lagrangian = self.objective
            elif self.learning_objective == 'er':
                self.lagrangian = self.cross_entropy_loss
            elif self.learning_objective == 'rcb':
                self.lagrangian = self.complexity
            else:
                raise ValueError(
                    'No learning objective named "%s"' % self.learning_objective)
            self.grad = tf.gradients(self.lagrangian, self.var_list)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_step = opt.minimize(self.lagrangian, var_list=self.var_list)

        self.setup = True

    def _setup_core_graph(self):

        with tf.name_scope('train_data'):

            self.x_train = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_train')
            self.y_train = tf.placeholder(tf_float_type, shape=[None, 1], name='y_train')
            self.n_train = tf.shape(self.x_train)[0]
            self.y_train_one_hot = tf.equal(self.y_train, self.classes, name='y_train_one_hot')
            self.y_train_indices = _decode_one_hot(self.y_train_one_hot, name='y_train_indices')

        with tf.name_scope('train_features'):

            self.z_train = self.features(self.x_train, name='z_train')

        with tf.name_scope('regularisation_matrix'):

            i = tf.cast(tf.eye(self.n_train), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('gram_matrix'):

            self.k = self.kernel(self.x_train, self.x_train, name='k')
            self.k_reg = tf.add(self.k, reg, name='k_reg')
            self.chol_k_reg = tf.cholesky(self.k_reg, name='chol_k_reg')

        with tf.name_scope('core_decision_probabilities'):

            y = tf.cast(self.y_train_one_hot, tf_float_type)
            self.v = tf.cholesky_solve(self.chol_k_reg, y, name='v')
            self.p = tf.matmul(self.k, self.v, name='p')
            # Extra
            self.p_valid = tf.transpose(_clip_normalize(tf.transpose(self.p)), name='p_valid')

        with tf.name_scope('core_cross_entropy_loss'):

            self.p_y = tf_label_prob(y, self.p, name='p_y')
            self.cross_entropy_loss = tf_info(self.p_y, name='cross_entropy_loss')
            # Extra
            self.p_y_valid = tf_label_prob(y, self.p_valid, name='p_y_valid')
            self.cross_entropy_loss_valid = tf_info(self.p_y_valid, name='cross_entropy_loss_valid')

        with tf.name_scope('complexity'):

            vkv = tf.matmul(tf.transpose(self.v), self.p)
            self.complexity = tf.multiply(self.theta[0], tf.sqrt(tf.trace(vkv)), name='complexity')

        with tf.name_scope('objective'):

            const = tf.cast(tf.constant(4. * np.exp(1.)), tf_float_type)
            self.objective = tf.add(self.cross_entropy_loss, tf.multiply(const, self.complexity), name='objective')

        with tf.name_scope('core_predictions'):

            self.y_pred = _classify(self.p, classes=self.classes, name='y_pred')

        with tf.name_scope('core_accuracy'):

            self.accuracy = tf_accuracy(self.y_train, self.y_pred, name='accuracy')

        with tf.name_scope('core_other'):

            self.msp = tf.reduce_mean(tf.reduce_sum(self.p, axis=1), name='msp')

    def _setup_query_graph(self):

        with tf.name_scope('query_input'):

            self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_query')
            self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')
            self.n_query = tf.shape(self.x_query)[0]
            self.y_query_one_hot = tf.equal(self.y_query, self.classes, name='y_query_one_hot')
            self.y_query_indices = _decode_one_hot(self.y_query_one_hot, name='y_query_indices')

        with tf.name_scope('query_features'):

            self.z_query = self.features(self.x_query, name='z_query')

        with tf.name_scope('query_gram_matrix'):

            self.k_query = self.kernel(self.x_train, self.x_query, name='k_query')

        with tf.name_scope('query_decision_probabilities'):

            self.query_p = tf.matmul(tf.transpose(self.k_query), self.v, name='query_p')
            self.query_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.query_p)), name='query_p_valid')

        with tf.name_scope('query_predictions'):

            self.query_y = _classify(self.query_p, classes=self.classes, name='query_y')

        with tf.name_scope('query_accuracy'):

            self.query_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.query_y, tf.reshape(self.y_query, [-1])), tf_float_type), name='query_accuracy')

        with tf.name_scope('query_cross_entropy_loss'):

            y = tf.cast(self.y_query_one_hot, tf_float_type)
            self.query_p_y = tf_label_prob(y, self.query_p, name='query_p_y')
            self.query_cross_entropy_loss = tf_info(self.query_p_y, name='query_cross_entropy_loss')
            self.query_p_y_valid = tf_label_prob(y, self.query_p_valid, name='query_p_y_valid')
            self.query_cross_entropy_loss_valid = tf_info(self.query_p_y_valid, name='query_cross_entropy_loss_valid')

        with tf.name_scope('query_other'):

            self.query_msp = tf.reduce_mean(tf.reduce_sum(self.query_p, axis=1), name='query_msp')


def compute_grad_norms(grad):

    return np.array([np.max(np.abs(grad_i)) for grad_i in grad])


def tf_label_prob(y, p, name=None):

    with tf.name_scope(name):
        return tf.reduce_sum(tf.multiply(y, p), axis=1)


def tf_info(p, eps=1e-15, name=None):

    with tf.name_scope(name):
        return tf.reduce_sum(- tf.log(tf.clip_by_value(p, eps, 1)))


def tf_accuracy(y_true, y_pred, name=None):

    with tf.name_scope(name):
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.reshape(y_true, [-1])), tf_float_type))
