"""
Define the stationary kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from .infer import clip_normalize as _clip_normalize
from .infer import classify as _classify
from .infer import decode_one_hot as _decode_one_hot
from .data_type_def import *


class NeuralEmbeddingClassifier():

    def __init__(self, n_dim=2, n_class=3, learning_objective='er+rcb', learning_rate=0.1, hidden_units=[20, 10]):

        self.n_dim = n_dim
        self.n_class = n_class
        self.learning_objective = learning_objective
        self.learning_rate = learning_rate
        self.n_layer = len(hidden_units)
        self.hidden_units = np.array(hidden_units)
        self.setup = False
        self.has_test_data = False
        self.directory = None

    def initialise_parameters(self, zeta=0.1, weights_std=0.1, biases=0.1, seed=0):

        with tf.name_scope('parameters'):

            self.zeta_init = zeta
            log_zeta = np.log(np.atleast_1d(self.zeta_init)).astype(np_float_type)
            self.log_zeta = tf.Variable(log_zeta, name="log_zeta")
            self.zeta = tf.exp(self.log_zeta, name="zeta")

            self.weight_list = []
            self.bias_list = []

            weights_std = weights_std * np.ones(self.n_layer)
            biases = biases * np.ones(self.n_layer)

            tf.set_random_seed(seed)
            old_hidden_dim = self.n_dim
            for l in range(self.n_layer):
                new_hidden_dim = self.hidden_units[l]
                self.weight_list.append(weight_variable([old_hidden_dim, new_hidden_dim], std=weights_std[l], name='weight_%d' % l))
                self.bias_list.append(bias_variable([new_hidden_dim], bias=biases[l], name='bias_%d' % l))
                old_hidden_dim = new_hidden_dim

            alpha = tf.sqrt(tf.cast(self.n_dim, tf_float_type))
            for l in range(self.n_layer):
                weight_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_list[l])))
                bias_norm = tf.sqrt(tf.reduce_sum(tf.square(self.bias_list[l])))
                alpha = tf.multiply(weight_norm, alpha) + bias_norm
            self.alpha = alpha

            self.var_list = [self.log_zeta] + self.weight_list + self.bias_list

        if not self.setup:
            self._setup_graph(self.n_dim, self.n_class)

    def features(self, x, name='features'):

        with tf.name_scope(name):

            h = x
            for l in range(self.n_layer):
                h = perceptron(h, self.weight_list[l], self.bias_list[l])
            return h

    def kernel(self, x_p, x_q, name=None):

        with tf.name_scope(name):
            return tf.matmul(self.features(x_p), tf.transpose(self.features(x_q)))

    def fit(self, x_train, y_train,
            max_iter=1000,
            n_sgd_batch=None,
            sequential_batch=False,
            save_step=10,
            log_all=False):
        # Run the optimisation
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.step = 0
        np.set_printoptions(precision=2)
        self.batch_train_feed_dict = {self.x_train: x_train, self.y_train: y_train}
        print('Starting Training')
        print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')

        n = x_train.shape[0]
        epoch = 0
        perm_indices = np.random.permutation(np.arange(n))
        print('Epoch: %d' % epoch)

        while self.step < max_iter:

            # Sample batch data
            if n_sgd_batch:
                if sequential_batch:
                    if int((self.step * n_sgd_batch) / n) > epoch:
                        epoch += 1
                        perm_indices = np.random.permutation(np.arange(n))
                        print('Epoch: %d' % epoch)
                    sgd_indices = np.arange(self.step * n_sgd_batch, (self.step + 1) * n_sgd_batch) % n
                    x_batch = x_train[perm_indices][sgd_indices]
                    y_batch = y_train[perm_indices][sgd_indices]
                else:
                    sgd_indices = np.random.choice(n, n_sgd_batch, replace=False)
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

        zeta = self.sess.run(self.zeta)

        weights = [self.sess.run(w) for w in self.weight_list]
        biases = [self.sess.run(b) for b in self.bias_list]

        batch_feed_dict = {self.x_train: x_batch, self.y_train: y_batch}

        train_acc = self.sess.run(self.accuracy, feed_dict=batch_feed_dict)
        train_cel = self.sess.run(self.cross_entropy_loss, feed_dict=batch_feed_dict)
        train_cel_valid = self.sess.run(self.cross_entropy_loss_valid, feed_dict=batch_feed_dict)
        train_msp = self.sess.run(self.msp, feed_dict=batch_feed_dict)
        complexity = self.sess.run(self.complexity, feed_dict=batch_feed_dict)

        grad = self.sess.run(self.grad, feed_dict=batch_feed_dict)
        grad_norms = compute_grad_norms(grad)

        result = {'step': self.step,
                  'zeta': zeta,
                  'weights': weights,
                  'bias': biases,
                  'train_acc': train_acc,
                  'train_cel': train_cel,
                  'train_cel_valid': train_cel_valid,
                  'train_msp': train_msp,
                  'complexity': complexity,
                  'grad_norms': grad_norms}

        print('Step %d (n=%d)' % (self.step, x_batch.shape[0]),
              '|REG:', zeta[0],
              '|W_NORM:', np.max([np.max(w) for w in weights]),
              '|B_NORM:', np.max([np.max(b) for b in biases]),
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

    def _setup_graph(self, n_dim, n_class):

        with tf.name_scope('metadata'):

            self.n_class = n_class
            self.n_dim = n_dim
            self.classes = tf.cast(tf.constant(np.arange(n_class)), tf_float_type, name='classes')

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

            self.x_train = tf.placeholder(tf_float_type, shape=[None, self.n_dim], name='x_train')
            self.y_train = tf.placeholder(tf_float_type, shape=[None, 1], name='y_train')
            self.n_train = tf.shape(self.x_train)[0]
            self.y_train_one_hot = tf.equal(self.y_train, self.classes, name='y_train_one_hot')
            self.y_train_indices = _decode_one_hot(self.y_train_one_hot, name='y_train_indices')

        with tf.name_scope('train_features'):

            self.z_train = self.features(self.x_train, name='z_train')

        with tf.name_scope('regularisation_matrix'):

            i = tf.cast(tf.eye(tf.shape(self.z_train)[1]), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('scatter_matrix'):

            zt = tf.transpose(self.z_train)
            z = self.z_train
            ztz_reg = tf.add(tf.matmul(zt, z, name='scatter'), reg, name='scatter_reg')
            self.chol_ztz_reg = tf.cholesky(ztz_reg, name='chol_scatter_reg')

        with tf.name_scope('weights'):

            y = tf.cast(self.y_train_one_hot, tf_float_type)
            self.w = tf.cholesky_solve(self.chol_ztz_reg, tf.matmul(zt, y), name='weights')

        with tf.name_scope('core_decision_probabilities'):

            self.p = tf.matmul(z, self.w, name='p')
            # Extra
            self.p_valid = tf.transpose(_clip_normalize(tf.transpose(self.p)), name='p_valid')

        with tf.name_scope('core_cross_entropy_loss'):

            self.p_y = tf_label_prob(y, self.p, name='p_y')
            self.cross_entropy_loss = tf_info(self.p_y, name='cross_entropy_loss')
            # Extra
            self.p_y_valid = tf_label_prob(y, self.p_valid, name='p_y_valid')
            self.cross_entropy_loss_valid = tf_info(self.p_y_valid, name='cross_entropy_loss_valid')

        with tf.name_scope('complexity'):

            w_tr = tf.sqrt(tf.reduce_sum(tf.square(self.w)))
            self.complexity = tf.multiply(1.0, w_tr, name='complexity')

        with tf.name_scope('objective'):

            const = tf.cast(tf.constant(4. * np.exp(1.)), tf_float_type)
            self.objective = tf.add(self.cross_entropy_loss, tf.multiply(const, self.complexity), 'objective')

        with tf.name_scope('core_predictions'):

            self.y_pred = _classify(self.p, classes=self.classes, name='y_pred')

        with tf.name_scope('core_accuracy'):

            self.accuracy = tf_accuracy(self.y_train, self.y_pred, name='accuracy')

        with tf.name_scope('core_other'):

            self.msp = tf.reduce_mean(tf.reduce_sum(self.p, axis=1), name='msp')

    def _setup_query_graph(self):

        with tf.name_scope('query_input'):

            self.x_query = tf.placeholder(tf_float_type, shape=[None, self.n_dim], name='x_query')
            self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')
            self.n_query = tf.shape(self.x_query)[0]
            self.y_query_one_hot = tf.equal(self.y_query, self.classes, name='y_query_one_hot')
            self.y_query_indices = _decode_one_hot(self.y_query_one_hot, name='y_query_indices')

        with tf.name_scope('query_features'):

            self.z_query = self.features(self.x_query, name='z_query')

        with tf.name_scope('query_decision_probabilities'):

            self.query_p = tf.matmul(self.z_query, self.w, name='query_p')
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


def perceptron(x, w, b, name='perceptron'):

    with tf.name_scope(name):
        return tf.nn.relu(tf.matmul(x, w) + b)


def weight_variable(shape, std=0.1, name='weight'):

    with tf.name_scope(name):
        initial = tf.truncated_normal(shape, stddev=std)
        return tf.Variable(tf.cast(initial, tf_float_type))


def bias_variable(shape, bias=0.1, name='bias'):

    with tf.name_scope(name):
        initial = tf.constant(bias, shape=shape)
        return tf.Variable(tf.cast(initial, tf_float_type))


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
