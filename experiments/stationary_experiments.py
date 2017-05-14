import numpy as np
import os, sys, datetime
from cake import StationaryKernelEmbeddingClassifier


def run_experiment(x_train, y_train, x_test, y_test,
                   name='stationary_experiment',
                   learning_objective='er+rcb',
                   learning_rate=0.1,
                   theta_init=np.ones(2),
                   zeta_init=1.0,
                   max_iter=1000,
                   n_sgd_batch=None,
                   sequential_batch=True,
                   save_step=10,
                   log_all=False):

    now = datetime.datetime.now()
    now_string = '_%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                         now.hour, now.minute, now.second)
    name += now_string
    print('\n--------------------------------------------------------------\n')
    full_directory = './%s/' % name
    os.mkdir(full_directory)
    print('Running Experiment: %s' % name)
    print('Results will be saved in "%s"' % full_directory)

    classes = np.unique(y_train)
    n_class = classes.shape[0]
    print('There are %d classes for this dataset: ' % n_class, classes)
    for c in classes:
        print('There are %d observations for class %d.' % (np.sum(y_train == c), c))
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    n_dim = x_train.shape[1]
    print('Training on %d examples' % n_train)
    print('Testing on %d examples' % n_test)
    print('This dataset has %d dimensions' % n_dim)

    kec = StationaryKernelEmbeddingClassifier(learning_objective=learning_objective, learning_rate=learning_rate)
    kec.log_test_data(x_test, y_test)
    kec.log_directory(full_directory)
    kec.initialise_parameters(theta_init, zeta_init)
    try:
        kec.fit(x_train, y_train,
                max_iter=max_iter,
                n_sgd_batch=n_sgd_batch,
                sequential_batch=sequential_batch,
                save_step=save_step,
                log_all=log_all)
    except:
        pass
    kec.save_status(x_train, y_train)
    kec.sess.close()
    total_steps = kec.save_step

    config = {'x_train': x_train,
              'y_train': y_train,
              'x_test': x_test,
              'y_test': y_test,
              'n_train': n_train,
              'n_test': n_test,
              'n_dim': n_dim,
              'n_class': n_class,
              'classes': classes,
              'name': name,
              'now_string': now_string,
              'learning_objective': learning_objective,
              'learning_rate': learning_rate,
              'theta_init': theta_init,
              'zeta_init': zeta_init,
              'max_iter': max_iter,
              'n_sgd_batch': n_sgd_batch,
              'sequential_batch': sequential_batch,
              'save_step': save_step,
              'log_all': log_all,
              'full_directory': full_directory}

    config_keys = ['n_train', 'n_test', 'n_dim', 'n_class', 'classes', 'name', 'now_string', 'learning_objective', 'learning_rate', 'theta_init', 'zeta_init', 'max_iter', 'n_sgd_batch', 'sequential_batch', 'save_step', 'log_all', 'full_directory']

    np.savez('%sconfig.npz' % full_directory, **config)

    f = open('%sresults.txt' % full_directory, 'w')
    for key in config_keys:
        quantity = config[key]
        if isinstance(quantity, np.ndarray):
            f.write('%s: %s\n' % (key, np.array_str(quantity, precision=8)))
        elif isinstance(quantity, bool):
            f.write('%s: %s\n' % (key, str(quantity)))
        elif isinstance(quantity, int):
            f.write('%s: %d\n' % (key, quantity))
        else:
            try:
                f.write('%s: %f\n' % (key, quantity))
            except:
                f.write('%s: %s\n' % (key, str(quantity)))
    f.write('----------------------------------------\n')
    f.write('Final Results:\n')
    result_keys = ['step', 'theta', 'zeta', 'complexity', 'grad_norms',
                   'train_acc', 'train_cel', 'train_cel_valid', 'train_msp',
                   'test_acc', 'test_cel', 'test_cel_valid', 'test_msp']

    result = np.load('%sresults_%d.npz' % (full_directory, total_steps))
    f.write('Iterations Completed: %d\n' % total_steps)
    for key in result_keys:
        quantity = result[key]
        if isinstance(quantity, np.ndarray):
            f.write('%s: %s\n' % (key, np.array_str(quantity, precision=8).replace('\n', '')))
        elif isinstance(quantity, bool):
            f.write('%s: %s\n' % (key, str(quantity)))
        elif isinstance(quantity, int):
            f.write('%s: %d\n' % (key, quantity))
        else:
            try:
                f.write('%s: %f\n' % (key, quantity))
            except:
                f.write('%s: %s\n' % (key, str(quantity)))
    f.close()

    return full_directory, total_steps


def run_cross_val_experiment(x, y, k=10, seed=0,
                             name='stationary_experiment',
                             learning_objective='er+rcb',
                             learning_rate=0.1,
                             theta_init=np.ones(2),
                             zeta_init=1.0,
                             max_iter=1000,
                             n_sgd_batch=None,
                             sequential_batch=True,
                             save_step=10,
                             log_all=False):

    np.random.seed(seed)
    n = x.shape[0]
    perm_indices = np.random.permutation(np.arange(n))
    x_perm = x[perm_indices]
    y_perm = y[perm_indices]

    results = []
    all_steps = np.zeros(k)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    for i, (train_index, test_index) in enumerate(kf.split(x)):

        x_train = x_perm[train_index]
        y_train = y_perm[train_index]
        x_test = x_perm[test_index]
        y_test = y_perm[test_index]

        fold_name = name + ('_fold_%d' % i)

        fold_directory, total_steps = run_experiment(x_train, y_train, x_test, y_test,
                                                     name=fold_name,
                                                     learning_objective=learning_objective,
                                                     learning_rate=learning_rate,
                                                     theta_init=theta_init,
                                                     zeta_init=zeta_init,
                                                     max_iter=max_iter,
                                                     n_sgd_batch=n_sgd_batch,
                                                     sequential_batch=sequential_batch,
                                                     save_step=save_step,
                                                     log_all=log_all)

        result = np.load('%sresults_%d.npz' % (fold_directory, total_steps))
        results.append(result)
        all_steps[i] = total_steps

    ### SAVE RESULT INTO A FILE
    result_keys = ['complexity',
                   'train_acc', 'train_cel', 'train_cel_valid', 'train_msp',
                   'test_acc', 'test_cel', 'test_cel_valid', 'test_msp']

    result_all_values = np.zeros((k, len(result_keys)))

    for i, result in enumerate(results):
        for j, key in enumerate(result_keys):
            result_all_values[i, j] = result[key]

    now = datetime.datetime.now()
    now_string = '_%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                         now.hour, now.minute, now.second)

    full_directory = './%s_average_%s/' % (name, now_string)
    os.mkdir(full_directory)
    np.savez('%saverage_results.npz' % full_directory,
             result_keys=np.array(result_keys),
             result_all_values=result_all_values)

    ### DISPLAY AS A TEXT FILE
    f = open('%sresults.txt' % full_directory, 'w')
    mean_values = np.mean(result_all_values, axis=0)
    std_values = np.std(result_all_values, axis=0)
    for j, key in enumerate(result_keys):
        sample_values = result_all_values[:, j]
        f.write('%s: %s\n' % (key, np.array_str(sample_values, precision=8).replace('\n', '')))
    f.write('\nSteps Taken: %s\n' % np.array_str(all_steps, precision=0).replace('\n', ''))

    f.write('\n\n-----\n\n')
    f.write('Average:\n')
    for j, key in enumerate(result_keys):
        f.write('%s: %.8f\n' % (key, mean_values[j]))

    f.write('\n\n-----\n\n')
    f.write('Standard Deviation:\n')
    for j, key in enumerate(result_keys):
        f.write('%s: %.8f\n' % (key, std_values[j]))

    f.close()


def load_train_test_data(name, normalize_features=True):
    """
    Load the dataset.

    Parameters
    ----------
    name : str
        The name of the dataset
    normalize_features : bool, optional
        To normalize the features or not

    Returns
    -------
    numpy.ndarray
        The training features (n_train, d)
    numpy.ndarray
        The training target labels (n_train, 1)
    numpy.ndarray
        The testing features (n_test, d)
    numpy.ndarray
        The testing target labels (n_test, 1)
    numpy.ndarray
        The class names (?,)
    """
    train_data = np.load('%s_train.npz' % name)
    x_train = train_data['x']
    y_train = train_data['y'][:, np.newaxis]
    class_names = train_data['class_names']
    test_data = np.load('%s_test.npz' % name)
    x_test = test_data['x']
    y_test = test_data['y'][:, np.newaxis]
    if normalize_features:
        x = np.concatenate((x_train, x_test), axis=0)
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_train = (x_train - x_min) / (x_max - x_min)
        x_test = (x_test - x_min) / (x_max - x_min)
    return x_train, y_train, x_test, y_test, class_names


def load_all_data(name, normalize_features=True):
    """
    Load the dataset.

    Parameters
    ----------
    name : str
        The name of the dataset
    normalize_features : bool, optional
        To normalize the features or not

    Returns
    -------
    numpy.ndarray
        The features (n, d)
    numpy.ndarray
        The target labels (n, 1)
    numpy.ndarray
        The class names (?,)
    """
    data = np.load('%s.npz' % name)
    x = data['x']
    if normalize_features:
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
    y = data['y'][:, np.newaxis]
    class_names = data['class_names']
    return x, y, class_names


def parse(key, currentvalue, arg=1):

    cast = type(currentvalue)
    if key in sys.argv:
        if cast == bool:
            return not currentvalue
        currentvalue = sys.argv[sys.argv.index(key) + arg]
    return cast(currentvalue)