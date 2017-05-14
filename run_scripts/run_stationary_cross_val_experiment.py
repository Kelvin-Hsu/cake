import numpy as np
from experiments.stationary_experiments \
    import run_cross_val_experiment, load_all_data, parse

name = parse('-name', 'name')

x, y, class_names = load_all_data(name)
n = x.shape[0]
d = x.shape[1]
m = len(class_names)

k = parse('-k', 10)
learning_objective = parse('-learning_objective', 'er+rcb')
learning_rate = parse('-learning_rate', 0.1)
theta_init = parse('-length_scale_init', 1.0) * np.ones(d + 1)
theta_init[0] = parse('-sensitivity_init', 1.0)
zeta_init = parse('-zeta_init', 1.0)
max_iter = parse('-max_iter', 1000)
n_sgd_batch = parse('-n_sgd_batch', 0)
n_sgd_batch = None if n_sgd_batch == 0 else n_sgd_batch
sequential_batch = parse('-sequential_batch_off', True)
save_step = parse('-save_step', 1)
log_all = parse('-log_all', False)

print('name: %s (n=%d, d=%d, m=%d)' % (name, n, d, m))
print('k: ', k)
print('learning_objective: ', learning_objective)
print('learning_rate: ', learning_rate)
print('theta_init: ', theta_init)
print('zeta_init: ', zeta_init)
print('max_iter: ', max_iter)
print('n_sgd_batch: ', n_sgd_batch)
print('sequential_batch: ', sequential_batch)
print('save_step: ', save_step)
print('log_all: ', log_all)

run_cross_val_experiment(x, y, k=k,
                         name='%s_stationary_experiment' % name,
                         learning_objective=learning_objective,
                         learning_rate=learning_rate,
                         theta_init=theta_init,
                         zeta_init=zeta_init,
                         max_iter=max_iter,
                         n_sgd_batch=n_sgd_batch,
                         sequential_batch=sequential_batch,
                         save_step=save_step,
                         log_all=log_all)
