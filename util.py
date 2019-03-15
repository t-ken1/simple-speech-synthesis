import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var


def get_max_length(paths, parm_dim):
    result = 0
    for path in paths:
        x = np.fromfile(path, dtype=np.float32).reshape(-1, parm_dim)
        T = len(x)
        if T > result:
            result = T

    return result


def get_var(dataset):
    mean_, var_ = 0., 0.
    count = 0
    dtype = dataset[0][1].dtype

    for idx, (_, x, _, length) in enumerate(dataset):
        x = x[:length]
        mean_, var_, _ = _incremental_mean_and_var(x, mean_, var_, count)
        count += len(x)

    return var_.astype(dtype)
