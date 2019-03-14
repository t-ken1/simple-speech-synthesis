import numpy as np
from torch.utils.data import Dataset
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale


class SpeechDataset(Dataset):

    def __init__(self, x_paths, t_paths, x_dim=262, t_dim=127,
                 max_len=5000, pad_value=9999):
        super().__init__()

        self.xs = self._fromfile(x_paths, x_dim)
        self.ts = self._fromfile(t_paths, t_dim)

        self.x_dim = x_dim
        self.t_dim = t_dim

        self.x_stat, self.t_stat = {}, {}
        self.x_stat['min'], self.x_stat['max'] = self._get_x_stat()
        self.t_stat['mean'], self.t_stat['var'] = self._get_t_stat()

        self.pad_value = pad_value
        self.max_len = max_len

    def _fromfile(self, paths, dim):
        result = []
        for path in paths:
            result.append(
                np.fromfile(path, dtype=np.float32).reshape(-1, dim)
            )
        return result

    def _get_x_stat(self):
        return minmax(self.xs, self._get_lengths(self.xs))

    def _get_t_stat(self):
        return meanvar(self.ts, self._get_lengths(self.ts))

    def _get_lengths(self, sequences):
        return np.array([len(x) for x in sequences], dtype=np.int)

    def _padding(self, seq):
        return np.pad(seq, [(0, self.max_len-len(seq)), (0, 0)],
                      'constant', constant_values=self.pad_value)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x, t = self.xs[idx], self.ts[idx]
        x = minmax_scale(x, self.x_stat['min'], self.x_stat['max'],
                         feature_range(0.01, 0.99))
        t = scale(t, self.t_stat['mean'], np.sqrt(self.t_stat['var']))

        pad_x = self._padding(x)
        pad_t = self._padding(t)

        return pad_x, pad_t, len(self.xs[idx]), len(self.ts[idx])
