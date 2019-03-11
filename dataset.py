import numpy as np
from torch.utils.data import Dataset


class SpeechDataset(Dataset):

    def __init__(self, x_paths, t_paths, x_dim=262, t_dim=127,
                 max_len=5000, pad_value):
        super().__init__()

        self.x_paths = x_paths
        self.t_paths = t_paths
        self.x_dim = x_dim
        self.t_dim = t_dim
        self.pad_value = pad_value
        self.max_len = max_len

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = np.fromfile(
            self.x_paths[idx], dtype=np.float32
        ).reshape(-1, self.x_dim)
        t = np.fromfile(
            self.t_paths[idx], dtype=np.float32
        ).reshape(-1, self.t_dim)

        pad_x = self._padding(x)
        pad_t = self._padding(t)

        return pad_x, pad_t, len(x), len(t)

    def _padding(self, seq):
        return np.pad(seq, [(0, self.max_len-len(seq)), (0, 0)],
                      'constant', constant_values=self.pad_value)
