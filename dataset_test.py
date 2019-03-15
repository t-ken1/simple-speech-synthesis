from os.path import join
from glob import glob

from dataset import SpeechDataset


work_dir = './'
data_dir = join(work_dir, 'data')

x_paths = glob(join(data_dir, 'x_acoustic', '*.bin'))
t_paths = glob(join(data_dir, 't_acoustic', '*.bin'))


dataset = SpeechDataset(x_paths, t_paths, x_dim=610, t_dim=127,
                        max_len=5000, pad_value=9999)

x, t, x_l, t_l = dataset[0]

x = x[:x_l]
t = t[:t_l]
print(t)
print(t.shape)
print(type(t))
