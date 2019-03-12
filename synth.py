import time
from os.path import join
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from config import ConfigLoader
from model import SimpleRNN
from dataset import SpeechDataset
from trainer import Trainer
from generator import ParameterGenerator, WaveGenerator
from util import get_max_length, get_var


# ---------- Directory Setting ----------

work_dir = './'
data_dir = join(work_dir, 'data')
conf_dir = join(work_dir, 'config')
out_dir = join(work_dir, 'out')

question_dir = join(data_dir, 'question')
synth_label_dir = join(data_dir, 'test_labels')


# ---------- Global Config ----------

types = ['duration', 'acoustic']
device = 'cuda' if torch.cuda.is_available else 'cpu'
config = ConfigLoader(join(conf_dir, 'trn.cnf'))
question_file = join(question_dir, 'linguistic.hed')


# ---------- Data Load ----------

x_paths = {}
t_paths = {}
dataset = {}
loader = {}

for type_ in types:
    print('Loading %s dataset ... ' % (type_), end='')
    x_paths[type_] = glob(join(data_dir, 'x_' + type_, '*.bin'))
    t_paths[type_] = glob(join(data_dir, 't_' + type_, '*.bin'))

    x_dim = config.get_feature_config().get_linguistic_dim(type_)
    t_dim = config.get_feature_config().get_parm_dim(type_)
    max_len = get_max_length(x_paths[type_], x_dim)
    
    dataset[type_] = SpeechDataset(x_paths[type_], t_paths[type_],
                                   x_dim=x_dim, t_dim=t_dim, max_len=max_len)
    batch_size = config.get_train_config().batch_size
    loader[type_] = DataLoader(dataset[type_], batch_size=batch_size,
                               shuffle=True)
    print('done!')
    print('\tDataset Size\t%d' % (len(x_paths[type_])))
    print('\tInput Dim\t%d' % (x_dim))
    print('\tTarget Dim\t%d\n' % (t_dim))


# ---------- Make Model ----------

print('Creating models ...')
model = {}

for type_ in types:
    conf = config.get_network_config()

    model[type_] = SimpleRNN(conf.input_dim[type_], conf.hidden_dim[type_],
                             conf.output_dim[type_], conf.num_layers[type_],
                             conf.bidirectional[type_]).to(device)
    print('%s model:' % (type_))
    print(model[type_], '\n')


# ---------- Optimizer and Criterion ----------

learning_rate = config.get_train_config().learning_rate
criterion = torch.nn.MSELoss()
optimizer = {}

for type_ in types:
    optimizer[type_] = Adam(model[type_].parameters(), lr=learning_rate)
    print('%s optimizer:' % (type_))
    print(optimizer[type_], '\n')


# ---------- Model Training ----------

for type_ in types:
    print('--- Training for %s model ---' % (type_))
    trainer = Trainer(model[type_], optimizer[type_],
                      config.get_train_config(), device=device)
    trainer.train(dataset[type_], criterion)
    print()


# ---------- Synthesize ----------

print('--- Synthesize ---')

synth_labels = glob(join(synth_label_dir, '*.lab'))
parm_var = get_var(dataset['acoustic'])

parameter_generator = ParameterGenerator(model['duration'],
                                         model['acoustic'],
                                         question_file,
                                         config.get_feature_config(),
                                         device=device)
wave_generator = WaveGenerator(synth_labels, out_dir,
                               parameter_generator,
                               config.get_feature_config(),
                               config.get_analysis_config())
wave_generator.generate(parm_var)
