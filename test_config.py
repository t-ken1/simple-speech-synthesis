from os.path import join

from config import ConfigLoader

work_dir = '/mnt/tsukimi/ken1/work/synth-by-dnn/simple-lstm2'
config_path = join(work_dir, 'config', 'trn.cnf')

configloader = ConfigLoader(config_path)
