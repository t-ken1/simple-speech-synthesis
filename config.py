import configparser
from abc import ABCMeta, abstractmethod

import numpy as np
import pysptk
import pyworld


class BaseConfig(metaclass=ABCMeta):

    def __init__(self, config_parser):
        self.config_parser = config_parser

    def _to(self, str_value, function):
        if not isinstance(str_value, str):
            raise TypeError("1st argument requires 'str'.")
        if not callable(function):
            raise TypeError("2nd argument requires 'callabel'.")
        return function(str_value)

    def to_int(self, arg):
        return self._to(arg, int)

    def to_float(self, arg):
        return self._to(arg, float)

    def to_bool(self, arg):
        if not isinstance(arg, str):
            raise TypeError("arg requires 'str'.")
        return (arg == 'True')

    def get_value(self, arg):
        return self.config_parser.get(self._get_type(), arg)

    @abstractmethod
    def _get_type(self):
        pass


class FeatureConfig(BaseConfig):

    def __init__(self, config_parser):
        super().__init__(config_parser)
        self._field_init()

        self.feat_dim['mgc'] = self.to_int(self.get_value('mgc_dim'))
        self.feat_dim['lf0'] = self.to_int(self.get_value('lf0_dim'))
        self.feat_dim['bap'] = self.to_int(self.get_value('bap_dim'))
        self.feat_dim['vuv'] = self.to_int(self.get_value('vuv_dim'))

        self.linguistic_dim['acoustic'] = self.to_int(
            self.get_value('acoustic_linguistic_dim')
        )
        self.linguistic_dim['duration'] = self.to_int(
            self.get_value('duration_linguistic_dim')
        )

        self.parm_dim['acoustic'] = self._get_acoustic_dim()
        self.parm_dim['duration'] = 1
        self.subphone_feature = self.get_value('subphone_feat')

        self.feat_index['mgc'] = 0
        self.feat_index['lf0'] = self.feat_index['mgc'] + self.feat_dim['mgc']
        self.feat_index['vuv'] = self.feat_index['lf0'] + self.feat_dim['lf0']
        self.feat_index['bap'] = self.feat_index['vuv'] + self.feat_dim['vuv']

    def _get_type(self):
        return 'feat'
        
    def _field_init(self):
        self.feat_dim = {}
        self.feat_index = {}        
        self.parm_dim = {}
        self.linguistic_dim = {}

    def _get_acoustic_dim(self):
        result = 0
        for feat in ['mgc', 'lf0', 'bap', 'vuv']:
            result += self.feat_dim[feat]
        return result

    def get_parm_dim(self, arg):
        return self.parm_dim[arg]

    def get_linguistic_dim(self, arg):
        return self.linguistic_dim[arg]

    def get_indices(self):
        return self.feat_index


class AnalysisConfig(BaseConfig):

    def __init__(self, config_parser):
        super().__init__(config_parser)

        self.sampling_rate = self.to_int(self.get_value('sampling_rate'))
        self.frame_period = self.to_int(self.get_value('frame_period'))
        self.has_delta = self.to_bool(self.get_value('has_delta'))

        if self.has_delta:
            self.window = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
                (1, 1, np.array([1.0, -2.0, 1.0]))
            ]
        else:
            self.window = [(0, 0, np.array([1.0]))]

        self.fft_length = pyworld.get_cheaptrick_fft_size(
            self.sampling_rate
        )
        self.alpha = pysptk.util.mcepalpha(self.sampling_rate)
        self.hop_length = int(
            self.sampling_rate * 0.001 * self.frame_period
        ) # require [Hz] -> [kHz]

    def _get_type(self):
        return 'analysis'


class TrainConfig(BaseConfig):

    def __init__(self, config_parser, kind):
        super().__init__(config_parser)
        
        self.num_epoch = self.to_int(self.get_value('num_epoch'))
        self.batch_size = self.to_int(self.get_value('batch_size'))
        self.learning_rate = self.to_float(self.get_value('learning_rate'))

    def _get_type(self):
        return 'train'


class NetworkConfig(BaseConfig):

    def __init__(self, config_parser):
        super().__init__(config_parser)
        self._field_init()

        network_types = ['acoustic', 'duration']
        feat_conf = FeatureConfig(config_parser)
        for type_ in network_types:
            self.input_dim[type_] = feat_conf.get_linguistic_dim(type_)
            self.output_dim[type_] = feat_conf.get_parm_dim(type_)
            self.hidden_dim[type_] = self.to_int(self.get_value('hidden_dim'))
            self.num_layers[type_] = self.to_int(self.get_value('num_layers'))
            self.bidirectional[type_] = self.to_bool(
                self.get_value('bidirectional')
            )

    def _get_type(self):
        return 'network'

    def _field_init(self):
        self.input_dim = {}
        self.output_dim = {}
        self.hidden_dim = {}
        self.num_layers = {}
        self.bidirectional = {}


class ConfigLoader(object):

    def __init__(self, path):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(path)

        self.analysis_config = AnalysisConfig(self.config_parser)
        self.feature_config = FeatureConfig(self.config_parser)
        self.network_config = NetworkConfig(self.config_parser)
        self.train_config = TrainConfig(self.config_parser)

    def get_analysis_config(self):
        return self.analysis_config

    def get_feature_config(self):
        return self.feature_config

    def get_netowrk_config(self):
        return self.network_config

    def get_train_config(self):
        return self.train_config
