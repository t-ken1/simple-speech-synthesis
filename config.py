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

    def get_acoustic_dim(self):
        return self.parm_dim['acoustic']

    def get_duration_dim(self):
        return self.parm_dim['duration']

    def get_acoustic_linguistic_dim(self):
        return self.linguistic_dim['acoustic']

    def get_duration_linguistic_dim(self):
        return self.linguistic_dim['duration']

    def get_indices(self):
        return self.feat_index
