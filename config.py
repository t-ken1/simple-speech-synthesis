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
