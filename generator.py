import os
from os.path import join, basename, splitext

import numpy as np
import torch
import pysptk
import pyworld
from nnmnkwii.io import hts
from nnmnkwii.frontend.merlin import linguistic_features
from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.postfilters import merlin_post_filter
from IPython.display import Audio


class ParameterGenerator(object):

    def __init__(self, duration_model, acoustic_model,
                 question_file, config, device='cpu'):
        self.duration_model = duration_model
        self.acoustic_model = acoustic_model
        self.bin_dict, self.con_dict = hts.load_question_set(question_file)
        self.config = config
        self.device = device

    def get_duration_label(self, path):
        label = hts.load(path)
        self.duration_model.eval()

        features = linguistic_features(label, self.bin_dict, self.con_dict,
                                       add_frame_features=False,
                                       subphone_features=None)
        features = features.astype(np.float32)
        self.duration_model.to(self.device)

        x = torch.tensor(features).float().to(self.device)
        xl = len(x)

        x = x.view(1, -1, x.size(-1))
        predicted = np.round(self.duration_model(x, [xl]).cpu().data.numpy())
        predicted[predicted <= 0] = 1
        label.set_durations(predicted)

        return label

    def get_acoustic_parameter(self, label):
        self.acoustic_model.eval()
        sil_index = label.silence_frame_indices()
        subphone_feat = self.config.subphone_feature

        input_ = linguistic_features(label, self.bin_dict, self.con_dict,
                                     add_frame_features=True,
                                     subphone_features=subphone_feat)
        input_ = np.delete(input_, sil_index, axis=0)

        x = torch.tensor(input_).float().to(self.device)
        xl = len(x)

        x = x.view(1, -1, x.size(-1))
        predicted = self.acoustic_model(x, [xl]).cpu().data.numpy()
        predicted = predicted.reshape(-1, predicted.shape[-1])

        return predicted

    def generate(self, path):
        label = self.get_duration_label(path)
        param = self.get_acoustic_parameter(label)

        return param


class WaveGenerator(object):

    def __init__(self, paths, out_dir, parameter_generator,
                 feature_config, analysis_config):
        self.paths = paths
        self.out_dir = out_dir
        self.parameter_generator = parameter_generator
        self.feature_config = feature_config
        self.analysis_config = analysis_config

    def generate(self, parm_var, do_postfilter=True):
        config = self.analysis_config

        for path in self.paths:
            file_id = splitext(basename(path))[0]
            mgc, lf0, vuv, bap = self._generate_parameters(path, parm_var)

            if do_postfilter:
                mgc = merlin_postfilter(mgc, config.alpha)

            sp = pysptk.mc2sp(mgc, fftlen=config.fft_length, alpha=config.alpha)
            ap = pyworld.decode_aperiodicity(bap.astype(np.float64),
                                             config.sampling_rate,
                                             config.fft_length)
            f0 = self._lf0_to_f0(lf0, vuv)
            generated = pyworld.syntheisze(f0.flatten().astype(np.float64),
                                           sp.astype(np.float64),
                                           ap.astype(np.float64),
                                           config.sampling_rate,
                                           config.frame_period)
            with open(join(self.out_dir, file_id + '.wav'), 'wb') as f:
                f.write(Audio(generated, rate=config.sampling_rate).data)

    def _lf0_to_f0(self, lf0, vuv, threshold=0.5):
        f0 = lf0.copy()
        fp[vuv < threshold] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

        return f0

    def _generate_parameters(self, path, var):
        seq = self.parameter_generator.generate(path)
        seq = trim_zeros_frames(seq)
        T = seq.shape[0]

        feat_index = self.feature_config.get_indices()
        mgc = seq[:, :feat_index['lf0']]
        lf0 = seq[:, feat_index['lf0']:feat_index['vuv']]
        vuv = seq[:, feat_index['vuv']]
        bap = seq[:, feat_index['bap']:]

        mgc_var = np.tile(var[:feat_index['lf0']], (T, 1))
        lf0_var = np.tile(var[feat_index['lf0']:feat_index['vuv']], (T, 1))
        bap_var = np.tile(var[feat_index['bap']:], (T, 1))

        mgc = paramgen.mlpg(mgc, mgc_var, self.analysis_config.window)
        lf0 = paramgen.mlpg(lf0, lf0_var, self.analysis_config.window)
        bap = paramgen.mlpg(bap, bap_var, self.analysis_config.window)

        return mgc, lf0, vuv, bap
