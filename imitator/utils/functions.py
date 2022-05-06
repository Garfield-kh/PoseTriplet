import torch
import numpy as np
import torch.nn as nn

def mask_features(features, masked_freq):
    """
    mask feature with give frequency
    :param features: t x feature size
    :return:
    """
    mask = np.zeros_like(features)
    mask[::masked_freq] = 1.
    feature_out =  features * mask
    return feature_out


def mask_features_withshuffle(features, masked_freq):
    """
    mask feature with give frequency
    :param features: t x feature size
    :return:
    """
    # mask feature
    mask = np.zeros_like(features)
    mask[::masked_freq] = 1.
    feature_out =  features * mask
    # shuffle feature
    b = feature_out[::masked_freq] * 1.
    np.random.shuffle(b)
    feature_out[::masked_freq, 2:] = b[:, 2:] * 1
    return feature_out