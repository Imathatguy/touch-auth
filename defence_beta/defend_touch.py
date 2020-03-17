#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""
import numpy as np


# Original definition from synthetic data tests
def sample_clipped(center, sd, dim, n, low=0.0, upp=1.0):
    data = np.random.normal(center, sd, (n, dim))
    # Repeat until clipped
    while ((len(data[data.min(axis=1) < low, :]) > 0) or
            (len(data[data.max(axis=1) >= upp, :]) > 0)):
        data = data[data.min(axis=1) >= low, :]
        data = data[data.max(axis=1) < upp, :]
        r = int(n - len(data))
        new = np.random.normal(center, sd, (r, dim))
        data = np.concatenate([data, new])
    if dim == 1 and n == 1:
        return data[0][0]
    elif n == 1:
        return data[0]
    else:
        return data


def generate_protection_noise(target_data, other_data, std_ratio):
    feat_mean = np.mean(target_data, axis=0)
    # feat_std = np.std(target_data, axis=0)

    alphas = abs(feat_mean - 0.5) + 0.5
    betas = np.array([0.5]*len(alphas))

    # add a pre-specified amount of vairance to the user's existing variance
    # gen_std = feat_std*(1+std_ratio)
    gen_data = np.random.beta(alphas, betas,
                              (len(target_data), len(feat_mean)))
    gen_data = np.abs((-1*np.round(feat_mean)) + gen_data)
    noise_other_data = np.concatenate([other_data, gen_data], axis=0)

    return target_data, noise_other_data


if __name__ == "__main__":
    print("No test sequence.")
