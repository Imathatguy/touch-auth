#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from math import hypot, atan2
from numpy.linalg import norm

import random
import timesynth as TimeSynth
from sklearn.preprocessing import MinMaxScaler


def extract_features(ts):
    '''
    The requires ts is a pandas array that contains the following columns
    index2 - the numerica index of the datapoint within a sample
    eventTime, the ms time of the datapoint
    eventPressure
    positionX
    positionY
    '''

    # Features
    # Table 5. Features Extracted From Each Swipe Event
    # Features Description
    # 1-2 inter-stroke time, stroke duration
    # 3-6 start x, start y, stop x, stop y
    # 7-8 direct end-to-end distance, mean resultant length
    # 9 up/down/left/right flag
    # 10-12 20%, 50%, 80% -perc. pairwise velocity
    # 13-15 20%, 50%, 80%-perc. pairwise acc
    # 16 median velocity at last 3 pts
    # 17 largest deviation from end-to-end (e-e) line
    # 18-20 20%, 50%, 80%-perc. dev. from e-e line
    # 21 average direction
    # 22 ratio of end-to-end dist and trajectory length
    # 23 median acceleration at first 5 points
    # 24 mid-stroke pressure

    def udlr_flag(x_start, x_stop, y_start, y_stop):
        # We arbitarily assign values, As long as the same method is
        # applied to all samples the flag should still match
        dx = x_stop - x_start
        dy = y_stop - y_start
        if abs(dx) >= abs(dy):
            if dx < 0:
                return 0
            else:
                return 1
        else:
            if dy < 0:
                return 2
            else:
                return 3

    mod_ts = ts.sort_values('index2')[['eventTime', 'positionX',
                                      'positionY', 'eventPressure']]
    mod_ts['eventTime'] = mod_ts['eventTime'] - mod_ts['eventTime'].min()
    feats = {}
    feats['inter-stroke-t'] = mod_ts['eventTime'].iloc[int(mod_ts.shape[0]/2)]
    feats['stroke-duration'] = mod_ts['eventTime'].max()
    feats['start-x'] = mod_ts['positionX'].iloc[0]
    feats['start-y'] = mod_ts['positionY'].iloc[0]
    feats['stop-x'] = mod_ts['positionX'].iloc[-1]
    feats['stop-y'] = mod_ts['positionY'].iloc[-1]
    # End to end distance
    dist = hypot(feats['stop-x'] - feats['start-x'],
                 feats['stop-y'] - feats['start-y'])
    feats['e2e-distance'] = dist
    # Mean resultant length
    resultants = [hypot((mod_ts.iloc[n + 1]['positionX'] -
                         mod_ts.iloc[n]['positionX']),
                        (mod_ts.iloc[n + 1]['positionY'] -
                         mod_ts.iloc[n]['positionY']))
                  for n in range(mod_ts.shape[0] - 1)]

    # No deviation, not a swipe
    if sum(resultants) == 0:
        return None

    feats['mean_resultant'] = np.mean(resultants)
    # Up, down, left, right flag
    feats['udlr_flag'] = udlr_flag(feats['start-x'], feats['stop-x'],
                                   feats['start-y'], feats['stop-y'])
    # Compute Velocity and Acceleration
    time_gaps = [(mod_ts.iloc[n + 1]['eventTime'] -
                  mod_ts.iloc[n]['eventTime'])
                 for n in range(mod_ts.shape[0] - 1)]
    pairwise_v = list(np.divide(resultants, time_gaps))
    pairwise_a = list(np.divide(pairwise_v, time_gaps))
    # pairwise_vel
    feats['20percentile-vel'] = np.percentile(pairwise_v, 20)
    feats['50percentile-vel'] = np.percentile(pairwise_v, 50)
    feats['80percentile-vel'] = np.percentile(pairwise_v, 80)
    # pairwise_acc
    feats['20percentile-acc'] = np.percentile(pairwise_a, 20)
    feats['50percentile-acc'] = np.percentile(pairwise_a, 50)
    feats['80percentile-acc'] = np.percentile(pairwise_a, 80)

    feats['last3median-vel'] = np.median(pairwise_v[-3:])

    p1 = np.array([feats['start-x'], feats['start-y']])
    p2 = np.array([feats['stop-x'], feats['stop-y']])
    p3 = [np.array([mod_ts.iloc[n]['positionX'],
                    mod_ts.iloc[n]['positionY']])
          for n in range(mod_ts.shape[0])]
    e2e_deviation = np.cross(p2 - p1, p3 - p1) / norm(p2 - p1)
    feats['largest-dev'] = e2e_deviation.max()
    feats['20percentile-dev'] = np.percentile(e2e_deviation, 20)
    feats['50percentile-dev'] = np.percentile(e2e_deviation, 50)
    feats['80percentile-dev'] = np.percentile(e2e_deviation, 80)

    directions = [atan2((mod_ts.iloc[n + 1]['positionX'] -
                         mod_ts.iloc[n]['positionX']),
                        (mod_ts.iloc[n + 1]['positionY'] -
                         mod_ts.iloc[n]['positionY']))
                  for n in range(mod_ts.shape[0] - 1)]

    feats['avg-dir'] = np.mean(directions)

    feats['ratio-e2edist-traj'] = feats['e2e-distance'] / sum(resultants)

    feats['first5median-acc'] = np.median(pairwise_a[:5])

    feats['mid-stroke-p'] = mod_ts['eventPressure'].iloc[int(mod_ts.shape[0]/2)]

    return feats


def generate_ts():
    # Approximate bounds pulled from original timeseries dataset.
    BOUNDS = {'eventPressure': (0, 1.5),
              'positionY': (0, 1919.0),
              'positionX': (0, 1919.0)}
    our_ts = {}
    feat_list = ['index2',
                 'eventTime',
                 'eventPressure',
                 'positionX',
                 'positionY']
    # Pick a random number of points within our swipe (30-200 points)
    n_points = random.randint(30, 200)
    # Pick a random duration of the swipe (0.5-2.0 seconds)
    end_time = 0.5 + (random.random() * 1.5)

    # Generate even time series object
    time_sampler = TimeSynth.TimeSampler(start_time=0, stop_time=end_time)
    # Generate irregular timeseries equal to n_points
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=n_points*2,
                                                                keep_percentage=50)
    # Generate sample indexes
    our_ts['index2'] = np.arange(0, n_points)
    # Scale seconds into ms (value sampling remains in seconds)
    our_ts['eventTime'] = irregular_time_samples * 1000

    for feat in ['eventPressure', 'positionX', 'positionY']:
        # Value range between 0 and 1
        scaler = MinMaxScaler((0, 1))
        # sample values from a Continuous autoregressive process (CAR) object
        waveform = TimeSynth.signals.CAR(0.8)
        timeseries = TimeSynth.TimeSeries(waveform)
        samples, signals, errors = timeseries.sample(irregular_time_samples)
        samples = samples.reshape(-1, 1)
        # rescale sampled values between 0-1
        dat = scaler.fit_transform(samples)
        # scale values to our raw bounds and save
        our_ts[feat] = (dat * BOUNDS[feat][1]).reshape(1, -1)[0]

    return pd.DataFrame.from_dict(our_ts)


if __name__ == "__main__":
    data = pd.DataFrame()

    for _ in tqdm(range(100000)):
        b = generate_ts()
        a = extract_features(b)

        data = data.append(a, ignore_index=True)

        # data.to_csv('random_dump.csv')
        print("No automatic file output, manual save from ipython")
