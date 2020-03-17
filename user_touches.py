#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import ceil, hypot, atan2
import numpy as np
from numpy.linalg import norm
import pandas as pd

from tqdm import tqdm


class TouchFeatureExtraction:
    def __init__(self, data_loc, ts_data=None):
        self.data_loc = data_loc
        if ts_data is None:
            ts_data = pd.read_csv(data_loc, index_col=0)

        ts_data['SESSION'] = ts_data['SESSION'].astype('category')
        ts_data['USER'] = ts_data['USER'].astype('category')
        ts_data['index1'] = ts_data['index1'].astype('category')
        ts_data = ts_data[ts_data['USER'] != 'Takeout']
        attempts = np.unique(zip(ts_data['index1'],
                                 ts_data['USER'],
                                 ts_data['SESSION']),
                             return_counts=True, axis=0)

        self.skipped_list = []
        self.data = pd.DataFrame()
        for interact, n_samp in tqdm(zip(*attempts)):
            if n_samp < 5:
                self.skipped_list.append((interact, n_samp))
                continue
            else:
                inter_range = ts_data[((ts_data['SESSION'] == interact[2]) &
                                       (ts_data['USER'] == interact[1]) &
                                       (ts_data['index1'] == int(interact[0])))
                                      ]
                # print("Extract ", interact)
                feats = self.extract_features(inter_range)

                if feats is None:
                    self.skipped_list.append((interact, n_samp))
                    continue

                feats['USER'] = interact[1]
                feats['SESSION'] = interact[2]
                feats['index1'] = int(interact[0])

                self.data = self.data.append(feats, ignore_index=True)
        # For every timeseries, extract the following features.

    def get_data(self):
        return self.data

    def extract_features(self, ts):
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
        feats['inter-stroke-t'] = mod_ts['eventTime'].iloc[mod_ts.shape[0]/2]
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
        resultants = [hypot((mod_ts.iloc[n+1]['positionX'] -
                             mod_ts.iloc[n]['positionX']),
                            (mod_ts.iloc[n+1]['positionY'] -
                             mod_ts.iloc[n]['positionY']))
                      for n in range(mod_ts.shape[0]-1)]

        # No deviation, not a swipe
        if sum(resultants) == 0:
            return None

        feats['mean_resultant'] = np.mean(resultants)
        # Up, down, left, right flag
        feats['udlr_flag'] = udlr_flag(feats['start-x'], feats['stop-x'],
                                       feats['start-y'], feats['stop-y'])
        # Compute Velocity and Acceleration
        time_gaps = [(mod_ts.iloc[n+1]['eventTime'] -
                      mod_ts.iloc[n]['eventTime'])
                     for n in range(mod_ts.shape[0]-1)]
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
        e2e_deviation = np.cross(p2-p1, p3-p1)/norm(p2-p1)
        feats['largest-dev'] = e2e_deviation.max()
        feats['20percentile-dev'] = np.percentile(e2e_deviation, 20)
        feats['50percentile-dev'] = np.percentile(e2e_deviation, 50)
        feats['80percentile-dev'] = np.percentile(e2e_deviation, 80)

        directions = [atan2((mod_ts.iloc[n+1]['positionX'] -
                             mod_ts.iloc[n]['positionX']),
                            (mod_ts.iloc[n+1]['positionY'] -
                             mod_ts.iloc[n]['positionY']))
                      for n in range(mod_ts.shape[0]-1)]

        feats['avg-dir'] = np.mean(directions)

        feats['ratio-e2edist-traj'] = feats['e2e-distance']/sum(resultants)

        feats['first5median-acc'] = np.median(pairwise_a[:5])

        feats['mid-stroke-p'] = mod_ts['eventPressure'].iloc[mod_ts.shape[0]/2]

        return feats


class TouchUserPopulation:
    # Initialize the population prameters
    def __init__(self, sensor_data, n_feat):
        # Store the population statistics within the class
        self.data_loc = sensor_data
        self.data = pd.read_csv(self.data_loc, index_col=[0])

        self.n_feat = n_feat
        self.labels = self.data['USER']
        self.data.drop('USER', axis=1, inplace=True)
        self.data.drop('SESSION', axis=1, inplace=True)
        self.data.drop('index1', axis=1, inplace=True)
        # 273 Swipes have broken deviation data, negligible
        # within in the entire dataset, so we fill with 0
        # [  1282,   3772,   6081,   7236,   7433,   8169,   8243,   9061,
        #  9951,  10259,  10388,  10568,  10632,  10935,  11759,  11833,
        # 12043,  12327,  12397,  12404,  12408,  12661,  12664,  12888,
        # 13170,  14201,  14625,  14637,  14981,  15157,  15190,  15196,
        # 15444,  15959,  16209,  16451,  16898,  16930,  16935,  17498,
        # 18220,  18342,  18842,  19004,  19074,  19076,  19304,  19305,
        # 19531,  19539,  19933,  20010,  20963,  20964,  20967,  20971,
        # 22744,  22949,  22958,  23184,  23186,  23191,  23613,  23616,
        # 24244,  24653,  25523,  26321,  26385,  27160,  27164,  27167,
        # 27168,  27174,  28089,  28217,  28822,  28955,  29199,  29493,
        # 29718,  30441,  30631,  30932,  31813,  31999,  32927,  32968,
        # 33265,  33803,  34236,  35194,  35453,  35585,  37873,  37895,
        # 38479,  38845,  39137,  39408,  39414,  39550,  39687,  40895,
        # 42038,  42061,  42280,  42580,  42619,  43840,  44431,  44761,
        # 46746,  47252,  47544,  48303,  49134,  50376,  51049,  52069,
        # 52073,  53851,  54809,  55651,  56750,  57142,  57144,  57561,
        # 57562,  57638,  57639,  57642,  57792,  58514,  58991,  59479,
        # 59832,  60300,  61459,  62941,  62949,  63050,  64039,  64585,
        # 65596,  66337,  66458,  66711,  68586,  68823,  70368,  72348,
        # 72844,  73347,  74054,  75634,  75843,  76421,  76668,  77141,
        # 77325,  77339,  78060,  78187,  78313,  79060,  79326,  80195,
        # 81393,  81458,  81584,  82598,  83977,  84077,  86040,  86090,
        # 86987,  87011,  87037,  87134,  88218,  88857,  89865,  89909,
        # 90015,  90049,  90706,  91524,  91567,  91660,  91682,  93991,
        # 94331,  94394,  94869,  94879,  95591,  95723,  96371,  96393,
        # 96508,  96559,  96994,  97296,  99478,  99692, 100168, 100850,
        # 100861, 100886, 102980, 104109, 104262, 104840, 105023, 105048,
        # 105505, 106337, 108140, 108542, 109249, 109271, 109817, 109844,
        # 110439, 110561, 112143, 112989, 113001, 113009, 113017, 114018,
        # 114395, 115954, 116415, 116857, 116863, 117313, 117319, 118980,
        # 119112, 119680, 119697, 120090, 120397, 120482, 120515, 120521,
        # 120957, 121301, 121716, 121721, 121747, 122436, 122523, 122781,
        # 122916, 123807, 123835, 125114, 125543, 125849, 126223, 126230,
        # 126232, 126919, 126989, 127261, 127911, 127922, 127940, 127941,
        # 128261]
        self.data.dropna(inplace=True)
        # self.data = self.data[self.data['udlr_flag'] == 2]
        a = pd.get_dummies(self.data['udlr_flag'])
        self.data[['1_f', '2_f', '3_f', '4_f']] = a
        self.data.drop('udlr_flag', axis=1, inplace=True)

        self.data = self.data.astype(float)
        # self.data = self.data[self.data[]]

        users = {}
        for l in set(self.labels):
            dat = self.data[self.labels == l]
            users[l] = UserData(l, dat, dat.shape[0], n_feat)
        self.users = users
        self.n_users = len(set(self.labels))

    def get_user(self, i=None):
        if i is None:
            return self.users
        else:
            return self.users[i]

    def get_feature_means(self, i=None):
        if i is None:
            return np.mean([self.users[h].get_feature_means() for
                            h in range(self.n_samples)])
        else:
            return self.users[i].get_feature_means()

    def get_feature_stds(self, i=None):
        if i is None:
            return self.users
        else:
            return self.users[i].get_feature_stds()

    def normalize_data(self):
        self.scaler = MinMaxScaler()
        # Read the user's data to build the scaler
        for u, u_data in self.users.items():
            self.scaler.partial_fit(u_data.get_user_data())
        # Apply the common scaler on every user's data
        for u, u_data in self.users.items():
            self.users[u].normalize_data(self.scaler)

    def normalize_external(self, arr):
        return self.scaler.transform(arr)

    def split_user_data(self, test_split=0.2):
        self.test_split = test_split
        for u, u_data in self.users.items():
            u_data.split_user_data(test_split)

    def get_train_sets(self, training_user, concatenate=True):
        train_user = self.get_user(training_user)
        size = int(ceil((1.0-self.test_split)*train_user.get_user_data(
                    count=True) / self.n_users))
        pos_d = train_user.get_train_data()
        data_arr = []
        for u_n, user_data in self.users.items():
            if u_n != training_user:
                data = user_data.get_train_data()
                data_arr.append(data[np.random.choice(data.shape[0], size), :])

        neg_d = np.concatenate(data_arr)

        if concatenate:
            label = np.concatenate([np.ones(pos_d.shape[0]),
                                    np.zeros(neg_d.shape[0])])
            return np.concatenate([pos_d, neg_d]), label
        else:
            return pos_d, neg_d

    def get_test_sets(self, target_user, concatenate=True):
        test_user = self.get_user(target_user)
        size = int(ceil((self.test_split)*test_user.get_user_data(
                    count=True) / self.n_users))
        pos_d = test_user.get_test_data()
        data_arr = []
        for u_n, user_data in self.users.items():
            if u_n != target_user:
                data = user_data.get_test_data()
                data_arr.append(data[np.random.choice(data.shape[0], size), :])

        neg_d = np.concatenate(data_arr)

        if concatenate:
            label = np.concatenate([np.ones(pos_d.shape[0]),
                                    np.zeros(neg_d.shape[0])])
            return np.concatenate([pos_d, neg_d]), label
        else:
            return pos_d, neg_d


class UserData:
    # Initialize the user distributions
    def __init__(self, label, features, n_samples, n_dim):
        print(features.shape)
        print(n_samples, n_dim)
        assert features.shape == (n_samples, n_dim)
        self.label = label
        self.features = features
        self.n_samp = n_samples
        self.n_dim = n_dim

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def get_user_data(self, count=False):
        if count:
            return len(self.features)
        else:
            return self.features

    def get_feature_means(self):
        return np.mean(self.features, axis=1)

    def get_feature_stds(self):
        return np.std(self.features, axis=1)

    def normalize_data(self, scaler):
        self.unnormalize = self.features
        self.features = scaler.transform(self.features)

    def split_user_data(self, test_size):
        self.train, self.test = train_test_split(self.features,
                                                 test_size=test_size)

    def get_train_data(self):
        return self.train

    def get_test_data(self):
        return self.test


def extract_features_from_bulk():
    print("Extracting Features")
    touch_loc = "./data/TouchEventDictionary.csv"
    ts_data = None
    if ts_data is None:
        ts_data = pd.read_csv(touch_loc, index_col=0)
    extracted_features = TouchFeatureExtraction(touch_loc, ts_data)
    a = extracted_features.get_data()
    a.to_csv('extracted_touch_features.csv')


if __name__ == "__main__":
    data_loc = "./extracted_touch_features.csv"

    n_feat = 27

    user_touches = TouchUserPopulation(data_loc, n_feat)
    print(user_touches)

    print([u_data.get_user_data() for u, u_data in user_touches.users.items()
           if u == user_touches.labels[0]])
    user_touches.normalize_data()
    print([u_data.get_user_data() for u, u_data in user_touches.users.items()
           if u == user_touches.labels[0]])
    user_touches.split_user_data(0.3)

    for u in user_touches.users.keys():
        print(u)
        print(list(map(len, user_touches.get_train_sets(u,
                                                        concatenate=False))))
        print(list(map(len, user_touches.get_test_sets(u,
                                                       concatenate=False))))
        print('')

    random_loc = "./random_dump.csv"
    rand_data = pd.read_csv(random_loc, header=0, index_col=0)
    temp_a = pd.get_dummies(rand_data['udlr_flag'])
    rand_data[['1_f', '2_f', '3_f', '4_f']] = temp_a
    rand_data.drop('udlr_flag', axis=1, inplace=True)
    rand_data = rand_data.astype(float)

    rand_data = user_touches.normalize_external(rand_data.values)
