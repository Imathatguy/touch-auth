#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pickle

verify_data = True

data = 'user_proba_results'

vary_dir = {0.00: join('./', data),
            }

arches = {'linsvm': 'linsvm_probaresults',
          'rbfsvm': 'rbfsvm_probaresults',
          'rndf': 'rndf_probaresults',
          'tfdnn': 'dnn_probaresults',
          }

data_holder = {}

# lines_count = {}
for vary_key, data_path in vary_dir.items():
    for arch, arch_path in arches.items():
        arch_holder = data_holder.get(arch, {})
        run_path = join(data_path, arch_path)
        onlyfiles = [join(run_path, f) for f in listdir(run_path) if
                     isfile(join(run_path, f)) & f.endswith('.pickle')]

        # Extract data from dirrectory
        data_arr = []
        for n, file_path in enumerate(onlyfiles):
            print(n, file_path)
            with open(file_path, 'rb') as open_file:
                lines = pickle.load(open_file)
                # if len(lines) == 10:
                print(len(lines))
                data_arr.extend(lines)

        # Check there are a correct number of test runs in each extraction
        print(arch, len(data_arr))
        if verify_data:
            assert len(data_arr) == 50

        arch_holder[vary_key] = data_arr
        data_holder[arch] = arch_holder


arch_order = ['linsvm', 'rbfsvm', 'rndf', 'tfdnn']

arch_names = {'linsvm': 'Linear SVM',
              'rbfsvm': 'Radial SVM',
              'rndf': 'Random Forest',
              'tfdnn': 'Deep Neural Network',
              }

fig, axes = plt.subplots(ncols=len(arches), figsize=(14, 3), sharey=True)

for n, (ax, clasif) in enumerate(zip(axes, arch_order)):
    print(clasif)
    # print(data_df[data_df['classifier'] == clasif].mean())
    data = data_holder[clasif]
    labels, values = zip(*[(d, data[d]) for d in sorted(data.keys())])

    TPR = 1 - np.average([np.average(i[0], axis=0) for i in values[0]], axis=0)
    FPR = np.average([np.average(i[1], axis=0) for i in values[0]], axis=0)
    AR = np.average([np.average(i[2], axis=0) for i in values[0]], axis=0)
    RAR = np.average([np.average(i[4], axis=0) for i in values[0]], axis=0)

    x_ref = np.arange(0, 1.01, 0.01)

    idx = np.argwhere(np.abs(TPR - FPR) == np.min(np.abs(TPR - FPR))).flatten()
    eer_pos = idx[0]
    print(TPR[eer_pos])
    ax.annotate("{:.2f}".format(TPR[eer_pos]),
                xy=(x_ref[eer_pos], TPR[eer_pos]), color='C0')

    ar_equal_eer = True
    if ar_equal_eer:
        print(AR[eer_pos])
        ax.annotate("{:.2f}".format(AR[eer_pos]),
                    xy=(x_ref[eer_pos], AR[eer_pos]), color='C1')
        ax.axvline(x=x_ref[eer_pos], color='C0', linestyle='--', linewidth=0.7)

        print(RAR[eer_pos])
        ax.annotate("{:.2f}".format(RAR[eer_pos]),
                    xy=(x_ref[eer_pos], RAR[eer_pos]), color='C3')
    else:
        idx = np.argwhere(np.diff(np.sign(TPR - AR))).flatten()
        print(TPR[eer_pos])
        ax.annotate("{:.2f}".format(TPR[eer_pos]),
                    xy=(x_ref[eer_pos], TPR[eer_pos]), color='C1')

        idx = np.argwhere(np.diff(np.sign(TPR - RAR))).flatten()
        print(RAR[eer_pos])
        ax.annotate("{:.2f}".format(RAR[eer_pos]),
                    xy=(x_ref[eer_pos], RAR[eer_pos]), color='C3')

    print(idx)
    ax.plot(x_ref, TPR, label='{:>3} - {:.2f}'.format('FRR', TPR[eer_pos]),
            color='C2', linestyle='solid')
    ax.plot(x_ref, FPR, label='{:>3} - {:.2f}'.format('FPR', FPR[eer_pos]),
            color='C0', linestyle='dashed')
    ax.plot(x_ref, AR, label='{:>4.3} - {:.2f}'.format(' AR', AR[eer_pos]),
            color='C1', linestyle='dashdot')
    ax.plot(x_ref, RAR, label='{:>3} - {:.2f}'.format('RAR', RAR[eer_pos]),
            color='C3', linestyle='dotted')

    # ax.set_title(clasif)
    ax.set_xlabel('{} threshold'.format(str.upper(clasif)))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    if n == 0:
        ax.set_ylabel('Error')
    ax.legend()
    # plt.show()
plt.savefig('touch_proba_rand_roc.pdf', bbox_inches='tight')
plt.show()
