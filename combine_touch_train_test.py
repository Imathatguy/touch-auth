#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao

The purpose of this script is to combine the training and testing files of the
University of Maryland Active Authentication-02 (UMDAA-02) Dataset, as we do
not seek to preserve the chronological train/test split of user recorded touch
activities.

Place this file in the same directory as the UMDAA-02 Dataset.

The UMDAA-02 Dataset is avaliable for download here:
https://umdaa02.github.io
"""
import pandas as pd

if __name__ == "__main__":
    train_loc = "./TrainEventDictionary_70.csv"
    test_loc = "./TestEventDictionary_70.csv"
    a = pd.read_csv(train_loc)
    b = pd.read_csv(test_loc)
    data = pd.concat([a, b], ignore_index=True)
    data.to_csv("./TouchEventDictionary.csv")
