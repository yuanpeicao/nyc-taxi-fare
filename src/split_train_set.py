#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 23:54:36 2018

@author: Yuanpei Cao
"""

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
from sklearn.utils import shuffle

###############################################################################
## load training set
###############################################################################
data = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                   'filtered_data/filter_train.csv')

###############################################################################
## random split data
###############################################################################
# shuffle the table
data = shuffle(data, random_state = 0)

# split the table into one holdout table and 5 equal size tables
df_train1 = data.iloc[0: 10000000]
df_train2 = data.iloc[10000000: 20000000]
#df_train3 = data.iloc[20000000: 30000000]
#df_train4 = data.iloc[30000000: 40000000]
#df_train5 = data.iloc[40000000: 50000000]
#df_holdout = data.iloc[50000000: data.shape[0]]

###############################################################################
## save tables
###############################################################################
df_train1.to_csv(
        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
        'filtered_data/filter_train1.csv',
        index = True
        )

df_train2.to_csv(
        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
        'filtered_data/filter_train2.csv',
        index = True
        )

#df_train3.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'filtered_data/filter_train3.csv',
#        index = False
#        )
#
#df_train4.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'filtered_data/filter_train4.csv',
#        index = False
#        )
#
#df_train5.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'filtered_data/filter_train5.csv',
#        index = False
#        )

#df_holdout.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'filtered_data/filter_train_holdout.csv',
#        index = False
#        )