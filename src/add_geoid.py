#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:45:45 2018

@author: Yuanpei Cao
"""

import pandas as pd

#table = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
#                    'geoid/1wan_geoid.csv')
#
#table = table[['GEOID', 'GEOID_1']]
#
#t1 = table.groupby(['GEOID', 'GEOID_1']).size().reset_index(name='counts')
#t1 = t1.sort_values('counts')
#
#t2 = table.groupby(['GEOID']).size().reset_index(name='counts')
#t2 = t2.sort_values('counts')
#
#t3 = table.groupby(['GEOID_1']).size().reset_index(name='counts')
#t3 = t3.sort_values('counts')


###############################################################################
filter1 = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
                      'all/filtered_data/filter_train_holdout.csv')

filter1 = filter1[['pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude']]

filter1.to_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
               'all/filtered_data/filter_train_holdout_for_block.csv')