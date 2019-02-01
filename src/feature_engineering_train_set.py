#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 00:21:18 2018

@author: Yuanpei Cao
"""

import pandas as pd
import numpy as np
from time import time
import sys
sys.path.append('/Users/ycao/Desktop/taxi_fare_prediction/src')
from basic_function import prepare_time_features, distance, airport_feats
from basic_function import calculate_fare, nn_reg_on_location
from basic_function import predict_osrm_feature, catboost_on_time
from basic_function import catboost_on_time_loc, county_feats
from basic_function import year_weekday_location_fare_stat

start = time()

###############################################################################
## load dataset
###############################################################################
### Case 1: training set 1
##df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
##                       'filtered_data/filter_train1.csv', nrows = 10000)
#
#df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                       'filtered_data/filter_train1.csv')
#
## file for saving statistic features
#stat_file = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'stat_feature/y_wd_loc_1.csv')
#
## file for saving processed dataset
#df_train_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_train1.csv')

## simple models
#pkl_nn_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                        'simple_model/train1/nn_pure_loc_1.pkl')
#
#pkl_catboost_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train1/catboost_pure_time_1.pkl')
#
#pkl_catboost_time_loc = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train1/catboost_loc_time_1.pkl')

################################################################################
### Case 2: training set 2
#df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                       'filtered_data/filter_train2.csv')
#
## file for saving statistic features
#stat_file = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'stat_feature/y_wd_loc_2.csv')
#
## file for saving processed dataset
#df_train_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_train2.csv')
#
## simple models
#pkl_nn_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                        'simple_model/train2/nn_pure_loc_2.pkl')
#
#pkl_catboost_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train2/catboost_pure_time_2.pkl')
#
#pkl_catboost_time_loc = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train2/catboost_loc_time_2.pkl')

################################################################################
## Case 3: training set 3
df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'filtered_data/filter_train3.csv')

# file for saving statistic features
stat_file = ('/Users/ycao/Desktop/taxi_fare_prediction/'
             'stat_feature/y_wd_loc_3.csv')

# file for saving processed dataset
df_train_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                 'processed_filtered_data/processed_train3.csv')

# simple models
pkl_nn_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                        'simple_model/train3/nn_pure_loc_3.pkl')

pkl_catboost_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                              'simple_model/train3/catboost_pure_time_3.pkl')

pkl_catboost_time_loc = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                              'simple_model/train3/catboost_loc_time_3.pkl')

################################################################################
### Case 4: training set 4
#df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                       'filtered_data/filter_train4.csv')
#
## file for saving statistic features
#stat_file = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'stat_feature/y_wd_loc_4.csv')
#
## file for saving processed dataset
#df_train_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_train4.csv')
#
## simple models
#pkl_nn_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                        'simple_model/train4/nn_pure_loc_4.pkl')
#
#pkl_catboost_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train4/catboost_pure_time_4.pkl')
#
#pkl_catboost_time_loc = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train4/catboost_loc_time_4.pkl')

###############################################################################
## Case 5: training set 5
#df_train = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                       'filtered_data/filter_train5.csv')
#
## file for saving statistic features
#stat_file = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'stat_feature/y_wd_loc_5.csv')
#
## file for saving processed dataset
#df_train_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_train5.csv')
#
## simple models
#pkl_nn_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                        'simple_model/train5/nn_pure_loc_5.pkl')
#
#pkl_catboost_pure_location = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train5/catboost_pure_time_5.pkl')
#
#pkl_catboost_time_loc = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#                              'simple_model/train5/catboost_loc_time_5.pkl')

###############################################################################
## feature engineering
###############################################################################
# convert the pickup datetime
df_train[['hour_of_day', 'week', 'month', 'year',
          'day_of_year', 'weekday', 'quarter',
          'day_of_month']] = prepare_time_features(
          df_train[['pickup_datetime']].copy()
          )

# get stat features
df_train[['mean_fare', 'median_fare', 'max_fare']] = \
year_weekday_location_fare_stat(
        df_train[[
                'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude',
                'year', 'weekday', 'ID', 'pickup_datetime'
                ]].copy(), 
                stat_file
                )

# drop unused feature
df_train = df_train.drop(['ID', 'pickup_datetime'], axis = 1)

# calculate the fare by pure location-based KNN prediction 
df_train[['nn_fare_pure_location']] = nn_reg_on_location(
        df_train[[
                'pickup_latitude', 'pickup_longitude', 
                'dropoff_latitude', 'dropoff_longitude'
                ]].copy(), 
        pkl_nn_pure_location
        )

# calculate the fare by OSRM data-based KNN prediction 
df_train[['osrm_distance', 'osrm_time', 'osrm_number_of_steps']] = \
predict_osrm_feature(df_train[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
        ]].copy())

# calculate the fare by pure time-based Catboost prediction 
df_train[['catboost_fare_pure_time']] = \
catboost_on_time(df_train[['hour_of_day', 'weekday', 'month', 'year']].copy(), 
                 pkl_catboost_pure_location)

# calculate the fare by time & location-based Catboost prediction 
df_train[['catboost_fare_time_loc']] = \
catboost_on_time_loc(df_train[[
        'hour_of_day', 'weekday', 'month', 'year',
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude']].copy(), 
pkl_catboost_time_loc)

# calculate the distance based on dropoff and pickup locations
df_train[['sphere_dist', 'Euclidean', 'manh_length', 'Euc_error']] = distance(
        df_train[[
                'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude'
                ]].copy()
        )

# check if the distance is nearly 0
df_train['no_loc_change'] = np.where((df_train['manh_length'] < 0.01), 1, 0)

# calculate the distance to each airport and downtown
df_train[['pickup_manh_length_nyc', 'dropoff_manh_length_nyc',
          'pickup_manh_length_jfk', 'dropoff_manh_length_jfk',
          'pickup_manh_length_ewr', 'dropoff_manh_length_ewr',
          'pickup_manh_length_lgr', 'dropoff_manh_length_lgr']] = \
          airport_feats(df_train[[
                  'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude'
                  ]].copy())

# calculate the distance to seven counties
df_train[['pickup_manh_length_nas', 'dropoff_manh_length_nas',
          'pickup_manh_length_suf', 'dropoff_manh_length_suf',
          'pickup_manh_length_wes', 'dropoff_manh_length_wes',
          'pickup_manh_length_roc', 'dropoff_manh_length_roc',
          'pickup_manh_length_dut', 'dropoff_manh_length_dut',
          'pickup_manh_length_ora', 'dropoff_manh_length_ora',
          'pickup_manh_length_put', 'dropoff_manh_length_put']] = \
          county_feats(df_train[[
                  'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude'
                  ]].copy())

# calculate the fare
df_train[['manh_fare', 'euc_fare', 'peak_hour', 'night_hour',
          'county_dropoff_1', 'county_dropoff_2', 'to_from_jfk',
          'jfk_rush_hour', 'ewr']] = \
calculate_fare(
        df_train[['manh_length', 'hour_of_day', 'weekday','Euclidean',
                  'pickup_manh_length_nyc', 'dropoff_manh_length_nas',
                  'dropoff_manh_length_wes', 'dropoff_manh_length_suf',
                  'dropoff_manh_length_roc', 'dropoff_manh_length_dut',
                  'dropoff_manh_length_ora', 'dropoff_manh_length_put',
                  'pickup_manh_length_jfk', 'dropoff_manh_length_nyc',
                  'dropoff_manh_length_jfk', 'dropoff_manh_length_ewr']].copy()
        )

###############################################################################
## save result
###############################################################################
df_train.to_csv(df_train_file, index = False)


stop = time()
print(stop - start)