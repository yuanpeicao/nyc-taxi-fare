#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:06:34 2018

@author: Yuanpei Cao
"""

import pandas as pd
import numpy as np
import catboost as cb
from catboost import Pool
#from math import sqrt
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import sys
sys.path.append('/Users/ycao/Desktop/taxi_fare_prediction/src')

###############################################################################
## set up parameter
###############################################################################
### Case 1: only train based on 2,3,4,5 subset
#trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train2.csv')
#trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train3.csv')
#trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train4.csv')
#trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train5.csv')
#
#pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'simple_model/train1/catboost_loc_time_1.pkl')

###############################################################################
## Case 2: only train based on 1,3,4,5 subset
trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train1.csv')
trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train3.csv')
trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train4.csv')
trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train5.csv')

pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
             'simple_model/train2/catboost_loc_time_2.pkl')

################################################################################
### Case 3: only train based on 1,2,4,5 subset
#trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train1.csv')
#trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train2.csv')
#trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train4.csv')
#trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train5.csv')
#
#pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'simple_model/train3/catboost_loc_time_3.pkl')

###############################################################################
## Case 4: only train based on 1,2,3,5 subset
#trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train1.csv')
#trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train2.csv')
#trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train3.csv')
#trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train5.csv')
#
#pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'simple_model/train4/catboost_loc_time_4.pkl')

###############################################################################
## Case 5: only train based on 1,2,3,4 subset
#trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train1.csv')
#trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train2.csv')
#trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train3.csv')
#trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train4.csv')
#
#pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'simple_model/train5/catboost_loc_time_5.pkl')

###############################################################################
## load dataset
###############################################################################
# load location and time only
fields = ['pickup_datetime', 'fare_amount',
          'pickup_latitude', 'pickup_longitude', 
          'dropoff_latitude', 'dropoff_longitude'
          ]

# read the data
data1 = pd.read_csv(trainset1, usecols = fields)
data2 = pd.read_csv(trainset2, usecols = fields)
data3 = pd.read_csv(trainset3, usecols = fields)
data4 = pd.read_csv(trainset4, usecols = fields)

# combine data
data = pd.concat([data1, data2, data3, data4], axis = 0)

# convert pickup datetime to informative features
data['pickup_datetime'] = data['pickup_datetime'].str.replace(" UTC", "")
data['pickup_datetime'] = pd.to_datetime(
            data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S'
            )
data['hour_of_day'] = data.pickup_datetime.dt.hour
data["weekday"] = data.pickup_datetime.dt.weekday
data['month'] = data.pickup_datetime.dt.month
data["year"] = data.pickup_datetime.dt.year

# remove unused features
data = data.drop(['pickup_datetime'], axis = 1)

###############################################################################
## Categorical gradient boosting regression
###############################################################################
# get feature and label
X = data[['hour_of_day', 'weekday', 'month', 'year',
          'pickup_latitude', 'pickup_longitude', 
          'dropoff_latitude', 'dropoff_longitude']]
Y = data['fare_amount']

# split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    random_state = 10, 
                                                    test_size = 0.25)

## fit the model
# initialize model parameter
params = {'iterations': 100,
          'learning_rate': 0.1,
          'eval_metric': 'RMSE',
          'random_seed': 42,
          'use_best_model': False
          }

# get the categorical feature index
cat_features_index = np.where(X.dtypes != np.float)[0]

# data pool
train_pool = Pool(X_train, Y_train, cat_features = cat_features_index)
test_pool = Pool(X_test, Y_test, cat_features = cat_features_index)

# fit the model
model = cb.CatBoostRegressor(**params)
model.fit(train_pool, eval_set = test_pool)

## performance
#Y_pred = model.predict(X_test)
#rmse = sqrt(mean_squared_error(Y_test, Y_pred))

###############################################################################
## save result
###############################################################################
# Save to file in the current working directory    
joblib.dump(model, pkl_filename)
