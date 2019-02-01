#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:15:20 2018

@author: Yuanpei Cao
"""

import pandas as pd
#from time import time
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

###############################################################################
## set up parameter
###############################################################################
### Case 1: only train based on 2,3,4,5 subset
trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train2.csv')
trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train3.csv')
trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train4.csv')
trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train5.csv')

pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
             'simple_model/train1/nn_pure_loc_1.pkl')

################################################################################
### Case 2: only train based on 1,3,4,5 subset
#trainset1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train1.csv')
#trainset2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train3.csv')
#trainset3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train4.csv')
#trainset4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train5.csv')
#
#pkl_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
#             'simple_model/train2/nn_pure_loc_2.pkl')

###############################################################################
# Case 3: only train based on 1,3,4,5 subset
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
#             'simple_model/train3/nn_pure_loc_3.pkl')

###############################################################################
### Case 4: only train based on 1,2,3,5 subset
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
#             'simple_model/train4/nn_pure_loc_4.pkl')

###############################################################################
### Case 5: only train based on 1,2,3,4 subset
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
#             'simple_model/train5/nn_pure_loc_5.pkl')

###############################################################################
## load dataset
###############################################################################
# load location and time only
fields = ['pickup_latitude', 'pickup_longitude', 
          'dropoff_latitude', 'dropoff_longitude',
          'fare_amount']

# read the data
data1 = pd.read_csv(trainset1, usecols = fields)
data2 = pd.read_csv(trainset2, usecols = fields)
data3 = pd.read_csv(trainset3, usecols = fields)
data4 = pd.read_csv(trainset4, usecols = fields)

# combine data
data = pd.concat([data1, data2, data3, data4], axis = 0)

###############################################################################
## KNN regression
###############################################################################
# get feature and label
X = data[[
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude'
        ]]
Y = data['fare_amount']

# fit the model
neigh = KNeighborsRegressor(n_neighbors = 5)
neigh.fit(X, Y) 

## split data into training and test set
#x_train, x_test, y_train, y_test = train_test_split(
#        X, Y, test_size = 0.2
#        )
#
## initialize the result list
#test_results = {}
#best_rmse = 100000
#
## fit the model
#start = time()
#neighbors = [1, 3, 5, 7, 10]
#for n in neighbors:
#    neigh = KNeighborsRegressor(n_neighbors = n)
#    neigh.fit(x_train, y_train) 
#
#    # predict the result
#    y_pred = neigh.predict(x_test)
#
#    # calculate RMSE
#    rmse = mean_squared_error(y_test, y_pred)
#    
#    # get the best parameter
#    if rmse < best_rmse:
#        best_rmse = rmse
#        best_neigh = neigh
#    
#    # record the performance
#    test_results[n] = rmse
#    
#    # print result
#    print(n)
#    print(rmse)
#
## record the time
#stop = time()

###############################################################################
## save result
###############################################################################
# Save to file in the current working directory    
joblib.dump(neigh, pkl_filename)