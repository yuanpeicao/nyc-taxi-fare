#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:30:06 2018

@author: Yuanpei Cao
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
#from time import time
#from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#import math

###############################################################################
# load osrm and training dataset
###############################################################################
# Adapted from https://www.kaggle.com/maheshdadhich/
# strength-of-visualization-python-visuals-tutorial

train_fr_1 = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
                         'new-york-city-taxi-with-osrm/'
                         'fastest_routes_train_part_1.csv')

train_fr_2 = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
                         'new-york-city-taxi-with-osrm/'
                         'fastest_routes_train_part_2.csv')

train_fr = pd.concat([train_fr_1, train_fr_2])
train_fr_new = train_fr[
        ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
        ]

# a different training set from OSRM (no fare amount)
data = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/'
                   'new-york-city-taxi-with-osrm/train.csv')

###############################################################################
# data preprocessing
###############################################################################
## clean data
# remove longitudes outliers
data = data[data['pickup_longitude'] >= -80]
data = data[data['pickup_longitude'] <= -70]
data = data[data['dropoff_longitude'] >= -80]
data = data[data['dropoff_longitude'] <= -70]

# remove latittudes outliers
data = data[data['pickup_latitude'] >= 35]
data = data[data['pickup_latitude'] <= 45] 
data = data[data['dropoff_latitude'] >= 35]
data = data[data['dropoff_latitude'] <= 45]

# remove missing samples
data = data.dropna()

# join two dataframes
data = pd.merge(data, train_fr_new, on = 'id', how = 'left')

osrm_data = data[
        ['pickup_longitude','pickup_latitude','dropoff_longitude',
         'dropoff_latitude','total_distance', 'total_travel_time', 
         'number_of_steps']
        ]
        
# remove missing samples
osrm_data = osrm_data.dropna()

# save result
osrm_data.to_csv(
        '/Users/ycao/Desktop/taxi_fare_prediction/'
        'new-york-city-taxi-with-osrm/osrm_data.csv',
        index = False
        )
        
###############################################################################
# fit KNN for OSRM train data
###############################################################################
# get feature and label
X = osrm_data[[
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude'
        ]]
Y_dist = osrm_data['total_distance']
Y_time = osrm_data['total_travel_time']
Y_step = osrm_data['number_of_steps']

## fit the model
# distance
neigh_dist = KNeighborsRegressor(n_neighbors = 6)
neigh_dist.fit(X, Y_dist) 

# time
neigh_time = KNeighborsRegressor(n_neighbors = 7)
neigh_time.fit(X, Y_time) 

# step
neigh_step = KNeighborsRegressor(n_neighbors = 9)
neigh_step.fit(X, Y_step) 

################################################################################
## cross-validation
################################################################################
#Y = Y_step.copy()
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
##neighbors = [5,6,7,8,9,10]
#neighbors = [1,5,6,7,8,9,10,50,100]
#for n in neighbors:
#    neigh = KNeighborsRegressor(n_neighbors = n)
#    neigh.fit(x_train, y_train) 
#
#    # predict the result
#    y_pred = neigh.predict(x_test)
#
#    # calculate RMSE
#    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
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
# save model
###############################################################################
# Save to file in the current working directory    
pkl_dist_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                     'simple_model/nn_osrm_dist.pkl')

pkl_time_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                     'simple_model/nn_osrm_time.pkl')

pkl_step_filename = ('/Users/ycao/Desktop/taxi_fare_prediction/'
                     'simple_model/nn_osrm_step.pkl')

joblib.dump(neigh_dist, pkl_dist_filename)
joblib.dump(neigh_time, pkl_time_filename)
joblib.dump(neigh_step, pkl_step_filename)