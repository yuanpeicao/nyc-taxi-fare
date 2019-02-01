#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 14:02:56 2018

@author: Yuanpei Cao
"""

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
from simpledbf import Dbf5

###############################################################################
## load training dataset
###############################################################################
## case 1: training 1
## directory 
##train_block_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
##                    'train_set/filter_train1_for_block.csv')
#
#train_original_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                       'filtered_data/filter_train1.csv')
#
#train_d_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
#                'block_id/train11_dropoff.dbf')
#
#train_p_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
#                'block_id/train1_pick.dbf')
#
#save_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#             'filtered_data/filter_train1_geoid.csv')

# case 2: training 2
# directory

train_original_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'filtered_data/filter_train2.csv')

train_d_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
                'block_id/train22_dropoff.dbf')

train_p_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
                'block_id/train2_pick.dbf')

save_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
             'filtered_data/filter_train2_geoid.csv')

###############################################################################
# load data
#train_df = pd.read_csv(train_block_file)
#train_df = train_df.rename(columns = {'Unnamed: 0': 'index'})
#train_df['index'] = train_df['index'] + 1

# get index for original file
train_o_df = pd.read_csv(train_original_file)
train_o_df = train_o_df.drop(['Unnamed: 0'], axis = 1)
train_o_df = train_o_df.drop(['ID'], axis = 1)
train_o_df['index'] = train_o_df.index.copy()

###############################################################################
## load block ID dataset
###############################################################################
train_p_df = Dbf5(train_p_file).to_dataframe()
train_d_df = Dbf5(train_d_file).to_dataframe()

###############################################################################
## inner join
###############################################################################
## get the pickup geoID
train_p_df = train_p_df[['Field1', 'GEOID']].drop_duplicates()

## drop duplicated fields
field_count = train_p_df.groupby(['Field1']).size().reset_index(name='counts')
field_count = field_count[field_count['counts'] == 1]
train_p_df = train_p_df[
        train_p_df['Field1'].isin(field_count['Field1'].unique())
        ]

## sort
train_p_df = train_p_df.sort_values(by = 'Field1').reset_index(drop=True)

# get the same index order
train_o_df = train_o_df[
        train_o_df['index'].isin(train_p_df['Field1'].unique())
        ].copy()

## sort
train_o_df = train_o_df.sort_values(by = 'index').reset_index(drop=True)

# combine together
train_o_df['geoid_p'] = train_p_df['GEOID']

## get the dropoff geoID
train_d_df = train_d_df[['Field1', 'GEOID']].drop_duplicates()
train_d_df = train_d_df[
        train_d_df['Field1'].isin(train_o_df['index'].unique())
        ].copy()

## drop duplicated fields
field_count = train_d_df.groupby(['Field1']).size().reset_index(name='counts')
field_count = field_count[field_count['counts'] == 1]
train_d_df = train_d_df[
        train_d_df['Field1'].isin(field_count['Field1'].unique())
        ]

## sort
train_d_df = train_d_df.sort_values(by = 'Field1').reset_index(drop=True)

# get the same index order
train_o_df = train_o_df[
        train_o_df['index'].isin(train_d_df['Field1'].unique())
        ].copy()

## sort
train_o_df = train_o_df.sort_values(by = 'index').reset_index(drop=True)

# combine together
train_o_df['geoid_d'] = train_d_df['GEOID']
train_o_df = train_o_df.drop(['index'], axis = 1)

## Remove observations with useless values base on the test data boundary
mask = train_o_df['pickup_longitude'].between(-74.3, -72.9)
mask &= train_o_df['dropoff_longitude'].between(-74.3, -72.9)
mask &= train_o_df['pickup_latitude'].between(40.5, 41.8)
mask &= train_o_df['dropoff_latitude'].between(40.5, 41.7)
mask &= train_o_df['passenger_count'].between(0, 6) # 1 - 6
mask &= train_o_df['fare_amount'].between(2, 200) # 2.5 - 200

# update training set
train_o_df = train_o_df[mask] 

################################################################################
### save result
################################################################################
train_o_df.to_csv(save_file, index = False)