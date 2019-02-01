#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 20:46:31 2018

@author: Yuanpei Cao
"""

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from simpledbf import Dbf5

###############################################################################
## load outlier
###############################################################################
outlier_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
                'block_id/85_0diff.dbf')

# load data
outlier_df = Dbf5(outlier_file).to_dataframe()

outlier_df = outlier_df[['ID', 'pickup_lon', 'pickup_lat',
                         'dropoff_lo', 'dropoff_la']]

###############################################################################
## load test file
###############################################################################
test_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/test_for_block.csv')

df_test = pd.read_csv(test_file)

# add index column
df_test = df_test.reset_index()
df_test['index'] = df_test.index + 1

###############################################################################
## load prediction
###############################################################################
df_file = ('/Users/ycao/Desktop/taxi_fare_prediction/all/submission/'
           'best_submission/submission_train15_and_2baseline_final.csv')

df_prediction = pd.read_csv(df_file)

###############################################################################
## get outlier samples
###############################################################################
# get the key
outlier_df_final = pd.merge(outlier_df, df_test, how = 'inner',
                            left_on = ['ID'], right_on = ['index']
                            ).drop_duplicates()

# get the prediction
outlier_df_final = pd.merge(outlier_df_final, df_prediction, how = 'inner',
                            on = ['key']).drop_duplicates()

###############################################################################
## save results
###############################################################################
outlier_df_final = outlier_df_final.sort_values(
        by = ['fare_amount'], ascending = [False]
        )

outlier_df_final.to_csv(
        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
        'outlier_df_final.csv', index = False
        )