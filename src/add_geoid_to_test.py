#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:45:31 2018

@author: Yuanpei Cao
"""

from simpledbf import Dbf5
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


# load geoid data
test_block_file = ('/Users/ycao/Dropbox/Yuan_Xin/kaggle-taxi-fare/'
                   'block_id/test_block_id.dbf')
test_block_df = Dbf5(test_block_file).to_dataframe()

# drop unused feature
test_block_df = test_block_df.drop(['ID_1'], axis = 1)

# sort ID
test_block_df = test_block_df.sort_values(by = ['ID'], ascending = [True])

# rename
test_block_df = test_block_df.rename(
        columns = {'GEOID': 'geoid_p', 'GEOID_1':'geoid_d'}
        )

# load test data
test_df = pd.read_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/test.csv')

# reset index
test_block_df = test_block_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# get geoid
test_df['geoid_p'] = test_block_df['geoid_p']
test_df['geoid_d'] = test_block_df['geoid_d']

# save result
test_df.to_csv('/Users/ycao/Desktop/taxi_fare_prediction/all/test_w_geoid.csv',
               index = False)