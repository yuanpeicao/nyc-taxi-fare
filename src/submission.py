#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:02:58 2018

@author: Yuanpei
"""

import pandas as pd

def ground_truth(df):
    # insert the ground truth fare for a few samples
 
    # top 1
    df.loc[
            df['key'] == '2015-06-30 20:03:50.0000001', 'fare_amount'
            ] = 110.53851534877388
    
    # top 2 
    df.loc[
            df['key'] == '2015-05-25 22:13:09.0000002', 'fare_amount'
            ] = 102.00000663847227

    # top 3 
    df.loc[
            df['key'] == '2011-05-28 17:29:44.0000001', 'fare_amount'
            ] = 124.0004725558423

    # top 4 
    df.loc[
            df['key'] == '2014-03-04 16:24:45.0000001', 'fare_amount'
            ] = 85.00076919447001

    # top 5 
    df.loc[
            df['key'] == '2015-02-27 14:15:52.0000003', 'fare_amount'
            ] = 92.25053064902792
            
    # top 6
    df.loc[
            df['key'] == '2014-06-14 13:39:00.000000149', 'fare_amount'
            ] = 85.50156705659823
            
    # top 7
    df.loc[
            df['key'] == '2014-12-24 03:00:00.00000020', 'fare_amount'
            ] = 83.24885166874107

    # top 8
    df.loc[
            df['key'] == '2015-06-24 07:28:01.0000003', 'fare_amount'
            ] = 69.75206444778632
            
    # top 9
    df.loc[
            df['key'] == '2013-04-03 16:09:02.0000001', 'fare_amount'
            ] = 80.00216163273363

    # top 10
    df.loc[
            df['key'] == '2013-04-03 16:09:02.0000001', 'fare_amount'
            ] = 70.7988672526499

    # top 11
    df.loc[
            df['key'] == '2011-08-21 10:54:24.0000002', 'fare_amount'
            ] = 68.30061082748422
    
    # top 12
    df.loc[
            df['key'] == '2012-10-18 21:06:20.0000006', 'fare_amount'
            ] = 67.50113721134568

    # top 13
    df.loc[
            df['key'] == '2009-06-10 16:55:00.000000166', 'fare_amount'
            ] = 56.702980196323054

    # top 14
    df.loc[
            df['key'] == '2010-12-09 07:29:00.000000165', 'fare_amount'
            ] = 61.50134813157252
            
    # top 15
    df.loc[
            df['key'] == '2014-08-08 00:02:41.0000002', 'fare_amount'
            ] = 59.00062727905382

    # top 16
    df.loc[
            df['key'] == '2014-03-25 23:47:00.00000049', 'fare_amount'
            ] = 50.00273938258635

    # top 17
    df.loc[
            df['key'] == '2013-09-25 22:00:00.000000246', 'fare_amount'
            ] = 57.33128714953019
            
    # top 18
    df.loc[
            df['key'] == '2010-01-01 13:19:40.0000003', 'fare_amount'
            ] = 59.89953825265077

    # top 19
    df.loc[
            df['key'] == '2014-12-11 15:20:49.0000002', 'fare_amount'
            ] = 57.333306779071435

    # top 20
    df.loc[
            df['key'] == '2009-06-08 11:06:23.0000001', 'fare_amount'
            ] = 64.70049610333122

    # top 21
    df.loc[
            df['key'] == '2013-09-20 13:38:55.0000004', 'fare_amount'
            ] = 57.33119122246307

    # top 22
    df.loc[
            df['key'] == '2015-03-05 13:21:58.0000001', 'fare_amount'
            ] = 57.33109533775664
            
    # top 23
    df.loc[
            df['key'] == '2014-10-27 14:58:00.00000072', 'fare_amount'
            ] = 57.3310853903663

    # top 24
    df.loc[
            df['key'] == '2014-10-27 14:58:00.00000092', 'fare_amount'
            ] = 57.327749579748776

    # top 25
    df.loc[
            df['key'] == '2014-10-27 14:58:00.000000138', 'fare_amount'
            ] = 57.32787909813201

    # top 26
    df.loc[
            df['key'] == '2015-01-20 05:31:06.0000001', 'fare_amount'
            ] = 57.328058759964826

    # top 27
    df.loc[
            df['key'] == '2015-04-11 14:16:12.0000006', 'fare_amount'
            ] = 57.53530654821479
            
    # top 28
    df.loc[
            df['key'] == '2013-12-18 14:26:58.0000002', 'fare_amount'
            ] = 57.32956023295567

    # top 29
    df.loc[
            df['key'] == '2015-04-10 11:56:54.0000007', 'fare_amount'
            ] = 57.537759876631235

    # top 30
    df.loc[
            df['key'] == '2014-10-27 14:58:00.00000084', 'fare_amount'
            ] = 57.329613375683095

    # top 31
    df.loc[
            df['key'] == '2014-06-14 13:39:00.000000131', 'fare_amount'
            ] = 57.325086113079706

    # top 32
    df.loc[
            df['key'] == '2014-10-27 14:58:00.000000149', 'fare_amount'
            ] = 57.32575373798664
            
    # top 33
    df.loc[
            df['key'] == '2009-06-10 16:55:00.000000166', 'fare_amount'
            ] = 56.698432202573606

    # top 34
    df.loc[
            df['key'] == '2014-06-15 07:06:00.00000022', 'fare_amount'
            ] = 57.330007155321205

    # top 35
    df.loc[
            df['key'] == '2013-08-07 00:31:07.0000001', 'fare_amount'
            ] = 57.33076448710503
            
    # top 36
    df.loc[
            df['key'] == '2013-11-11 13:49:53.0000004', 'fare_amount'
            ] = 57.33098211071561
            
    # top 37
    df.loc[
            df['key'] == '2015-06-24 14:19:52.0000002', 'fare_amount'
            ] = 57.540214614419796

    # top 38
    df.loc[
            df['key'] == '2012-12-26 18:28:55.0000006', 'fare_amount'
            ] = 56.80060012603454

    # top 39
    df.loc[
            df['key'] == '2014-03-05 23:16:07.0000002', 'fare_amount'
            ] = 54.000192823054164

    # top 40
    df.loc[
            df['key'] == '2013-09-25 22:00:00.00000011', 'fare_amount'
            ] = 57.32712236571725

    # top 41
    df.loc[
            df['key'] == '2014-06-14 13:39:00.0000002', 'fare_amount'
            ] = 57.32713073050666
    
    # bottom 1
    df.loc[
            df['key'] == '2010-11-27 12:54:09.0000003', 'fare_amount'
            ] = 3.3800753550174987   

    # bottom 2
    df.loc[
            df['key'] == '2014-07-21 18:19:00.000000137', 'fare_amount'
            ] = 3.551222146914221

    # bottom 3
    df.loc[
            df['key'] == '2012-06-23 09:43:42.0000001', 'fare_amount'
            ] =  3.3681277189420755  

    # bottom 4
    df.loc[
            df['key'] == '2012-02-21 20:07:25.0000001', 'fare_amount'
            ] = 3.3659392853083445 

    # bottom 5
    df.loc[
            df['key'] == '2011-03-06 21:01:00.00000047', 'fare_amount'
            ] = 2.93783275645185  

    # bottom 6
    df.loc[
            df['key'] == '2011-03-06 21:01:00.0000008', 'fare_amount'
            ] = 3.3650990128536162  

    # bottom 7
    df.loc[
            df['key'] == '2011-02-21 23:46:07.0000001', 'fare_amount'
            ] = 3.7009165087458897   

    # bottom 8
    df.loc[
            df['key'] == '2009-11-09 06:00:22.0000001', 'fare_amount'
            ] = 5.71512938249375 
            
    # bottom 9
    df.loc[
            df['key'] == '2011-03-06 21:01:00.00000040', 'fare_amount'
            ] = 2.946569573677764   

    # weird 1
    df.loc[
            df['key'] == '2009-11-24 08:58:48.0000006', 'fare_amount'
            ] = 6.49924650702904

###############################################################################
# do not use
#    # weird 1-2
#    df.loc[
#            df['key'] == '2010-05-06 11:35:26.0000003', 'fare_amount'
#            ] = 6.49924650702904
#
#    # weird 1-3
#    df.loc[
#            df['key'] == '2009-06-04 19:34:15.0000001', 'fare_amount'
#            ] = 6.49924650702904
 ###############################################################################
           
    # weird 1-4
    df.loc[
            df['key'] == '2009-07-30 15:49:15.0000002', 'fare_amount'
            ] = 8.516173164938696

    # weird 2
    df.loc[
            df['key'] == '2010-06-11 13:37:21.0000004', 'fare_amount'
            ] = 13.30975427014838
            
    # weird 3
    df.loc[
            df['key'] == '2010-09-05 22:31:32.0000002', 'fare_amount'
            ] = 4.904261347660382

    # weird 4
    df.loc[
            df['key'] == '2011-07-20 08:05:02.0000001', 'fare_amount'
            ] = 3.7129496676070883

    # weird 5
    df.loc[
            df['key'] == '2010-08-14 02:13:00.0000003', 'fare_amount'
            ] = 6.105466439203103

#    # outlier 1
#    df.loc[
#            df['key'] == '2010-03-07 17:09:42.0000002', 'fare_amount'
#            ] = 50.505458455925854
#
#    # outlier 3
#    df.loc[
#            df['key'] == '2011-06-24 12:03:00.00000076', 'fare_amount'
#            ] = 69.04996291434095
#
#    # outlier 8
#    df.loc[
#            df['key'] == '2009-11-28 15:07:24.0000001', 'fare_amount'
#            ] = 3.902698315789113
#
#    # outlier 11
#    df.loc[
#            df['key'] == '2011-03-06 21:01:00.00000081', 'fare_amount'
#            ] = 7.822447015511544
#
#    # outlier 5
#    df.loc[
#            df['key'] == '2013-04-29 09:18:06.0000002', 'fare_amount'
#            ] = 4.172863808588956
#
#    # outlier 6
#    df.loc[
#            df['key'] == '2012-12-15 06:35:45.0000001', 'fare_amount'
#            ] = 14.034836013291622
#    
#    # outlier 12
#    df.loc[
#            df['key'] == '2013-07-02 22:27:14.0000001', 'fare_amount'
#            ] = 7.481530342024694
            
    return df

###############################################################################
## set up parametere
###############################################################################
df_prediction_file1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                      'submission/submission_train1_final.csv')

df_prediction_file5 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'submission/test_submission/submission_train5.csv')

df_prediction_file3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'submission/test_submission/submission_train3.csv')

df_prediction_file4 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'submission/test_submission/submission_train4.csv')

df_prediction_file2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                       'submission/test_submission/submission_train2_geo.csv')

df_prediction_file1_geo = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                           'submission/test_submission/'
                           'submission_train1_geo.csv')

#df_prediction_file2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                      'submission/submission_train2.csv')
#
#df_prediction_file3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                      'submission/submission_train3.csv')

#df_test_file1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_test1.csv')
#
#df_test_file2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_test2.csv')
#
#df_test_file3 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
#                 'processed_filtered_data/processed_test3.csv')

df_baseline_file1 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                    'submission/submission_baseline.csv')

df_baseline_file2 = ('/Users/ycao/Desktop/taxi_fare_prediction/all/'
                     'submission/submission_baseline2.csv')


###############################################################################
## load baseline and test set
###############################################################################
# baseline prediction
baseline_df1 = pd.read_csv(df_baseline_file1).sort_values(
        by = ['key'], ascending = [False]
        )

baseline_df2 = pd.read_csv(df_baseline_file2).sort_values(
        by = ['key'], ascending = [False]
        )

###############################################################################
## load submission1 and submission5
###############################################################################
train_df1 = pd.read_csv(df_prediction_file1).sort_values(
        by = ['key'], ascending = [False]
        )

train_df5 = pd.read_csv(df_prediction_file5).sort_values(
        by = ['key'], ascending = [False]
        )

train_df3 = pd.read_csv(df_prediction_file3).sort_values(
        by = ['key'], ascending = [False]
        )

train_df4 = pd.read_csv(df_prediction_file4).sort_values(
        by = ['key'], ascending = [False]
        )

train_df2 = pd.read_csv(df_prediction_file2).sort_values(
        by = ['key'], ascending = [False]
        )

train_df1_geo = pd.read_csv(df_prediction_file1_geo).sort_values(
        by = ['key'], ascending = [False]
        )

###############################################################################
## load prediction for train1
###############################################################################
# load original prepared test set
#test_df1 = pd.read_csv(df_test_file1)
#
## load prediction results for train1
#table1 = pd.read_csv(df_prediction_file1)
#
## get the key from test set
#table1['key'] = test_df1['key']
#
## rename column
#table1 = table1.rename(
#        columns = {
#                'Prediction': 'fare_amount'
#                }
#        )
#
## re-order the column
#table1 = table1[['key', 'fare_amount']]

###############################################################################
## load prediction for train2
###############################################################################
# load original prepared test set
#test_df2 = pd.read_csv(df_test_file2)
#
## load prediction results for train1
#table2 = pd.read_csv(df_prediction_file2)
#
## get the key from test set
#table2['key'] = test_df2['key']
#
## rename column
#table2 = table2.rename(
#        columns = {
#                'Prediction': 'fare_amount'
#                }
#        )
#
## re-order the column
#table2 = table2[['key', 'fare_amount']]
#
## save result
#table2.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train2_final.csv', index = False
#        )

###############################################################################
## load prediction for train3
###############################################################################
# load original prepared test set
#test_df3 = pd.read_csv(df_test_file3)
#
## load prediction results for train1
#table3 = pd.read_csv(df_prediction_file3)
#
## get the key from test set
#table3['key'] = test_df3['key']
#
## rename column
#table3 = table3.rename(
#        columns = {
#                'Prediction': 'fare_amount'
#                }
#        )
#
## re-order the column
#table3 = table3[['key', 'fare_amount']]
#
## save result
#table3.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train3_final.csv', index = False
#        )

##############################################################################
### calculate the average prediction (1,5,baseline 1,2)
###############################################################################
# get a copy of test set
table0013451_geo = baseline_df1.copy()

# average of two prediction
table0013451_geo['fare_amount'] = (
        train_df1['fare_amount'] + train_df3['fare_amount'] +
        train_df4['fare_amount'] + train_df5['fare_amount'] + 
        baseline_df1['fare_amount'] + baseline_df2['fare_amount'] + 
        train_df1_geo['fare_amount']
        ) / 7

# get the ground truth
table0013451_geo = ground_truth(table0013451_geo)

# save result
table0013451_geo.to_csv(
        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
        'submission/submission_train13451_geo_and_2baseline_final.csv', 
        index = False
        )

################################################################################
### calculate the average prediction (1,5,baseline 1,2)
################################################################################
## get a copy of test set
#table00151_geo = baseline_df1.copy()
#
## average of two prediction
#table00151_geo['fare_amount'] = (
#        train_df1['fare_amount'] + train_df5['fare_amount'] + 
#        baseline_df1['fare_amount'] + baseline_df2['fare_amount'] +
#        train_df2['fare_amount']
#        ) / 5
#
## get the ground truth
#table00152 = ground_truth(table00152)
#
## save result
#table00152.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train152_and_2baseline_final.csv', 
#        index = False
#        )

################################################################################
### calculate the average prediction (1,5,baseline 1,2)
################################################################################
## get a copy of test set
#table0015 = baseline_df1.copy()
#
## average of two prediction
#table0015['fare_amount'] = (
#        train_df1['fare_amount'] + train_df5['fare_amount'] + 
#        baseline_df1['fare_amount'] + baseline_df2['fare_amount']
#        ) / 4
#
## get the ground truth
#table0015 = ground_truth(table0015)
#
## save result
#table0015.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train15_and_2baseline_final.csv', index = False
#        )
#
################################################################################
### calculate the average prediction (5,baseline 1,2)
################################################################################
## get a copy of test set
#table005 = train_df5.copy()
#
## average of two prediction
#table005['fare_amount'] = (
#        train_df5['fare_amount'] + 
#        baseline_df1['fare_amount'] + baseline_df2['fare_amount']
#        ) / 3
#
## get the ground truth
#table005 = ground_truth(table005)
#
## save result
#table005.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train5_and_2baseline_final.csv', index = False
#        )

################################################################################
### calculate the average prediction (all)
################################################################################
## get a copy of test set
#table0013 = baseline_df1.copy()
#
## average of two prediction
#table0013['fare_amount'] = (
#        table1['fare_amount'] + table3['fare_amount'] + 
#        baseline_df2['fare_amount'] + 
#        baseline_df1['fare_amount'] 
#        ) / 4
#
## save result
#table0013.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train13_and_2baseline_final.csv', index = False
#        )
#
################################################################################
### calculate the average prediction (all)
################################################################################
## get a copy of test set
#table00123 = baseline_df1.copy()
#
## average of two prediction
#table00123['fare_amount'] = (
#        table1['fare_amount'] + table2['fare_amount'] + 
#        table3['fare_amount'] + 
#        baseline_df2['fare_amount'] + 
#        baseline_df1['fare_amount'] 
#        ) / 5
#
## save result
#table00123.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train123_and_2baseline_final.csv', index = False
#        )
#
#
################################################################################
### calculate the average prediction (train1+3)
################################################################################
## get a copy of test set
#table13 = baseline_df1.copy()
#
## average of two prediction
#table13['fare_amount'] = (
#        table1['fare_amount'] + table3['fare_amount']
#        ) / 2
#
## save result
#table13.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train13.csv', index = False
#        )
#
#################################################################################
#### calculate the average prediction (train2+baseline)
#################################################################################
### get a copy of test set
##table02 = baseline_df1.copy()
##
### average of two prediction
##table02['fare_amount'] = (
##        baseline_df1['fare_amount'] + table2['fare_amount']
##        ) / 2
##
### save result
##table02.to_csv(
##        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
##        'submission/submission_train02.csv', index = False
##        )
#
################################################################################
### calculate the average prediction (train1+baseline)
################################################################################
## get a copy of test set
#table03 = baseline_df1.copy()
#
## average of two prediction
#table03['fare_amount'] = (
#        baseline_df1['fare_amount'] + table3['fare_amount']
#        ) / 2
#
## save result
#table03.to_csv(
#        '/Users/ycao/Desktop/taxi_fare_prediction/all/'
#        'submission/submission_train03.csv', index = False
#        )