import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

def iniParams():
    '''
    init the class Params to save the origial params
    return: class Params have those params
    '''
    params = {}
    params['IDNum'] = 1
    #data propress from startTime to endTime
    params['startTime'] = '2014-01'
    params['endTime'] = '2015-09'
    #load data from inputpath
    params['inputPath'] = "../training_data/train_data_for_model/lagrange/"
    #save data in this path
    params['outputPath'] = "../training_data/train_data_for_model/genFeature/"
    return params

def inipredictParams():
    '''
    init the class Params to save the origial params
    return: class Params have those params
    '''
    params = {}
    params['IDNum'] = 1
    #data propress from startTime to endTime
    params['startTime'] = '2015-09'
    params['endTime'] = '2017-01'
    #load data from inputpath
    params['inputPath'] = ""
    #save data in this path
    params['outputPath'] = ""
    return params

def get_data(params,ID):
    """get the date from start to end"""
    start = params['startTime']
    end = params['endTime']
    inputPath = params['inputPath'] + str(ID)+'.csv'
    dftemp = pd.read_csv(inputPath,index_col='product_date',usecols=['product_date','ciiquantity'],
                         parse_dates=True,na_values=['nan'])
    df = dftemp.ix[start:end,:]
    return df

def getTest_data(params,ID):
    """get the date from start to end"""
    start = params['startTime']
    end = params['endTime']
    inputPath = params['inputPath'] + str(ID)+'.csv'
    dftemp = pd.read_csv(inputPath,index_col='product_date',usecols=['product_date','ciiquantity'],
                         parse_dates=True,na_values=['nan'])
    df = dftemp.ix[start:end,:]
    return df

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values.shift(1), window=window)

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values.shift(1), window=window)

def get_rolling_median(values,window):
    """Return rolling median of given values, using specified window size."""
    return pd.rolling_median(values.shift(1),window=window)

def get_rolling_min(values,window):
    """Return rolling median of given values, using specified window size."""
    return pd.rolling_min(values.shift(1),window=window)

def get_rolling_max(values,window):
    """Return rolling median of given values, using specified window size."""
    return pd.rolling_max(values.shift(1),window=window)

def get_max_value(values):
    """Return max of given values, using specified window size."""
    return values.shift(1).max()

def get_min_value(values):
    """Return min of given values, using specified window size."""
    return values.shift(1).min()

def get_rolling_skew(values,window):
    """Return rolling skew of given values, using specified window size."""
    return pd.rolling_skew(values.shift(1),window=window)

def get_rolling_kurt(values,window):
    """Return rolling kurt of given values, using specified window size."""
    return pd.rolling_kurt(values.shift(1),window=window)

def get_statics(df):
    Features = {}
    # param a
    for i in range(3,7):
        Features['mean'+str(i)] = get_rolling_mean(df,i)
        Features['max'+str(i)] = get_rolling_max(df,i)
        Features['median'+str(i)] = get_rolling_median(df,i)
        Features['min' + str(i)] = get_rolling_min(df,i)
    #param b
    for i in range(2,7):
         for j in range(2,7):
             Features['decline'+str(i)+'_'+str(j)] = get_rolling_mean(df,i).div(get_rolling_min(df,j))-1
             Features['decline'+str(i)+'__'+str(j)] = get_rolling_mean(df,i).div(get_rolling_max(df,j))-1
    #param c
    Features['cummax'] = df.shift(1).cummax()
    Features['cummin'] = df.shift(1).cummin()
    Features['increase'] = df - df.ix[0,:]
    #param d
    for i in range(3,7):
        Features['std'+str(i)] = get_rolling_std(df,i)
        for j in range(3,7):
             Features['std2mean'+str(i)+str(j)] = get_rolling_std(df,i).div(get_rolling_mean(df,j))
             Features['std2median'+str(i)+str(j)] = get_rolling_std(df,i).div(get_rolling_median(df,j))
    #param e
    Features['skew'] = get_rolling_skew(df,3)
    Features['kurt'] = get_rolling_kurt(df,3)
    #param divide b1
    #Features['divide'] = get_rolling_mean(df,3).div(get_first_mean(df,3))
    for symbol in Features:
        dftemp = Features.get(symbol)
        df = df.join(dftemp.rename(columns={'ciiquantity':str(symbol)}))
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    return df

def preprocess(params,ID):
    '''
    preprocess to the dataframe had all the feature
    param params:
    param shopID:
    '''
    df = get_data(params, ID)
    dfstatics = get_statics(df)
    dfstatics.to_csv(params['outputPath']+str(ID)+'.csv')
    return dfstatics

if __name__ == "__main__":
    params = iniParams()
    for filename in os.listdir(params['inputPath']):
        if filename == '.DS_Store':pass
        elif os.path.isfile(os.path.join(params['inputPath'],filename)):
            path = os.path.join(params['inputPath']+str(filename))
            #test path
            # path = params['inputPath']+"1.csv"
            # print "id:", filename, ",is processing..."
            ID = filename[:-4]
            print "id:", ID, ",is processing..."
            preprocess(params,ID)
            print "over..."
