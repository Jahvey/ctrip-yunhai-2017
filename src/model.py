import numpy as np
import pandas as pd
import xgboost as xgb
from _feature import *
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def xgbr(X,y):
	'''
		desc: using GridSearchCV to get best params of model
	'''
	_params={'max_depth':[3,10],
		 'learning_rate':[0.1],
		 'n_estimators':[100,150],
		 'min_child_weight':[1]
		}
	xgbr = xgb.XGBRegressor(seed=1)
	gs = GridSearchCV(xgbr,param_grid=_params,n_jobs=1,cv=5,verbose=True)
	gs.fit(X,y)
	return gs

def get_training_sample(train_filePath):
	label1_0 = pd.read_csv(train_filePath)
	X = label1_0.ix[:,'mean4':'max5'].as_matrix()
	y = label1_0.ix[:,'ciiquantity'].as_matrix()
	return X,y

#def get_predict_sample(predict_filePath):
#	label1_0 = pd.read_csv(predict_filePath)
#	_X = label1_0.ix[:,'mean4':'min6'].as_matrix()
#	return _X

def get_predict_sample(product_id,model):
    params = iniParams()

    #load data from inputpath
    params['inputPath'] = "../training_data/train_data_for_model/lagrange/"

    dfData = get_data(params,product_id)
    dateRange = pd.date_range('2015-10','2017-02',freq='M')
    for insertDate in dateRange:

        dfData.ix[insertDate,['ciiquantity']] = 0
        dfFeatureData = get_statics(dfData)
        dfFeatureData =  dfFeatureData.ix[-1,:-1].as_matrix()
        dfFeatureDataMatrix = dfFeatureData.reshape(1,len(dfFeatureData))
        pre =  model.predict(dfFeatureDataMatrix)
        dfData.ix[insertDate,['ciiquantity']] = pre
    return dfData

if __name__ == '__main__':
	
	train_filePath = '../training_data/train_data_for_model/label/label1_0.csv'
	predict_filePath = '../training_data/train_data_for_model/label/label1_0.csv'
	X,y = get_training_sample(train_filePath)
	
	#X_impute_value = 40
	#pd.DataFrame(X).fillna(X_impute_value,inplace=True)
	#lr = linear_model.LinearRegression().fit(X,y)
	
	#gs =xgbr(X,y)
	#joblib.dump(gs,'gs.dmp')
	#svr = SVR(verbose = True)
	#svr.fit(X,y)
	#exit()
	gs = joblib.load('gs.dmp')
	pre = get_predict_sample(1,gs)

	#_X = get_predict_sample(predict_filePath)
	#pre = pd.DataFrame( np.array(gs.predict(_X).reshape(-1,1)) )
	plt.figure()
	plt.plot(pre,color='green')
	plt.show()
	plt.close()
