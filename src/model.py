"""
	desc: using gbdt(xgboost) to make classifier and using linear regression to find weights
	author: zhpmatrix@datarush
	date: 2017-03-27
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_blobs,load_boston
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import linear_model

def _xgbc(X,y):
	'''
		desc: using GridSearchCV to get best params of model
	'''
	_params={'max_depth':[3,4],
		 'learning_rate':[0.1,0.2],
		 'n_estimators':[100,120],
		 'min_child_weight':[4,5]
		}
	xgbc = xgb.XGBClassifier(seed=1)
	gs = GridSearchCV(xgbc,param_grid=_params,scoring='roc_auc',n_jobs=2,cv=5,verbose=True)
	gs.fit(X,y)
	return gs

def xgbc(X,y):
	X_train,X_validation,y_train,y_validation = train_test_split(X,y,random_state=0)
	xgbc_boost = xgb.XGBClassifier(seed=1)
	xgbc_boost.fit(X_train,y_train.ravel())
	#print 'training error:',1.0 - xgbc_boost.score(X_train,y_train)
	#print 'validation error:',1.0 - xgbc_boost.score(X_validation,y_validation)
	return xgbc_boost


def get_training_sample(type='classifier'):
	if type == 'classifier':
		X, y = make_blobs(n_samples=10000, n_features=10, centers=2, random_state=0)
	else:
		boston = load_boston()
		X = boston.data
		y = boston.target
	return X,y

def get_predict_sample(type='classifier'):
	if type == 'classifier':
		_X = np.array([[ 6.96957981,  1.81516411,  1.59550583,  9.41886896, -8.72573898,-7.37707708, -7.99193605,  6.61645677,  4.98371677,  6.55234379],[ 1.16528608,  5.25931666,  1.78076368,  3.66390691, -1.11218459,2.51443522, -0.87689402,  8.20462526,  8.4568143 , -2.37316877]])
	else:
		_X = np.array([[6.32000000e-03,   1.80000000e+01,   2.31000000e+00,0.00000000e+00,   5.38000000e-01,   6.57500000e+00,6.52000000e+01,   4.09000000e+00,   1.00000000e+00,2.96000000e+02,   1.53000000e+01,   3.96900000e+02,4.98000000e+00]])
	return _X

if __name__ == '__main__':

	# classifier(XGBOOST)	
	X,y = get_training_sample()
	print 'predict result:',_xgbc(X,y).predict(get_predict_sample())
	print 'predict result:',xgbc(X,y).predict(get_predict_sample())
	
	# regressor(LR)
	X,y = get_training_sample('regressor')
	print 'predict result:',linear_model.LinearRegression().fit(X,y).predict(get_predict_sample('regressor'))
