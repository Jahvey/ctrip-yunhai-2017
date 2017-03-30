import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from product_not_exist_info import *

def data_watching():
	product_quantity = pd.read_csv('../product_data/product_quantity.txt')
	product_quantity.sort_values(['product_id','product_date'],inplace=True)

	train_day=product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()

	plt.figure()
	train_day.apply(lambda x: sum(x.isnull())).plot(figsize=(12,6))
	plt.show()

	train_day.fillna(method='backfill',axis=1) 
	train_day.fillna(method='ffill',axis=1) 

	plt.figure()
	train_day.sum().plot(figsize=(12,6))
	plt.show()

	product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
	train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()

	train_month.apply(lambda x: sum(x.isnull()))

	plt.figure()
	train_month.sum().plot(figsize=(12,6))
	plt.show()

	plt.figure()
	train_day[train_month.index==1].sum().plot(figsize=(12,6))
	plt.show()

	plt.figure()
	train_month[train_month.index==1].sum().plot(figsize=(12,6))
	plt.show()
	pass
def idx_to_date(data):
	if data == 1:
		data = '2015-12-01'
	if data > 1 and data <= 10:
		data = '2016-0'+str(data - 1)+'-01'
	if data > 10 and data <= 13:
		data = '2016-'+str(data - 1)+'-01'
	if data == 14:
		data = '2017-01-01'
	return data
if __name__ == '__main__':
	
	product_quantity = pd.read_csv('../product_data/product_quantity.txt')
	product_quantity.sort_values(['product_id','product_date'],inplace=True)
	product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])

	start_month = '2014-12'
	end_month = '2015-11'
	month_range = (pd.to_datetime(end_month) - pd.to_datetime(start_month)).days/30 + 1
	product_quantity=product_quantity[product_quantity.product_month>=start_month]

	train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()

	# fill nan with diff methods,fill_label:0,1,2,3,4
	fill_label = '0'
	
	#0
	train_month_impute_value = 140
	train_month.fillna(train_month_impute_value,inplace=True) 
	
	#1
	#train_month.fillna(np.mean(train_month.mean(axis=0)),inplace=True)
	
	#2
	#train_month.fillna(np.mean(train_month.mean(axis=1)),inplace=True)
	
	#3
	#train_month.fillna(train_month.mean(axis=0),inplace=True) 
	
	#4
	#train_month.fillna(train_month.mean(axis=1),inplace=True) 
	

	#plt.figure()
	#train_month.sum().plot(figsize=(10,6))
	#plt.show()

	average_all=pd.DataFrame(train_month.mean(axis=1),columns=['average_all']).reset_index()
	
	# find nearest neighbor for product without info
	for key in product_nn_pair:
		idx = average_all['product_id'][average_all.product_id == product_nn_pair[key]].index
		tmp = average_all.ix[idx].copy()
		tmp.ix[idx,'product_id'] = key
		average_all = average_all.append(tmp,ignore_index=True)
	submission=pd.read_csv('../product_data/prediction_lilei_20170320.txt')

	col=['product_id','product_month','ciiquantity_month']

	submission.columns=col

	out_month_impute_value = 132
	out=pd.merge(submission,average_all,on='product_id',how='left').fillna(out_month_impute_value)

	out.apply(lambda x: sum(x.isnull()))

	out.ciiquantity_month=out.average_all

	out.drop(['average_all'],axis=1,inplace=True)

	# set zero for some product overdate
	for key in product_idx_pair:
		cq_idx  = idx_to_date(product_idx_pair[key])
		idxs = out['product_id'][out.product_id == key].index
		for idx in idxs:
			if out.ix[idx,'product_month'] < cq_idx:
				out.ix[idx,'ciiquantity_month'] = 0
	# data to csv
	out.to_csv('rule_prediction/'+str(month_range)+'_fill_'+fill_label+'_'+str(out_month_impute_value)+'.txt',index=False)
