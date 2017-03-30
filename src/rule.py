import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from product_not_exist_info import *

def data_watching():
	#product_quantity = pd.read_csv('../product_data/product_quantity.txt')
	#product_quantity.sort_values(['product_id','product_date'],inplace=True)

	#train_day=product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()

	#plt.figure()
	#train_day.apply(lambda x: sum(x.isnull())).plot(figsize=(12,6))
	#plt.show()

	#train_day.fillna(method='backfill',axis=1) 
	#train_day.fillna(method='ffill',axis=1) 

	#plt.figure()
	#train_day.sum().plot(figsize=(12,6))
	#plt.show()

	#product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
	#train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()

	#train_month.apply(lambda x: sum(x.isnull()))

	#plt.figure()
	#train_month.sum().plot(figsize=(12,6))
	#plt.show()

	#plt.figure()
	#train_day[train_month.index==1].sum().plot(figsize=(12,6))
	#plt.show()

	#plt.figure()
	#train_month[train_month.index==1].sum().plot(figsize=(12,6))
	#plt.show()
	pass

if __name__ == '__main__':
	
	product_quantity = pd.read_csv('../product_data/product_quantity.txt')
	product_quantity.sort_values(['product_id','product_date'],inplace=True)
	product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])

	start_month = '2015-03'
	end_month = '2015-11'
	month_range = (pd.to_datetime(end_month) - pd.to_datetime(start_month)).days/30 + 1
	product_quantity=product_quantity[product_quantity.product_month>=start_month]

	train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()

	train_month_impute_value = 140
	train_month.fillna(train_month_impute_value,inplace=True) 

	#plt.figure()
	#train_month.sum().plot(figsize=(10,6))
	#plt.show()

	average_all=pd.DataFrame(train_month.mean(axis=1),columns=['average_all']).reset_index()

	submission=pd.read_csv('../product_data/prediction_lilei_20170320.txt')

	col=['product_id','product_month','ciiquantity_month']

	submission.columns=col

	out_month_impute_value = 132
	out=pd.merge(submission,average_all,on='product_id',how='left').fillna(out_month_impute_value)

	out.apply(lambda x: sum(x.isnull()))

	out.ciiquantity_month=out.average_all

	out.drop(['average_all'],axis=1,inplace=True)

	# find nearest neighbor for product without info
	# and
	# set zero for some product overdate

	








	# data to csv
	out.to_csv('rule_prediction/'+str(month_range)+'_'+str(train_month_impute_value)+'_'+str(out_month_impute_value)+'.txt',index=False)
