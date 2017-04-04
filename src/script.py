import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from product_not_exist_info import *

def month_mean(data,window,valid_month_num):
	data = np.array(data)
	valid_month = valid_month_num
	sample_num = data.shape[0]
	try:
		weight = np.array([[1.0/window]] * window)
	except ZeroDivisionError:
		print '[TRAIN_ERR]:window len = 0!'
	test_data = data[:,0:data.shape[1]-valid_month]
	errors = []
	for i in range(0,valid_month):
		# index=0 indicates the column product_id
		if test_data.shape[1] - window == 0:
			raise ValueError('[TRAIN_ERROR]:product_id can\'t be used!')
		cur_cq = np.dot(test_data[:,test_data.shape[1] - window:test_data.shape[1]],weight)
		test_data = np.column_stack((test_data,cur_cq))
	try:
		valid_err = np.sqrt( np.sum( (data-test_data)**2 ) / (valid_month*sample_num) )
	except ZeroDivisionError:
		print'[TRAIN_ERROR]:sample_num = 0!'
	return valid_err

def mean_fit(data,window_max_len,valid_month_num):
	window_min = 2
	window_max = window_max_len
	errors = []
	for window in range(window_min,window_max + 1):
		error = month_mean(data,window,valid_month_num)
		print 'mean:',' window=',window,'error=',error
		errors.append([window,error])
	params = sorted(errors,key=lambda err:err[1],reverse=False)
	
	best_params = {}
	best_params['best_window'] = params[0][0]
	best_params['min_err'] = params[0][1]
	return best_params

def mean_predict(data,window,filePath,cq_num):

	data = np.array(data)
	try:
		weight = np.array([[1.0/window]] * window)
	except ZeroDivisionError:
		print '[PREDICT_ERR]:window len = 0!'
	month_num = 14
	for i in range(0,month_num):
		# index=0 indicates the column product_id
		if data.shape[1] - window == 0:
			raise ValueError('[PREDICT_ERR]:product_id can\'t be used!')
		cur_cq = np.dot(data[:,data.shape[1] - window : data.shape[1]],weight)
		data = np.column_stack((data,cur_cq))
	saveData(data,filePath+str(cq_num)+'.csv')

def month_percent(data,window,percent,valid_month_num):
	data = np.array(data)
	sample_num = data.shape[0]
	valid_month = valid_month_num
	test_data = data[:,0:data.shape[1]-valid_month]
	errors = []
	for i in range(0,valid_month):
		# index=0 indicates the column product_id
		if test_data.shape[1] - window == 0:
			raise ValueError('[TRAIN_ERROR]:product_id can\'t be used!')
		train_data = test_data[:,test_data.shape[1] - window:test_data.shape[1]]
		cur_cq = np.array([np.percentile(sample,percent) for sample in train_data]).reshape(-1,1)		
		test_data = np.column_stack((test_data,cur_cq))
	try:
		valid_err = np.sqrt( np.sum( (data-test_data)**2 ) / (valid_month*sample_num) )
	except ZeroDivisionError:
		print'[TRAIN_ERROR]:sample_num = 0!'
	return valid_err

def percent_fit(data,window_max_len,valid_month_num):
	window_min = 2
	window_max = window_max_len
	
	percent_min = 10
	percent_max = 100
	percent_stepsize = 5
	
	errors = []
	for window in range(window_min,window_max + 1):
		for percent in range(percent_min,percent_max,percent_stepsize):
			error = month_percent(data,window,percent,valid_month_num)
			print 'percent:',' window=',window,'percent=',percent,'error=',error
			errors.append([window,percent,error])
	params = sorted(errors,key=lambda err:err[2],reverse=False)
	
	best_params = {}
	best_params['best_window'] = params[0][0]
	best_params['best_percent'] = params[0][1]
	best_params['min_err']= params[0][2]
	return best_params

def percent_predict(data,window,percent,filePath,cq_num):
	data = np.array(data)
	month_num = 14
	for i in range(0,month_num):
		# index=0 indicates the column product_id
		if data.shape[1] - window == 0:
			raise ValueError('[PREDICT_ERR]:product_id can\'t be used!')
		train_data = data[:,data.shape[1] - window:data.shape[1]]
		cur_cq = np.array([np.percentile(sample,percent) for sample in train_data]).reshape(-1,1)		
		data = np.column_stack((data,cur_cq))
	saveData(data,filePath+str(cq_num)+'.csv')

	
def magic_box(data,window_max_len,valid_month_num,predict_dirPath,cq_num):
	if cq_num >=1 and cq_num < 3:
		print 'cq_num=',cq_num,'using mean method!','\n'
		print '-' * 80
		mean_predict(data,np.array(data).shape[1] - 1,predict_dirPath,cq_num)
	else:
		if cq_num >=3 and cq_num < 11:
			window_max_len = 2
			valid_month_num = 1
		
		percent_best_params= percent_fit(data,window_max_len,valid_month_num)
		mean_best_params = mean_fit(data,window_max_len,valid_month_num)
		if percent_best_params['min_err'] < mean_best_params['min_err']:
			print 'cq_num=',cq_num,'best_params:',percent_best_params,'\n'
			print '-' * 80
			percent_predict(data,percent_best_params['best_window'],percent_best_params['best_percent'],predict_dirPath,cq_num)
		else:
			print 'cq_num=',cq_num,'best_params:',mean_best_params,'\n'
			print '-' * 80
			mean_predict(data,mean_best_params['best_window'],predict_dirPath,cq_num)

def gen_training_data(c_pq,c_pi,group_index,target_label,dirPath):
	grouped_pq = c_pq.groupby(group_index)['ciiquantity'].agg(np.sum).reset_index()
    	counter_grouped = grouped_pq.groupby(group_index[0])[group_index[1]]
	#print counter_grouped.count()
	product_set = get_product_set(counter_grouped,1,23)
	# plot missing month ciiquantity for each product
	# plt.show(counter_grouped.count().plot(kind='bar'))
	# exit()
	col_grouped = grouped_pq.groupby(group_index[0])
	for key,items in col_grouped:
		filePath = dirPath+str(key)+'.csv'
		if target_label == 'product':
			dropped_items = items.drop(group_index[0],axis=1)
			training_data = dropped_items			
		else:
			dropped_items = items.drop(group_index[0],axis=1)
			training_data = pd.merge(c_pi,dropped_items)
		save_data(training_data,filePath)
	return product_set
def exit_wrapper(func):
	def wrapper(*args,**kwargs):
		func(*args,**kwargs)
		exit()
	return wrapper

@exit_wrapper
def show_grouped_data(grouped_data):
	for key,items in grouped_data:
		print key
		for subitem in items:
			print subitem
		print '-' * 30

def get_product_set(data_grouped,cq_min_len,cq_max_len):
	product_set = {}	
	cq_min_len = 1
	cq_max_len = 23
	product_set = {}
	for i in range(cq_min_len,cq_max_len+1):
		product_set[i] = [key for key,items in data_grouped if len(items) == i]
	#print product_set		
	return product_set

def saveData(data,filePath):
	f = open(filePath,'w')
	# get down integer(chinglish!!)
	data = pd.DataFrame(data.astype('int'))
	data.to_csv(f,index=False)
	f.close()

def save_data(data,filePath):
	f = open(filePath,'w')
	data.to_csv(f,index=False)
	f.close()

def product_fig(inPath,outPath):
	product_not_exist = []
	product_start_idx = 1
	product_end_idx = 4000
	for i in range(product_start_idx,product_end_idx+1):
		filePath = inPath+str(i)+'.csv'
		if os.path.exists(filePath):
			print i
			product_data = pd.read_csv(filePath)
			# set product_date as datetime index
			product_data['product_date'] = pd.to_datetime(product_data['product_date'])
			date_index_product_data = product_data.set_index('product_date')
			# plot ciiquantity
			plt.figure()
			date_index_product_data.ciiquantity.plot(x_compat=True)
			plt.savefig(outPath+str(i))
			plt.close()
		else:
			print 'product {} isn\'t exist! '.format(i)
			product_not_exist.append(i)
	return product_not_exist

if __name__ == '__main__':		

	cq_min_len = 1
	cq_max_len = 23
	
	window_max_len = 5
	valid_month_num = 4
	
	
	month_dirPath = '../training_data/month/'
	
	product_dirPath = '../training_data/23_month_product/'
	predict_dirPath = '../23_predict_data/'
	
	product_figPath = '../training_data/images/month_quantity/'
	pq = pd.read_csv('../product_data/product_quantity.txt')
	pi = pd.read_csv('../product_data/product_info.txt')

	c_pq = pq.copy()
	c_pq.set_index(['product_id','product_date'])
	c_pi = pi.copy()
	c_pq['product_date'] = c_pq['product_date'].apply(lambda x:x[:7])
	
	#product_set = gen_training_data(c_pq,c_pi,['product_id','product_date'],'product',product_dirPath)
	
#	for i in range(cq_min_len,cq_max_len + 1):
#		data = []
#		for name in product_set[i]:
#			tmp = pd.read_csv(product_dirPath+str(name)+'.csv')
#			# get cq values
#			row = tmp.loc[:,'ciiquantity'].tolist()
#			# insert product_id
#			row.insert(0,name)
#			# add id+cq into data
#			data.append(row)
#		magic_box(data,window_max_len,valid_month_num,predict_dirPath,i)
	
	# using filling data and the same cq num as 23
	i = 23
	data = []
	for name in product_set[i]:
		tmp = pd.read_csv(product_dirPath+str(name)+'.csv')
		# get cq values
		row = tmp.loc[:,'ciiquantity'].tolist()
		# insert product_id
		row.insert(0,name)
		# add id+cq into data
		data.append(row)
	magic_box(data,window_max_len,valid_month_num,predict_dirPath,i)


