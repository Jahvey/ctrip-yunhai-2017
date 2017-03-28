import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
	data = pd.DataFrame(data)
	data.to_csv(f,index=False)
	f.close()

def mean_fit(data):
	window_max = 4
	errors = []
	for window in range(2,window_max):
		error = month_mean(data,window)
		errors.append([window,error])
	params = sorted(errors,key=lambda err:err[1],reverse=False)
	
	best_window = params[0][0]
	return best_window

def mean_predict(data,window,filePath,cq_num):
	data = np.array(data)
	weight = np.array([[1.0/window]] * window)
   	month_num = 14
	for i in range(0,month_num):
		cur_cq = np.dot(data[:,data.shape[1] - window : data.shape[1]],weight)
		data = np.column_stack((data,cur_cq))
	saveData(data,filePath+str(cq_num)+'.csv')

def month_mean(data,window):
	data = np.array(data)
	sample_num = data.shape[0]
	weight = np.array([[1.0/window]] * window)
	valid_month = 3
	test_data = data[:,0:data.shape[1]-valid_month]
	errors = []
	for i in range(0,valid_month):
		cur_cq = np.dot(test_data[:,test_data.shape[1] - window:test_data.shape[1]],weight)
		
		test_data = np.column_stack((test_data,cur_cq))
	return np.sqrt( np.sum(data-test_data)**2 / (valid_month*sample_num) )

def month_percent(data,window=3,percent=0.6):
	pass

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
	
	cq_min_len = 22
	cq_max_len = 23

	month_dirPath = '../training_data/month/'
	product_dirPath = '../training_data/product/'
	product_figPath = '../training_data/images/month_quantity/'
	predict_dirPath = '../predict_data/'
	pq = pd.read_csv('../product_data/product_quantity.txt')
	pi = pd.read_csv('../product_data/product_info.txt')

	c_pq = pq.copy()
	c_pq.set_index(['product_id','product_date'])
	c_pi = pi.copy()
	c_pq['product_date'] = c_pq['product_date'].apply(lambda x:x[:7])
	product_set = gen_training_data(c_pq,c_pi,['product_id','product_date'],'product',product_dirPath)
	for i in range(cq_min_len,cq_max_len + 1):
		data = []
		for name in product_set[i]:
			tmp = pd.read_csv(product_dirPath+str(name)+'.csv')
			# get cq values
			row = tmp.loc[:,'ciiquantity'].tolist()
			# insert product_id
			row.insert(0,name)
			# add id+cq into data
			data.append(row)
		best_window = mean_fit(data)
		mean_predict(data,best_window,predict_dirPath,i)
		exit()
	

