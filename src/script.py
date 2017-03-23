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

def gen_training_data(c_pq,c_pi,group_index,target_label,dirPath):
	grouped_pq = c_pq.groupby(group_index)['ciiquantity'].agg(np.sum).reset_index()
    	counter_grouped = grouped_pq.groupby(group_index[0])[group_index[1]]
    	print counter_grouped.count()
	col_grouped = grouped_pq.groupby(group_index[0])
	for key,items in col_grouped:
		if target_label == 'product':
			filePath = dirPath+str(key)
			dropped_items = items.drop(group_index[0],axis=1)
			training_data = dropped_items			
		else:
			filePath = dirPath+str(key)
			dropped_items = items.drop(group_index[0],axis=1)
			training_data = pd.merge(c_pi,dropped_items)
		save_data(training_data,filePath)

def save_data(data,filePath):
	f = open(filePath,'w')
	data.to_csv(f,index=False)
	f.close()

def product_fig(inPath,outPath):
	product_not_exist = []
	product_start_idx = 1
	product_end_idx = 4000
	for i in range(product_start_idx,product_end_idx+1):
		filePath = inPath+str(i)
		if os.path.exists(filePath):
			print i
			product_data = pd.read_csv(inPath+str(i))
			plt.plot(pd.to_datetime(product_data['product_date']),product_data['ciiquantity'])
			plt.savefig(outPath+str(i))
			plt.close()
		else:
			print 'product {} isn\'t exist! '.format(i)
			product_not_exist.append(i)
	return product_not_exist
		
month_dirPath = '../training_data/month/'
product_dirPath = '../training_data/product/'
product_figPath = '../training_data/images/month_quantity/'

pq = pd.read_csv('../product_data/product_quantity.txt')
pi = pd.read_csv('../product_data/product_info.txt')

c_pq = pq.copy()
c_pq.set_index(['product_id','product_date'])
c_pi = pi.copy()
c_pq['product_date'] = c_pq['product_date'].apply(lambda x:x[:7])

gen_training_data(c_pq,c_pi,['product_id','product_date'],'product',product_dirPath)

product_not_exist = product_fig(product_dirPath,product_figPath)
print 'product_not_exist: ',product_not_exist,'\nnumber of product not exist: ',len(product_not_exist)
