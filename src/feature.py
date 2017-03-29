import time
import pandas as pd
import numpy as np
from product_not_exist_info import *
from script import *

def get_month_ratio():
	month_dirPath = '../training_data/month/'
	startMonth = '2014-01'
	endMonth = '2015-12'
	months_range = pd.date_range(startMonth,endMonth,freq='M')
	month_list = [str(month)[:7] for month in months_range]
	cols_name = ['startdate','upgradedate','cooperatedate']
	month_ratio = []
	for month in month_list:
		filePath = month_dirPath + month + '.csv'
		data = pd.read_csv(filePath)
		data_len = len(data)
		col_ratio = []
		for col in cols_name:
			missing_val_num = len(data[col][data[col] == '-1'])
			outlier_val_num = len(data[col][data[col] == '1753-01-01'])
			col_ratio.append( float(missing_val_num) /data_len )
			col_ratio.append( float(outlier_val_num) /data_len )
		month_ratio.append(col_ratio)
	print month_ratio

def set_zero(product_info_file_path):
	product = pd.read_csv(product_info_file_path)
	idx_pair = {}
	data = product[['product_id','startdate']][product.startdate > '2015-11-30'].copy()
	for i in range(0,data.shape[0]):
		 if data['startdate'].tolist()[i][0:4] == '2015':
			 idx_pair[data['product_id'].tolist()[i]] = 1
		 if data['startdate'].tolist()[i][0:4] == '2016':
			 idx_pair[data['product_id'].tolist()[i]] = int(data['startdate'].tolist()[i][5:7]) + 1
	return idx_pair	


def data_to_rank(data,label):
	if label == 'district_id1':
		if  data >=6500 and data < 13000:
			data = 0
		if data >= 13000 and data < 19500:
			data = 1
		if data >= 19500 and data < 26000:
			data = 2
		else:
			data = 3
	elif label == 'district_id2':
		if data >= 0 and data < 700000:
			data = 0
		if data >= 700000 and data < 1050000:
			data = 1
		if  data >= 1050000 and data < 1400000:
			data = 2
		else:
			data = 3

	elif label == 'district_id3':
		if data >= 0 and data < 1250000:
			data = 0
		if data >= 1250000 and data < 2500000:
			data = 1
		if  data >= 2500000 and data < 3750000:
			data = 2
		if data >= 3750000 and data < 5000000:
			data = 3
		else:
			data = 4
	else:
		
		if data >= 0 and data < 400000:
			data = 0
		if data >= 400000 and data < 800000:
			data = 1
		if  data >= 800000 and data < 1200000:
			data = 2
		if data >= 1200000 and data < 1600000:
			data = 3
		else:
			data = 4
	return data

def cooperate_date_imputer(data):
	if data == '-1':
		data = '2014-01-01'
	return data

def get_month_from_date(data):
	return int(data[5:7])

def start_date_imputer(data):
	for i in range(0,data.shape[0]):
		if data['startdate'][i] == '-1' or data['startdate'][i] == '1753-01-01':
			data['startdate'][i] = data['cooperatedate'][i]
	return data

if __name__ == '__main__':
	
	ordered_submit_file_path = '../ordered_submission.csv'
	ordered_data = pd.read_csv(ordered_submit_file_path)	
	format_data = pd.DataFrame()
	for i in range(1,14 + 1):
		month_data = pd.DataFrame()
		month_data['product_id'] = ordered_data['0']
		if i == 1:
			product_month = '2015-12-01'
		elif i > 1 and i <= 10:
			product_month = '2016-0'+str(i-1)+'-01'
		elif i > 10 and i < 14:
			product_month = '2016-'+str(i-1)+'-01'
		else:
			product_month = '2017-01-01'

		month_data['product_month'] = product_month
		month_data['ciiquantity_month'] = ordered_data[str(i)]
		format_data = format_data.append(month_data,ignore_index=True)
	save_data(format_data,'../prediction_zhpmatrix_'+time.strftime('%Y%m%d',time.localtime(time.time()))+'.txt')
	exit()
	product_info_path = '../product_data/product_info.txt'
	
	# ordered submission and set 0 to some product
	idx_pair = set_zero(product_info_path)
	submit_data = pd.read_csv('../submission.csv')	
	for key in idx_pair:
		product_idx = submit_data['0'][submit_data['0'] == key].index[0]
		submit_data.loc[product_idx,[str(x) for x in range(1,idx_pair[key])]] = 0
	submit_data = submit_data.sort(columns='0')
	save_data(submit_data,ordered_submit_file_path)
	exit()
	
	data = pd.read_csv(product_info_path)
	district_set = ['district_id1','district_id2','district_id3','district_id4']
	for col in district_set:
		data.loc[:,col] = map(lambda x:data_to_rank(x,col),data[col])
	
	data.loc[:,'cooperatedate'] = map(lambda x:cooperate_date_imputer(x),data['cooperatedate'])
	data = start_date_imputer(data)
	date_set = ['startdate','cooperatedate']
	for dat in date_set:
		data.loc[:,dat] = map(lambda x:get_month_from_date(x),data[dat])
	cluster_cols = ['product_id','district_id1','district_id2','district_id3','district_id4','lat','lon','eval','eval2','eval3','eval4','voters','startdate','cooperatedate','maxstock']	
	cluster_data = data[cluster_cols]

	dummy_cols = ['district_id1','district_id2','district_id3','district_id4','eval','eval2','startdate','cooperatedate']	
	cluster_data = pd.get_dummies(cluster_data,columns=dummy_cols)
	
	s_product_not_exist = set(product_not_exist)
	s_product_id = set(cluster_data['product_id'])
	product_pool = list(s_product_id - s_product_not_exist)
#	nn_pair = {}
#	
#	counter = 0	
#	for id_1 in product_not_exist:
#			
#			dists = []
#			counter += 1
#			for id_2 in product_pool:
#				dist = np.sqrt( np.sum( (cluster_data.loc[id_1 - 1][1:cluster_data.shape[1]] - cluster_data.loc[id_2 - 1][1:cluster_data.shape[1]])**2 ) )
#				dists.append([id_1,id_2,dist])
#			params = sorted(dists,key=lambda d:d[2],reverse=False)
#			print 'Over/Total','(',counter,'/505)','Finding ',id_1,' Best partner:(',params[0][0],params[0][1],')'
#			nn_pair[params[0][0]] = params[0][1]
#	print len(nn_pair),nn_pair
	print nn_pair	
	predict_dirPath = '../predict_data/'
	predict_data = pd.DataFrame()
	cq_min_num = 1
	cq_max_num = 23
	for i in range(cq_min_num,cq_max_num + 1):
		cq_data = pd.read_csv(predict_dirPath+str(i)+'.csv')
		cols = range(cq_data.shape[1] - 14,cq_data.shape[1])
		cols.insert(0,0)
		tmp = cq_data.iloc[:,cols]
		tmp.columns = range(14+1)
		predict_data = predict_data.append(tmp,ignore_index=True)
	
	for key in nn_pair:
		index = predict_data[0][predict_data[0] == nn_pair[key]].index
		partner = predict_data.loc[index].copy()
		partner[0] = key
		predict_data = predict_data.append(partner,ignore_index=True)
	save_data(predict_data,'../submission.csv')
