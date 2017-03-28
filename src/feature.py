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

	product_info_path = '../product_data/product_info.txt'
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
	nn_pair = {}
	
	for id_1 in product_not_exist[0:2]:
			
			dists = []
			for id_2 in product_pool:
				dist = np.sqrt( np.sum( (cluster_data.loc[id_1 - 1][1:cluster_data.shape[1]] - cluster_data.loc[id_2 - 1][1:cluster_data.shape[1]])**2 ) )
				dists.append([id_1,id_2,dist])
			params = sorted(dists,key=lambda d:d[2],reverse=False)
			print 'Finding ',id_1,' Best partner:(',params[0][0],params[0][1],')'
			nn_pair[params[0][0]] = params[0][1]
	print len(nn_pair),nn_pair
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
		predict_data = predict_data.append(partner)
	save_data(predict_data,'../submission.csv')
	pass
