import pandas as pd

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
	cluster_cols = ['district_id1','district_id2','district_id3','district_id4','lat','lon','eval','eval2','eval3','eval4','voters','startdate','cooperatedate','maxstock']	
	cluster_data = data[cluster_cols]

	dummy_cols = ['district_id1','district_id2','district_id3','district_id4','eval','eval2','startdate','cooperatedate']	
	cluster_data = pd.get_dummies(cluster_data,columns=dummy_cols)
	pass
