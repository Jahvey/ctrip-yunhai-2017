import pandas as pd
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
