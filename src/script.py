import pandas as pd
import numpy as np

def saveData(data,filePath):
	f = open(filePath,'w')
	data.to_csv(f,index=False)
	f.close()

dirPath = '../training_data/'
pq = pd.read_csv('../product_data/product_quantity.txt')
pi = pd.read_csv('../product_data/product_info.txt')

c_pq = pq.copy()
c_pq.set_index(['product_id','product_date'])
c_pi = pi.copy()

c_pq['product_date'] = c_pq['product_date'].apply(lambda x:x[:7])
month_id_grouped_pq = c_pq.groupby(['product_date','product_id'])

new_month_id_grouped_pq = month_id_grouped_pq['ciiquantity'].agg(np.sum).reset_index()

# get number of product for each month
product_grouped = new_month_id_grouped_pq.groupby('product_date')['product_id']
#print product_grouped.count()

month_grouped = new_month_id_grouped_pq.groupby('product_date')
for key,items in month_grouped:
	dropped_items = items.drop('product_date',axis=1)
	month_training_data = pd.merge(c_pi,dropped_items)
	filePath = dirPath+key
	saveData(month_training_data,filePath)
