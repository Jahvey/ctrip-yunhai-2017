import copy
from script import *
import pandas as pd
from collections import OrderedDict

data = pd.read_csv('../training_data/23_month_product.csv')
training_dirPath = '../training_data/23_month_product/'
cols = copy.copy(data.columns.tolist())
cols.remove('product_id')
ids = copy.copy(data['product_id'].tolist())
for _id in ids:
	row = data[data.product_id == _id]
	product = OrderedDict()
	product['product_date'] = cols
	product['ciiquantity'] = row.ix[:,1:row.shape[1]].as_matrix()[0].tolist()
	_product = copy.copy(pd.DataFrame(product))
	save_data(_product,training_dirPath+str(_id)+'.csv')

