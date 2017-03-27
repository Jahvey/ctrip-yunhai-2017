import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

productQuantityPath = "product_data/product_quantity.txt"
outputPath = "train_data/month/"
#
#load data
dfQuantity = pd.read_csv(productQuantityPath,index_col='product_date',parse_dates=True,
                         usecols=['product_id', 'product_date', 'ciiquantity'])
dfTemp = dfQuantity.copy()
IDnum = 4000
lengthID = []
for ID in range(1,IDnum+1):
    dfID = dfTemp[dfTemp['product_id']==ID].sort_index().resample('M',how=sum,kind='period').ix[:,['ciiquantity']]
    lengthID.append(len(dfID))
    dfID.to_csv(outputPath+str(ID)+'.csv')

countDict = Counter(lengthID)
# print countDict
# print dfQuantity[dfQuantity['product_id']==1].resample('M',how=sum,kind='period')