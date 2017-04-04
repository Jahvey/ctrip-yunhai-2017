import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")


inputLabelPath = "../training_data/train_data_for_model/divideLabelDataframe.csv"
inputFeaturePath = "../training_data/train_data_for_model/genFeature/"
outputLabelPath = "../training_data/train_data_for_model/label/"

labelData = pd.read_csv(inputLabelPath)
labelData.columns = ['ID','label1','label2','label3']
labelData =  labelData.set_index(['ID']).sort_index()

label1_1 = labelData[labelData['label1']==1].ix[:,['label1']]
label1_0 = labelData[labelData['label1']==0].ix[:,['label1']]

label2_1 = labelData[labelData['label2']==1].ix[:,['label2']]
label2_0 = labelData[labelData['label2']==0].ix[:,['label2']]

label3_1 = labelData[labelData['label3']==1].ix[:,['label3']]
label3_0 = labelData[labelData['label3']==0].ix[:,['label3']]

label1_1Dataframe = pd.DataFrame()
label1_0Dataframe = pd.DataFrame()
label2_1Dataframe = pd.DataFrame()
label2_0Dataframe = pd.DataFrame()
label3_1Dataframe = pd.DataFrame()
label3_0Dataframe = pd.DataFrame()

for ID in label1_1.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label1'] = 1.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label1_1Dataframe = label1_1Dataframe.append(tempData)
    print ID,"..........."

for ID in label1_0.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label1'] = 0.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label1_0Dataframe = label1_0Dataframe.append(tempData)
    print ID,"..........."

for ID in label2_1.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label2'] = 1.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label2_1Dataframe = label2_1Dataframe.append(tempData)
    print ID,"..........."

for ID in label2_0.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label2'] = 0.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label2_0Dataframe = label2_0Dataframe.append(tempData)
    print ID,"..........."

for ID in label3_1.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label3'] = 1.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label3_1Dataframe = label3_1Dataframe.append(tempData)
    print ID,"..........."

for ID in label3_0.index:
    inputPath = inputFeaturePath + str(ID) + ".csv"
    tempData = pd.read_csv(inputPath)
    tempData['label3'] = 0.0
    tempData = tempData[:]
    cols = tempData.columns.tolist()
    cols = cols[0:-2] + cols[-1:] + cols[-2:-1]
    tempData = tempData[cols]
    label3_0Dataframe = label3_0Dataframe.append(tempData)
    print ID,"..........."

label1_1Dataframe = label1_1Dataframe.reset_index().ix[:,2:]
label1_1Dataframe.to_csv(outputLabelPath+"label1_1.csv")

label1_0Dataframe = label1_0Dataframe.reset_index().ix[:,2:]
label1_0Dataframe.to_csv(outputLabelPath+"label1_0.csv")

label2_1Dataframe = label2_1Dataframe.reset_index().ix[:,2:]
label2_1Dataframe.to_csv(outputLabelPath+"label2_1.csv")

label2_0Dataframe = label2_0Dataframe.reset_index().ix[:,2:]
label2_0Dataframe.to_csv(outputLabelPath+"label2_0.csv")

label3_1Dataframe = label3_1Dataframe.reset_index().ix[:,2:]
label3_1Dataframe.to_csv(outputLabelPath+"label3_1.csv")

label3_0Dataframe = label3_0Dataframe.reset_index().ix[:,2:]
label3_0Dataframe.to_csv(outputLabelPath+"label3_0.csv")
