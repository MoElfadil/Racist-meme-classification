# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:22:17 2021

@author: Mohammed
"""
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

directory = r'C:\Users\Mohammed\Documents\DE4\Masters\img'
dev_seen = pd.read_json("dev_seen.jsonl", lines=True)
dev_unseen = pd.read_json("dev_unseen.jsonl", lines=True)
test_seen = pd.read_json("test_seen.jsonl", lines=True)
test_unseen = pd.read_json("test_unseen.jsonl", lines=True)
train = pd.read_json("train.jsonl", lines=True)

sorted_train = train.sort_values('id')
sorted_train = sorted_train[0:815]

id_list = []

for filename in os.listdir(directory):
    ids = filename[1:5]
    id_list.append(int(ids))

for index, row in sorted_train.iterrows():
    x = row['id']
    if x not in id_list:
        sorted_train = sorted_train.drop(index)

count = 0

survey1 = pd.read_csv('SoGoSurvey_Automatic detection of racist _1.csv')
survey2 = pd.read_csv('SoGoSurvey_Labelling racist memes study_2.csv')
survey3 = pd.read_csv('SoGoSurvey_Automatic detection of racist _3.csv')
survey4 = pd.read_csv('SoGoSurvey_Copy 1 of Automatic detectio_4.csv')
survey5 = pd.read_csv('SoGoSurvey_Copy 2 of Automatic detectio_6.csv')
survey6 = pd.read_csv('SoGoSurvey_Copy 3 of Automatic detectio_7.csv')
survey7 = pd.read_csv('SoGoSurvey_Copy 4 of Automatic detectio_8.csv')
survey8 = pd.read_csv('SoGoSurvey_Copy 5 of Automatic detectio_9.csv')
survey9 = pd.read_csv('SoGoSurvey_Copy 6 of Automatic detectio_10.csv')
survey10 = pd.read_csv('SoGoSurvey_Copy 7 of Automatic detectio_11.csv')
survey11 = pd.read_csv('SoGoSurvey_Copy 8 of Automatic detectio_12.csv')
survey12 = pd.read_csv('SoGoSurvey_Copy 9 of Automatic detectio_13.csv')

survey2 = survey2.drop([4, 5, 6, 7])
survey = pd.concat([survey1, survey2,survey3, survey4,survey5, survey6,survey7, survey8,survey9, survey10,survey11, survey12], axis=1)

zeros_misclassified = 0
ones_misclassified = 0
ones_unanimous = 0
zeros_unanimous = 0


for (columnName, columnData) in survey.iteritems():
    name = columnName
    values = columnData.values
    if values[0] == "Yes":
        nans = sum(pd.isnull(values))
        x = sorted_train.iloc[count].name
        if nans == 3:
            ones_unanimous += 1
            if sorted_train.at[x,'label'] == 1:
                ones_misclassified += 1
                sorted_train.at[x,'label'] = 0
                count += 1
            else:
                sorted_train.at[x,'label'] = 0
                count += 1
        elif nans == 2:
            if sorted_train.at[x,'label'] == 1:
                ones_misclassified += 1
                sorted_train.at[x,'label'] = 0
                count += 1
            else:
                sorted_train.at[x,'label'] = 0
                count += 1
        elif nans == 1:
            if sorted_train.at[x,'label'] == 0:
                zeros_misclassified += 1
                sorted_train.at[x,'label'] = 1
                count += 1
            else:
                sorted_train.at[x,'label'] = 1
                count += 1
        else:
            zeros_unanimous += 1
            if sorted_train.at[x,'label'] == 0:
                zeros_misclassified += 1
                sorted_train.at[x,'label'] = 1
                count += 1
            else:
                sorted_train.at[x,'label'] = 1
                count += 1
    if count == 599:
        break

train, test = train_test_split(sorted_train, test_size=0.3, random_state=42)
test, dev = train_test_split(test, test_size=0.25, random_state=42)
test_unseen, test_seen = train_test_split(test, test_size=0.33, random_state=42)
dev_unseen, dev_seen = train_test_split(dev, test_size=0.5, random_state=42)

del test["label"]
del test_seen["label"]

train.to_json (r'C:\Users\Mohammed\Documents\DE4\Masters\data\train.jsonl',orient='records',lines=True)
test_seen.to_json (r'C:\Users\Mohammed\Documents\DE4\Masters\data\test_seen.jsonl',orient='records',lines=True)
test_unseen.to_json (r'C:\Users\Mohammed\Documents\DE4\Masters\data\test_unseen.jsonl',orient='records',lines=True)
dev_seen.to_json (r'C:\Users\Mohammed\Documents\DE4\Masters\data\dev_seen.jsonl',orient='records',lines=True)
dev_unseen.to_json (r'C:\Users\Mohammed\Documents\DE4\Masters\data\dev_unseen.jsonl',orient='records',lines=True)