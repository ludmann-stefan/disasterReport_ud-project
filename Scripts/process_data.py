# import libraries

import numpy as np
import pandas as pd
import csv
from sqlalchemy import create_engine
import sqlite3
import pickle
from funktions import tokenize, dummies, split_categories

# load data, drop duplicates
messages = pd.read_csv ('./data/messages.csv', dialect = 'excel')
messages = messages [messages ['id'].duplicated()== False]

categories = pd.read_csv ('./data/categories.csv', dialect = 'excel')
categories = categories [categories ['id'].duplicated()== False]

# split the category-data and get binaries
categories = split_categories(categories)

b = (categories[1])

# split data into training set and test set
data_all = pd.DataFrame (messages.merge (categories[0], left_index=True, right_index=True))

y_cats = data_all[categories[1]].sum (axis =0)[data_all[categories[1]].sum (axis =0)> 1].index
y_cats
print (data_all [y_cats].sum (axis = 0))
data_all = data_all.set_index (['id'])
print (data_all.head())

print (data_all[data_all ['related_cat'] == 2])
data_all = data_all.drop (data_all[data_all ['related_cat'] == 2].index, axis = 0)

engine = create_engine('sqlite:///data/data.db')

data_all.to_sql ('data', engine, if_exists = 'replace')
print ('finished ETL')
