# import libraries

import numpy as np
import pandas as pd
import csv
from sqlalchemy import create_engine
import sqlite3
import pickle
from NLPpackage import tokenize, get_predictions, split_categories



def load (filename):

    '''
        load data, drop duplicates
    '''

    filename = str (filename)
    path = './data/'
    dialect = 'excel'

    file = pd.read_csv (str(path + filename), dialect = dialect)
    return file


def clean (raw_df):
    '''
        Drop Dublicates
    '''
    clean_df = raw_df [raw_df ['id'].duplicated()== False]
    return clean_df

def combineDF (DF_a, DF_b):
    '''
        Merge the DataFrames on the Index
    '''

    data = pd.DataFrame (DF_a.merge (DF_b, left_index=True, right_index=True))
    data = data.set_index (['id'])
    return data

messages = load ('messages.csv')
categories = load ('categories.csv')

messages =  clean (messages)
categories = clean (categories)

# split the category-data and get binaries
categories = split_categories(categories)

data_all =  combineDF (messages, categories[0])


'''
    Drop Records, where Related == 2, because I cannot interpret it
'''
print (len(data_all[data_all ['related_cat'] == 2]))
data_all = data_all.drop (data_all[data_all ['related_cat'] == 2].index, axis = 0)


'''
    Upload the new DataFrame to a SQL Database
'''
engine = create_engine('sqlite:///data/data.db')

data_all.to_sql ('data', engine, if_exists = 'replace')
print ('finished ETL')
