# import libraries

import numpy as np
import pandas as pd

# import scikit learn libs
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree

# data save and load
import joblib
import csv
from sqlalchemy import create_engine
import sqlite3


from NLPpackage import funktions



# functions here:


# load data and set index
engine = create_engine('sqlite:///data/data.db')

data = pd.read_sql_table ('data', engine).set_index (['id'])


categories = list ([word for word in data.columns if word.find('_cat') > 0])
categories.remove ('child_alone_cat') # only zeros in this category

# split into X and y
X_train, X_test, y_train, y_test = train_test_split(data['message'], data[categories], test_size = 0.5)
print ('loading complete, analyzing the model')



'''
________________________________________________________________________________

    short test [with 20 lines]
'''

# Pipiline simple test-version
simple_pipeline = Pipeline([

    ('count', CountVectorizer(tokenizer = funktions.tokenize))
    , ('tfidf', TfidfTransformer())
    , ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

simple_pipeline.fit(X_train.iloc[0:20], y_train.iloc[0:20])

y_pred = simple_pipeline.predict(X_test.iloc[0:20])
y_pred = pd.DataFrame (y_pred, index = X_test.iloc[0:20],  columns = categories)
filename = './Models/test_model.sav'
joblib.dump(simple_pipeline, filename)
(classification_report(y_test.iloc[0:20].values, y_pred.iloc[0:20].values,  target_names = categories, zero_division=0))


print ('check done')

'''
________________________________________________________________________________

    start of the ML teaching

'''


reduced_pipeline = Pipeline([
    ('count', CountVectorizer(tokenizer = funktions.tokenize, binary = True)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
#

reduced_pipeline.fit(X_train, y_train)
print ('training complete')

filename = './Models/finalized_model.sav'
joblib.dump(reduced_pipeline, filename)

y_pred = reduced_pipeline.predict(X_test)
y_pred = pd.DataFrame (y_pred, index = X_test,  columns = categories)

'''
    Classification Report only works when there are binary values...
'''

classifi = []
for i in range(len(y_test.columns)):
    cat_name = ('Category: {} '.format(y_test.columns[i]))
    cat_score = (classification_report(y_test.iloc[:, i].values, y_pred.iloc[:, i], output_dict = True))
    classifi = [cat_name, cat_score]

print (classifi)

a = pd.DataFrame (y_test.max (axis = 0))
b = pd.DataFrame (y_pred.max (axis = 0))
a[1] = b[0]
a [a.max (axis = 1) != 1]== True
list((a [a.max (axis = 1) != 1]== True).index)

accu_score = classification_report(y_test.drop  (list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            , y_pred.drop(list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            ,target_names = y_test.columns.drop(list((a [a.max (axis = 1) != 1]== True).index)), output_dict = True )
accu_score = pd.DataFrame.from_dict (accu_score)
accu_score.T.to_csv ('Models/accuracy_score.csv')


mod = joblib.load ('./Models/finalized_model.sav')
in_arg = 'test need some water or something like this'
predictions = mod.predict ([in_arg])
predictions = pd.DataFrame (predictions)

print ('finished')
