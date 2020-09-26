'''
    Decide if Grid Search On (1) or Off (0)
'''

GridSearchOn = 1

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

# my own Package: pip install NLPpackage-Package==1.3
from NLPpackage import tokenize, get_predictions, split_categories


'''
functions part:
'''

def save_model (model, filename):
    model = model
    filename = filename
    joblib.dump(model, filename)
    return ()


engine = create_engine('sqlite:///data/data.db')


# load data and set index
data = pd.read_sql_table ('data', engine).set_index (['id'])


# add _cat, so every category is clear identifiable
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

    ('count', CountVectorizer())
    , ('tfidf', TfidfTransformer())
    , ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

simple_pipeline.fit(X_train.iloc[0:20], y_train.iloc[0:20])

filename = './Models/test_model.sav'
save_model (simple_pipeline, filename)

y_pred = pd.DataFrame (simple_pipeline.predict(X_test.iloc[0:20]), index = X_test.iloc[0:20],  columns = categories)
(classification_report(y_test.iloc[0:20].values, y_pred.iloc[0:20].values,  target_names = categories, zero_division=0))

print ('check done')


'''
________________________________________________________________________________

    start of the ML teaching

'''


reduced_pipeline = Pipeline([

    ('count', CountVectorizer(tokenizer = tokenize, binary = True)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
#

reduced_pipeline.fit(X_train, y_train)
print ('training complete')


filename = './Models/finalized_model.sav'
save_model (reduced_pipeline, filename)


y_pred = pd.DataFrame (reduced_pipeline.predict(X_test), index = X_test,  columns = categories)


'''
    Classification Report only works when there are binary values...

    Bit messy part of code,
'''

# detailed Report
classifi = []
for i in range(len(y_test.columns)):
    cat_name = ('Category: {} '.format(y_test.columns[i]))
    cat_score = (classification_report(y_test.iloc[:, i].values, y_pred.iloc[:, i], output_dict = True))
    classifi = [cat_name, cat_score]

print (classifi)

# list-like, more compact report, should work for this need
accu_score = classification_report(y_test, y_pred,target_names = y_test.columns, output_dict = True )
accu_score = pd.DataFrame.from_dict (accu_score)


'''
    Only Classifying Binary Cols, so looking for Maximum in each Category,
    if its greater than 1 the Category will be dropped out of the Report.

    The Code below can be run, if the Code stops, while the record is made.
'''

### BACKUP-CODE
'''
a = pd.DataFrame (y_test.max (axis = 0))
    # maximum each col in Validation set
b = pd.DataFrame (y_pred.max (axis = 0))
    # maximum each col in Predicted set
a[1] = b[0]
    # combining both sets
a [a.max (axis = 1) != 1]== True
list((a [a.max (axis = 1) != 1]== True).index)

print (a)

accu_score = classification_report(y_test.drop  (list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            , y_pred.drop(list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            ,target_names = y_test.columns.drop(list((a [a.max (axis = 1) != 1]== True).index)), output_dict = True )
accu_score = pd.DataFrame.from_dict (accu_score)

'''

accu_score.T.to_csv ('Models/accuracy_score.csv')


'''
    Test the Model:
'''

mod = joblib.load ('./Models/finalized_model.sav')
in_arg = 'test need some water or something like this'
predictions = mod.predict ([in_arg])
predictions = pd.DataFrame (predictions)

avail_data = pd.DataFrame (y_train.sum(axis = 0))


'''
    Finally Store some Information about the Training (data and precision)
'''
accu_score.T.to_sql ('accuracy', engine, if_exists = 'replace')
avail_data.to_sql ('avail_data', engine, if_exists = 'replace')

print ('finished - Pipeline')


if GridSearchOn == 1:
    '''
        Grid Search part

        '''


    parameters = {
        'count__ngram_range': ((1, 1) , (1, 2))
        , 'count__max_df': (0.5, 1.0)
        , 'tfidf__use_idf': (True, False)
        }

    cv_pipeline = GridSearchCV(reduced_pipeline, param_grid = parameters)
    cv_pipeline.fit(X_train, y_train)

    print (cv_pipeline.best_params_)
    y_pred_cv = pd.DataFrame (cv_pipeline.predict (X_test), index = X_test, columns = categories)

    filename = 'finalized_model_cv.sav'
    save_model (cv_pipeline, filename)


    accu_score_cv = classification_report (y_test, y_pred_cv,target_names = y_test.columns, output_dict = True )

    accu_score_cv = pd.DataFrame.from_dict (accu_score_cv)
    accu_score_cv.T.to_sql ('accuracy_cv', engine, if_exists = 'replace')



print ('finished')
