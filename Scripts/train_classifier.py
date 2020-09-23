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

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download ('punkt')
nltk.download ('stopwords')
nltk.download ('wordnet')

tknzr = TweetTokenizer ()
import re



# functions here:
def get_predictions (in_arg):
    filename = './Models/finalized_model.sav'
    dt_model = joblib.load(filename)
    predictions = dt_model.predict ([in_arg])
    predictions = pd.DataFrame (predictions)

    predictions.rename (columns = {0: 'related',
     1: 'request', 2: 'offer', 3: 'aid related',
     4: 'medical help', 5: 'medical products',
     6: 'search and rescue', 7:'security',
     8: 'military', 9:'child alone',
     10: 'water', 11: 'food', 12: 'shelter',
     13: 'clothing', 14: 'money', 15: 'missing people',
     16: 'refugees', 17: 'death', 18: 'other aid',
     19: 'infrastructure related', 20: 'transport',
     21: 'buildings', 22: 'electricity', 23: 'tools',
     24: 'hospitals', 25: 'shops', 26: 'aid centers',
     27: 'other infrastructure', 28: 'weather related',
     29: 'floods', 30: 'storm', 31: 'fire',
     32: 'earthquake', 33: 'cold',
     34: 'other weather', 35: 'direct report'}, inplace = True)

    sumation = predictions.sum (axis = 1)
    print (sumation)
    result = pd.DataFrame(predictions)
    return result

def tokenize (tweet):
    tweet = re.sub(r"[^a-zA-Z0-9?#-]", " ", tweet.lower())
    tweet = tknzr.tokenize(tweet)
    tweet = [WordNetLemmatizer().lemmatize (a) for a in tweet]
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    return tweet

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

    ('count', CountVectorizer())
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

    ('count', CountVectorizer( binary = True)),
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

avail_data = data.sum(axis = 0)[4:]

accu_score.T.to_sql ('accuracy', engine, if_exists = 'replace')
avail_data.to_sql ('avail_data', engine, if_exists = 'replace')

print ('finished')
