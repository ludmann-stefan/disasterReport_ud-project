# import libraries

import numpy as np
import pandas as pd
import pickle
import csv
from sqlalchemy import create_engine
import sqlite3

from funktions import tokenize, dummies, predictions
from sklearn import tree
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

engine = create_engine('sqlite:///data/data.db')

data = pd.read_sql_table ('data', engine)
data = data.set_index (['id'])

categories = list ([word for word in data.columns if word.find('_cat') > 0])
X_train, X_test, y_train, y_test = train_test_split(data['message'], data[categories], test_size = 0.5)



reduced_pipeline = pickle.load(open('./Models/nonCVpredict', 'rb'))

parameters = {
    'count__ngram_range': ((1, 1) , (1, 2))
    , 'count__max_df': (0.5, 1.0)
    , 'tfidf__use_idf': (True, False)
    , 'clf__estimator__min_samples_split': [2, 4]
}

cv_pipeline = GridSearchCV(reduced_pipeline, param_grid = parameters)
cv_pipeline.fit(X_train, y_train)
print (cv_pipeline.best_params_)
y_pred_cv = cv_pipeline.predict (X_test)

model_cv = cv_pipeline
pickle.dump (model_cv, open('./Models/CVpredict', 'wb'))
