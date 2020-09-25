# import libraries
'''
    This one searches the best parameters and saves the model as the '_cv.sav' file.



'''
import numpy as np
import pandas as pd
import joblib
import csv
from sqlalchemy import create_engine
import sqlite3

from  NLPpackage import tokenize, get_predictions
from sklearn import tree
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

'''
    read in the data
'''
engine = create_engine('sqlite:///data/data.db')
data = pd.read_sql_table ('data', engine)
data = data.set_index (['id'])

categories = list ([word for word in data.columns if word.find('_cat') > 0])
categories.remove ('child_alone_cat')
X_train, X_test, y_train, y_test = train_test_split(data['message'], data[categories], test_size = 0.5)


'''
    Load the pipeline that is intended to be optimized
'''
reduced_pipeline = joblib.load ('./Models/finalized_model.sav')


'''
    Set the parameters
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

'''
    Save the Pipeline as a joblib file
'''

model_cv = cv_pipeline
filename = './Models/finalized_model_cv.sav'
joblib.dump(model_cv, filename)


'''
    Check the Classification report
'''
classifi = []
for i in range(len(y_test.columns)):
    cat_name = ('Category: {} '.format(y_test.columns[i]))
    cat_score = (classification_report(y_test.iloc[:, i].values, y_pred_cv.iloc[:, i], output_dict = True))
    classifi = [cat_name, cat_score]

print (classifi)

a = pd.DataFrame (y_test.max (axis = 0))
b = pd.DataFrame (y_pred_cv.max (axis = 0))
a[1] = b[0]
a [a.max (axis = 1) != 1]== True
list((a [a.max (axis = 1) != 1]== True).index)

accu_score_cv = classification_report(y_test.drop  (list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            , y_pred_cv.drop(list((a [a.max (axis = 1) != 1]== True).index), axis= 1)
                            ,target_names = y_test.columns.drop(list((a [a.max (axis = 1) != 1]== True).index)), output_dict = True )
accu_score_cv = pd.DataFrame.from_dict (accu_score_cv)
accu_score_cv.T.to_sql ('accuracy_cv', engine, if_exists = 'replace')
