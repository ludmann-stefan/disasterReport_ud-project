from __init__ import app
import flask as Flask
from flask import request, redirect, render_template, send_from_directory


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly, json
import csv
from sqlalchemy import create_engine
import sqlite3

import numpy as np
import pandas as pd
import csv
from sqlalchemy import create_engine
import sqlite3
import pickle
import joblib

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

def tokenize (tweet):
    tweet = re.sub(r"[^a-zA-Z0-9?#-]", " ", tweet.lower())
    tweet = tknzr.tokenize(tweet)
    tweet = [WordNetLemmatizer().lemmatize (a) for a in tweet]
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    return tweet

def get_predictions (in_arg):
    filename = '/Models/finalized_model.sav'
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

accuracy_score = pd.read_csv ('./Models/accuracy_score.csv')

graph_one = [(go.Bar(
    x = accuracy_score['Unnamed: 0'],
    y = accuracy_score['precision'],
    name = 'test'))]

layout_one = dict (title = 'Precission Score of test-Data',
    xaxis = dict (title = ''),
    yaxis = dict (title= 'accuracy of predicted Tweets')
                )

figures = []
figures.append (dict(data = graph_one, layout = layout_one))

ids = ['figures-{}'.format (i) for i, _ in enumerate(figures)]

figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
tweet = []

engine = create_engine('sqlite:///data/data.db')
data = pd.read_sql_table ('data', engine)


# test whether classification is working

@app.route('/home', methods=['GET', 'POST'])
def index():

    number = np.random.randint(0, len(data), 1)[0]
    example = data['message'].loc [number]


    return render_template('Website.html',  figuresJSON = figuresJSON, ids = ids, example = example)




@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # user input in 'tweet'
    number = np.random.randint(0, len(data), 1)[0]
    example = data['message'].loc [number]
    if request.method == "POST":
        tweet_in = request.form
        tweet =  (tweet_in.get ('UserInput'))
        print (tweet)
        # use classifier
        pred = get_predictions (str (tweet)).T

        a = pred[pred[0]== 1]
        b = pred[pred[0]== 0]
        print (a)
        # redirect (str(request.url + '#item-1'))

        return render_template('predict.html', tweet = tweet, predictedCat = list(a.index), otherCat = list(b.index),  figuresJSON = figuresJSON, ids = ids, example = example)


    return render_template('website.html',   figuresJSON = figuresJSON, ids = ids, example = example)
# predictedCat = a.index, otherCat = b.index
