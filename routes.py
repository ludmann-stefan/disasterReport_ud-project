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

from NLPpackage import tokenize, get_predictions



engine = create_engine('sqlite:///data/data.db')

accuracy_score = pd.read_sql_table ('accuracy', engine)
accuracy_score_cv = pd.read_sql_table ('accuracy_cv', engine)
data = pd.read_sql_table ('data', engine)
avail_data = (pd.read_sql_table ('avail_data', engine))



print (avail_data)

x = list(accuracy_score['index'])
print (x)
y1 = list (accuracy_score['precision'])
y2 = list (accuracy_score_cv['precision'])
xJSON = json.dumps(list(x))

graph_three = [(go.Bar(
    x = avail_data['index'],
    y = avail_data['0'],
    name = 'available Data'))]

layout_three = dict (title = 'Training Data',
    xaxis = dict (title = ''),
    yaxis = dict (title= 'count')
                )

figures = []
figures.append (dict(data = graph_three, layout = layout_three))

ids = ['figures-{}'.format (i) for i, _ in enumerate(figures)]

figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
tweet = []



# test whether classification is working

@app.route('/', methods=['GET', 'POST'])
def index():

    number = np.random.randint(0, len(data), 1)[0]
    example = data['message'].loc [number]


    return render_template('Website.html', x = xJSON, y1 = y1, y2 = y2, figuresJSON = figuresJSON, ids = ids, example = example)




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

        return render_template('predict.html', x = xJSON, y1 = y1, y2 = y2, tweet = tweet, predictedCat = list(a.index), otherCat = list(b.index),  figuresJSON = figuresJSON, ids = ids, example = example)


    return render_template('Website.html',  x = xJSON, y1 = y1, y2 = y2, figuresJSON = figuresJSON, ids = ids, example = example)
# predictedCat = a.index, otherCat = b.index
