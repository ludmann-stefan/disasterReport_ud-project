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


from NLPpackage import funktions

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
print (funktions.get_predictions ('test 234').T)

@app.route('/', methods=['GET', 'POST'])
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
        pred = funktions.get_predictions (str (tweet)).T

        a = pred[pred[0]== 1]
        b = pred[pred[0]== 0]
        print (a)
        # redirect (str(request.url + '#item-1'))

        return render_template('predict.html', tweet = tweet, predictedCat = list(a.index), otherCat = list(b.index),  figuresJSON = figuresJSON, ids = ids, example = example)


    return render_template('website.html',   figuresJSON = figuresJSON, ids = ids, example = example)
# predictedCat = a.index, otherCat = b.index
