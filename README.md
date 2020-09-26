# disasterReport_ud-project
Natural Language Processing, with ETL, ML and HTML-Dashboard

Figure Eight Project – analyzing Disaster Tweets
within the Udacity Data Scientist NanoDegree


	Table of Content
1. Project Motivation
2. Installation and Files
3. Results
4. Licensing, Authors and Acknowledgements


	1. Project Motivation
Fast reaction following a disaster is crucial. Emergency-help can be asked via twitter or other social media channels. This can be due to high frequency in calls. Using Machine Learning algorithms to connect those needs with the right help organization, without the need of manpower, can be key in maintaining help coordination, during a disaster. 

Being a part of the Study-Program ‚Data-Scientist‘ @Udacity, this Project is aiming on supervised learning, natural language processing and categorization.
The Project is split into three parts- ETL, Machine Learning Pipeline and Web development:

The goal is to link between an input and an organization-service. The input-Data is combined, cleaned and stored in a SQLite database.
Then a Machine Learning Pipeline with a CountVectorizer (inlcuding a custom Tokenizer), TF-IDF Transformer and a Classifier is set up and taught with the training data. Further, a best Parameter analysis is done with the GridSearch CV tool from Scikit Learn.
To show you the results, a Web-Page is set up and runs the classifier. 

https://ud-ds-disasterresponse-stefan.herokuapp.com/

￼


— How to Part — 

	2. Installation and Files
This Code runs on Python 3.x and requires the following Libraries:
- NumPy and Pandas
- scikit-learn
- nltk
- sqalchemy
- joblib, csv

An own tokenizer was built and put on PyPi
	pip install pip install NLPpackage-Package==1.3
and	from NLPpackage import tokenize, get_predictions 		
		as call options


The Code is deployed on a heroku webserver, so it was modified in some Parts. Those Parts where highlighted with a [xxx] squared-bracket, where xxx shows the Code Line to change.



Data:
categories.csv, Categories and messages
messages.csv, Multilingual disaster response messages.
	

| — Project
    |  run.py 	(runs the code)
    |  routes.py	(required)
    |  __init__.py	(-||-)			[ app.run (host = '0.0.0.0', port = 8001, debug = True) — Line 4]
    |  READme.rtf

    | — data
         |  categories.csv
         |  messages.csv
         |  Twitter-sentiment-self-drive-DFE.csv
         |  data.db				(SQLite Database)

    | — Scripts
         |  process_data.py		(ETL)
         |  train_classifier.py		(ML, nlp)
         |  gridsearcht.py			(ParameterSearch on the ML)
				NEW:	(gridSearch is integrated in train_classifier.py)	

    | — Models
         |  accuracy_score.csv
         |  finalized_model.sav	
         |  finallized_model_cv.sav
         |  test_model.sav			(only to check if code is running)

    | — templates
         |  Website.html			Landing-Page
         |  predict.html			Result-Page

    | — static
	some css and js files for the templates



 Steps:
	process Data
	train Classifier
		find the best Parameters with GridSearch
	real time Message classification


	3. Results
Running the ML Pipeline (count-Vectorizer, TF-IDF and AdaBoostClassifier)
The Classifier was mainly chosen because of the file size.



 Webpage while it is on: (since this is a demonstration, it can be taken down by time…) 
	https://ud-ds-disasterresponse-stefan.herokuapp.com/




	4. Licensing, Authors and Acknowledgements
The Data is provided by Figure Eight.
Project Design by Udacity

Code written by Stefan Ludmann

