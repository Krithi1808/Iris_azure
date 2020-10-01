from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import urllib2
import json
import os
#TEMPLATE_DIR = os.path.abspath('../templates')
#STATIC_DIR = os.path.abspath('../static')
#Initialise your name of Flask application as app

app= Flask(__name__)


#Define a view function named 'home', which renders the html page 'home.html'
#Ensure that the view function 'home' is routed when a user access the URL '/' .

@app.route('/')
def home():
	return render_template('index.html')



#Define a view function named 'predict', which does the function of getting the text entered by user in home.html and predicts if it is spam or not and renders the result in result.html
#Ensure that the view function 'predict' is routed when a user access the URL '/predict' .


'''steps
1. Load and store your vectorizer from your pickle file 'vector.pkl'
2. Load and store your classifier from your pickle file 'NB_spam_model.pkl'
3. Retrive the message given in the text area of home page.
4. Use vectoriser to fit transform the given message and store it in vect
6. Convert the data into array
7. Predict the label for the message array with the classifier loaded and store it in variable predicted_label

'''

@app.route('/predict',methods=['POST'])
def predict():
	#id= request.form['id_iris']
	#sl=request.form['sepallength']
	#sw=request.form['sepalwidth']
	#pl=request.form['petallength']
	#pw=request.form['petalwidth']
	#print(id)
	#print(sl)
	#print(sw)
	#print(pl)
	#print(pw)
	
	data = {
			"Inputs": {
                "input1":
                [
                    {
                            'Id': request.form['id_iris'],   
                            'SepalLengthCm': request.form['sepallength'],   
                            'SepalWidthCm': request.form['sepalwidth'],   
                            'PetalLengthCm': request.form['petallength'],   
                            'PetalWidthCm': request.form['petalwidth']
                    }
                ],
        },
		"GlobalParameters":  {
							}
	}

	body = str.encode(json.dumps(data))

	url = 'https://ussouthcentral.services.azureml.net/workspaces/0766bd3baf6d480c96c48e944e0e60cf/services/cf8c3ee2b1704ab2aa7c0025c2955e91/execute?api-version=2.0&format=swagger'
	api_key = 'L61BLFcmrdadbHCfaPhek8pz7T40lzhqPwXuPkfFOQRiGRQ0ykz7VIA2MwRNHGcq2lI31Hj8x2UA68n9nKmS5A==' # Replace this with the API key for the web service
	headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

	req = urllib2.Request(url, body, headers)
	response = urllib2.urlopen(req)

	result = response.read()
	print(result)
	y = json.loads(result)
	print(y['Results']['output1'][0]['Scored Labels'])
	prediction_value=y['Results']['output1'][0]['Scored Labels']
	prediction_value1=y['Results']['output1'][0]['Scored Probabilities for Class "Iris-setosa"']
	print(prediction_value1)
	prediction_value2=y['Results']['output1'][0]['Scored Probabilities for Class "Iris-versicolor"']
	print(prediction_value2)
	prediction_value3=y['Results']['output1'][0]['Scored Probabilities for Class "Iris-virginica"']
	print(prediction_value3)
	
	
	

	
	return render_template('result.html',prediction = prediction_value,prediction1 = prediction_value1,prediction2 = prediction_value2,prediction3 = prediction_value3)



	
		






# make your app run in 0.0.0.0 host and port 8000
if __name__ == '__main__':
	app.run(debug=True)
