# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:00:39 2021

@author: LENOVO
"""



import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pickle

df= pd.read_csv("C:/Users/LENOVO/Desktop/dp/Copy of data_new.csv")
df=df.dropna(how='any')

# Features and Labels
df['label'] = df['Class'].map({'Non Abusive': 0, 'Abusive': 1})
X = df['cleaned_3']
y = df['label']
	
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
   
pickle.dump(cv, open('tranform.pkl', 'wb'))
    
   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model=MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))