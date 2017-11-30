# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:28:26 2017

@author: Ankita
"""

import pandas as pd
import numpy as np
import nltk
from nltk import tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import csv
import sklearn
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
#import wordcloud
#from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
#import ggplot


classifier =  LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200)
 







url1 = 'C:/Users/ARUN/Desktop/FSU/data mining/project/TSA/tweets_sentiment_score/CSCO_sentiment_score.csv'
url2 = 'C:/Users/ARUN/Desktop/FSU/data mining/project/TSA/processed_djia/CSCO.csv'
df = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')

djia.columns= ['Date', 'Adjvalue', 'Label']









'''djia['Date'] =  pd.to_datetime(djia['Date'])

djia['Date'] = djia['Date'].apply(lambda x: x.date())

djia['Date'] = djia['Date'].astype(str)'''



label = ['Date', 'Compound']
testdf = pd.DataFrame.from_records(df, columns = label)
testdf['Date'].dropna(inplace = True)
testdf[['Compound']] = testdf[['Compound']].astype(float)
#print(testdf.isnull().sum())

testdf = testdf[testdf.Compound != 0.0000]

newtestdf = testdf.groupby('Date').mean().reset_index()

newtestdf['Date'] = newtestdf['Date'].astype(str)

#print(newtestdf['Date','Compound'])
#print(djia['Date','Adjvalue','Label'])

result = pd.merge(djia, newtestdf, how = 'outer', on=['Date'])

result['Compound'].fillna(0.0, inplace = True)
result['Label'].fillna(0.0, inplace = True)
result['Label'].fillna(0.0, inplace = True)
result.dropna(axis=0, inplace = True)


train,test = train_test_split(result,test_size=0.2,random_state=10)# about random_state


compound_list = train['Compound'].tolist()
label_list = train['Compound'].tolist()

fit = classifier.fit(train['Compound'].reshape(len(train['Compound']),1), train['Label'])
y_pred = classifier.predict(test['Label'].reshape(len(test['Label']),1))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(test['Compound'].reshape(len(test['Compound']),1),test['Label'].reshape(len(test['Label']),1) )))








