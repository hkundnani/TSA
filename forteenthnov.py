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


c =[]


complist = result['Compound'].tolist()
for i in range(len(complist)):
    if i<3:
        c.append(0.0)
    if i >3:
        val= (complist[i-1]+complist[i-2]+complist[i-3])/3
        c.append(val)
        
date_list = result['Date'].tolist()
adj_list = result['Adjvalue'].tolist()
label_list = result['Label'].tolist()
final_list = zip(date_list, adj_list, label_list, complist, c)
label = ['Date', 'Adjvalue', 'Label', 'Compound', 'Prev3comp']
final_dframe = pd.DataFrame.from_records(final_list, columns = label)

train, test = train_test_split(final_dframe, test_size=0.3, random_state=10)



        

        
        


fit = classifier.fit(train['Prev3comp'].reshape(len(train['Prev3comp']),1), train['Label'])
y_pred = classifier.predict(test['Label'].reshape(len(test['Label']),1))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(test['Prev3comp'].reshape(len(test['Prev3comp']),1),test['Label'].reshape(len(test['Label']),1) )))
accuracy = accuracy_score(y_pred,test['Label'].reshape(len(test['Label'])))
print(accuracy)







