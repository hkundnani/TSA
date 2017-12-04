# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:20:17 2017

@author: Jarvis
"""


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression



url1 = 'tweets_sentiment_score/CSCO_sentiment_score.csv'
url2 = 'processed_djia/CSCO.csv'
df = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')
djia.columns= ['Date', 'Adjvalue', 'Label']

label = ['Date', 'Compound']
testdf = pd.DataFrame.from_records(df, columns = label)
testdf['Date'].dropna(inplace = True)
testdf[['Compound']] = testdf[['Compound']].astype(float)
testdf = testdf[testdf.Compound != 0.0000]
newtestdf = testdf.groupby('Date').mean().reset_index()
newtestdf['Date'] = newtestdf['Date'].astype(str)
result = pd.merge(djia, newtestdf, how = 'outer', on=['Date'])
result['Compound'].fillna(0.0, inplace = True)
result['Label'].fillna(0.0, inplace = True)
result['Label'].fillna(0.0, inplace = True)
result.dropna(axis=0, inplace = True)

c =[]
complist = result['Compound'].tolist()
for i in range(len(complist)):
    if i==1:
        c.append(0.0)
    if i==2:
        c.append(complist[i-1])
    if i==3:
        val = (complist[i-1]+complist[i-2])/2
        c.append(val)
    if i > 3:
        val = (complist[i-1]+complist[i-2]+complist[i-3])/3
        c.append(val)
        
date_list = result['Date'].tolist()
adj_list = result['Adjvalue'].tolist()
label_list = result['Label'].tolist()
final_list = zip(date_list, adj_list, label_list, complist, c)
label = ['Date', 'Adjvalue', 'Label', 'Compound', 'Compound3']
final_dframe = pd.DataFrame.from_records(final_list, columns = label)

classifier =  LinearRegression()

train,test = train_test_split(final_dframe,test_size=0.2,random_state=42)# about random_state


fit = classifier.fit(train['Compound3'].reshape(len(train['Compound3']),1), train['Label'])
y_pred = classifier.predict(test['Label'].reshape(len(test['Label']),1))
print('Accuracy of linear regression classifier on test set: {:.2f}'.format(classifier.score(test['Compound3'].reshape(len(test['Compound3']),1),test['Label'].reshape(len(test['Label']),1) )))
print (y_pred)
print (test['Label'])