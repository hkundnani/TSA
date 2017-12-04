# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:28:26 2017

@author: Ankita
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import pylab as pl
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
#import ggplot


classifier =  LogisticRegression(C=32.0)
 







url1 = 'tweets_sentiment_score/MSFT_sentiment_score.csv'
url2 = 'processed_djia/MSFT.csv'
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
final_dframe = final_dframe[final_dframe.Compound != 0.0000]
# print(final_dframe)

train, test = train_test_split(final_dframe, test_size=0.3, random_state=10)

# C_range = np.logspace(-2, 10, 13)
# param_grid = dict(C=[1, 10, 100, 1000])
# param_grid = dict(C=C_range)
# param_grid = dict(C=[2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11,  2**13, 2**15])
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv)
# grid.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))    

# print(cross_val_score(grid, final_dframe['Prev3comp'].values.reshape(len(final_dframe['Prev3comp']),1), final_dframe['Label']))

fit = classifier.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])
y_pred = classifier.predict(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1),test['Label'].values.reshape(len(test['Label']),1) )))
accuracy = accuracy_score(y_pred,test['Label'].values.reshape(len(test['Label'])))
print(accuracy)