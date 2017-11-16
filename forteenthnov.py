# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:28:26 2017

@author: Ankita
"""

import pandas as pd
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
 







url1 = 'https://raw.githubusercontent.com/hkundnani/TSA/master/data/twitter_feeds_apple_processed.csv'
url2 = 'C:/Users/ARUN/Downloads/TSA-master/TSA-master/Applefinal.csv'
di = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')

djia.columns= ['Date', 'Adjvalue', 'Label']



df = di.iloc[1:2000]
djia = djia.iloc[97:]


djia['Date'] =  pd.to_datetime(djia['Date'])

djia['Date'] = djia['Date'].apply(lambda x: x.date())

djia['Date'] = djia['Date'].astype(str)

print(djia)

L = list()

df['"Tweet content"'].fillna('Neutral', inplace = True)
id_list = df['"Tweet Id"'].tolist()
date_list = df['Date'].tolist()
hour_list = df['Hour'].tolist()


tweets = df['"Tweet content"'].tolist()

'''for tweet in tweets:
    for i,row in df.iterrows():
 
            tweet = str(tweet)
            tweet = re.sub(r'[/|?=#@$%&1234567890:()!.]', "", tweet)
            # df.ix[row,'Tweet content'] = tweet
            df.set_value(i,'"Tweet content"',tweet)
        

            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(tweet)
            #print( ss['compound'] )
            # df.ix[rows, '"Polarity"'] = ss['compound']
            df.set_value(i,'"Polarity"',ss['compound'])
        
print (df)'''

hash_regex = re.compile(r"#(\w+)")
def hash_r(match):
	return '__HASH_'+match.group(1).upper()


user_regex = re.compile(r"@(\w+)")
def user_r(match):
	return '__USER'


url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")


rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_r(match):
	return match.group(1)+match.group(1)



punctuations = \
	[	#('',		['.', ] )	,\
		#('',		[',', ] )	,\
		#('',		['\'', '\"', ] )	,\
		('__PUNC_EXCL',		['!', '¡', ] )	,\
		('__PUNC_QUES',		['?', '¿', ] )	,\
		('__PUNC_ELLP',		['...', '…', ] )	,\
		
	]

def print_config(cfg):
	for (x, arr) in cfg:
		print x, '\t',
		for a in arr:
			print a, '\t',
		print ''


def print_punctuations():
	print_config(punctuations)


def escape_paren(arr):
	return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'


				

#For punctuation replacement
def punctuations_repl(match):
	text = match.group(0)
	repl = []
	for (key, parr) in punctuations :
		for punc in parr :
			if punc in text:
				repl.append(key)
	if( len(repl)>0 ) :
		return ' '+' '.join(repl)+' '
	else :
		return ' '

def processHashtags( 	text, subject='', query=[]):
	return re.sub( hash_regex, hash_r, text )

def processHandles( 	text, subject='', query=[]):
	return re.sub( user_regex, user_r, text )

def processUrls( 		text, subject='', query=[]):
	return re.sub( url_regex, ' __URL ', text )



def processPunctuations( text, subject='', query=[]):
	return re.sub( word_bound_regex , punctuations_repl, text )

def processRepeatings( 	text, subject='', query=[]):
	return re.sub( rpt_regex, rpt_r, text )

def processQueryTerm( 	text, subject='', query=[]):
	query_regex = "|".join([ re.escape(q) for q in query])
	return re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )


def processAll( 		text, subject='', query=[]):

	if(len(query)>0):
		query_regex = "|".join([ re.escape(q) for q in query])
		text = re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

	text = re.sub( hash_regex, hash_r, text )
	text = re.sub( user_regex, user_r, text )
	text = re.sub( url_regex, ' __URL ', text )

	

	text = text.replace('\'','')
	

	text = re.sub( word_bound_regex , punctuations_repl, text )
	text = re.sub( rpt_regex, rpt_r, text )

	return text
 
comp_list = []
tweet_list = []
neg_list=[]
pos_list=[]
neu_list=[]
label_list =[]
     
for tweet in tweets: 
       
       tweet = str(tweet)
       tweet1 = processHashtags(tweet)
       tweet2 = processHandles(tweet1)
       tweet3 = processUrls(tweet2)
       
       tweet5 = processPunctuations(tweet3)
       tweet6 = processRepeatings(tweet5)
       tweet7 = processAll(tweet6)
       tweet_list.append(str(tweet7))
       
      # print(tweet7)  
       sid = SentimentIntensityAnalyzer()
       ss = sid.polarity_scores(tweet7)
       
       if ss['compound']>0:
           Label = 1
       elif ss['compound']<0:
           Label = 0
       else:
           Label =-1
           
       comp_list.append(str(ss['compound']))
       neg_list.append(str(ss['neg']))
       pos_list.append(str(ss['pos']))
       neu_list.append(str(ss['neu']))
       label_list.append(str(Label))
       
       final = zip(id_list, date_list, hour_list, tweet_list, neg_list, pos_list,neu_list, comp_list, label_list)
       testfinal = zip(date_list, comp_list)
       labels= ['ID', 'Date', 'Hour','Tweet', 'Negative','Positive','Neutral','Compound', 'Label']
       finaldf = pd.DataFrame.from_records(final, columns=labels)
#print(finaldf.head(20))

label = ['Date', 'Compound']
testdf = pd.DataFrame.from_records(testfinal, columns = label)
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

train,test = train_test_split(result,test_size=0.2,random_state=42)
non_decrease = train[train['Label']==1]
decrease = train[train['Label']==0]
print(len(non_decrease)/len(df))


train_comp_list = train['Compound'].tolist()
test_comp_list = test['Compound'].tolist()

#dense_features=train_comp_list.toarray()
#dense_test= test_comp_list.toarray()
Accuracy=[]
Model=[]

   
    fity = classifier.fit(train['Compound'].reshape(8,1),train['Label'].reshape(8,1))
    pred = fity.predict(test['Compound'].reshape(2,1))
    prob = fity.predict_proba(test['Compound'].reshape(2,1))#[:,1]
    
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))







