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

nltk.download()



url = 'https://raw.githubusercontent.com/hkundnani/TSA/master/data/twitter_feeds_apple_processed.csv'
di = pd.read_csv(url, header = 0,engine = 'python', sep = '\,')

df = di.head(50)
df['"Tweet content"'].fillna('Neutral', inplace = True)
tweets = df['"Tweet content"'].tolist()
for i,row in df.iterrows():
        for tweet in tweets:
            tweet = str(tweet)
            tweet = re.sub(r'[/|?=#@$%&1234567890:()!.]', "", tweet)
           # df.ix[row,'Tweet content'] = tweet
            df.set_value(i,'"Tweet content"',tweet)
        

            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(tweet)
        #print( ss['compound'] )
           # df.ix[rows, '"Polarity"'] = ss['compound']
           df.set_value(i,'"Polarity"',ss['compound'])
        
print (df.head(50))
    
