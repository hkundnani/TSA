import sys
import pandas as pd
import nltk
from nltk import tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if (len(sys.argv) < 3):
  print ('You have given wrong number of arguments.')
  print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:
  in_file = sys.argv[1]
  out_file = sys.argv[2]
  di = pd.read_csv(in_file, header = 0,engine = 'python', sep = '\,')
  di['tweet_content'].fillna('Neutral', inplace = True)
  date_list = di['Date'].tolist()
  tweets = di['tweet_content'].tolist()
  comp_list = []
  tweet_list = []
  neg_list=[]
  pos_list=[]
  neu_list=[]
  label_list =[]
       
  for tweet in tweets: 
           
         sid = SentimentIntensityAnalyzer()
         ss = sid.polarity_scores(tweet)
         
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
         
         final = zip(date_list, tweets, neg_list, pos_list,neu_list, comp_list, label_list)
         testfinal = zip(date_list, comp_list)
         labels= ['Date', 'Tweet', 'Negative', 'Positive', 'Neutral', 'Compound', 'Label']
         finaldf = pd.DataFrame.from_records(final, columns=labels)
  finaldf.to_csv(out_file, sep=',', encoding='utf-8')