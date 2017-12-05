import numpy as np
import pandas as pd
import pylab as pl
import sys
from sklearn.preprocessing import MinMaxScaler

if (len(sys.argv) < 3):
	print ('You have given wrong number of arguments.')
	print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:
	column_names = ['Date', 'adj_close', 'label']
	djia_file = pd.read_csv(sys.argv[1], header=None)
	djia_file.columns = column_names

	tweet_file = pd.read_csv(sys.argv[2])
	i = 1
	djia_processed = []
	while i < djia_file.shape[0]:
		djia_processed.append({
			'Date': djia_file['Date'][i],
			'adj_close': djia_file['adj_close'][i] - djia_file['adj_close'][i-1],
			'label': djia_file['label'][i]
			})
		i += 1
	djia_pro = pd.DataFrame(djia_processed)

	tweets = tweet_file.groupby('Date', as_index=False)['Compound'].mean()
	tweets['Compound'] = tweets['Compound'].rolling(window = 3).mean()
	tweets = tweets[pd.notnull(tweets['Compound'])]

	result = pd.merge(djia_pro, tweets, how = 'outer', on=['Date'])
	result = result[pd.notnull(result['Compound'])]
	scaler = MinMaxScaler(feature_range=(0, 1))
	# print (result)

	result['Date'] = pd.to_datetime(result['Date'])
	mask = (result['Date'] > '2016-05-15') & (result['Date'] <= '2016-05-22')
	result = result.loc[mask]
	result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
	
	result['adj_close'] = scaler.fit_transform(result['adj_close'].values.reshape(len(result['adj_close']), 1))
	result['Compound'] = scaler.fit_transform(result['Compound'].values.reshape(len(result['Compound']), 1))
	 
	print (result)
	
	# file3['Date'] = (pd.to_datetime(file1['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')
	# file4 = tweet_file.groupby('Date', as_index=False)['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'].mean()
	# file_sc = file4.tail(-1)

	# pl.subplot(2, 1, 1)
	# pl.ylim(0, 1)
	# pl.plot(file_pb['Date'], file_pb['Anger'], 'r')
	# pl.plot(file_pb['Date'], file_pb['Depression'], 'b')
	# pl.plot(file_pb['Date'], file_pb['Fatigue'], 'g')
	# pl.plot(file_pb['Date'], file_pb['Vigour'], 'm')
	# pl.plot(file_pb['Date'], file_pb['Tension'], 'k')
	# pl.plot(file_pb['Date'], file_pb['Confusion'], 'c')

	# pl.subplot(2, 1, 2)
	pl.ylim(0, 1)
	pl.plot(result['Date'], result['adj_close'], 'r')
	pl.plot(result['Date'], result['Compound'], 'b')
	pl.xticks(rotation=45)
	# pl.ylim(0, 1)
	# pl.plot(file_sc['Date'], file_sc['Anger'], 'r')
	# pl.plot(file_sc['Date'], file_sc['Disgust'], 'b')
	# pl.plot(file_sc['Date'], file_sc['Fear'], 'g')
	# pl.plot(file_sc['Date'], file_sc['Joy'], 'g')
	# pl.plot(file_sc['Date'], file_sc['Sadness'], 'b')
	# pl.plot(file_sc['Date'], file_sc['Surprise'], 'c')
	
	pl.show()