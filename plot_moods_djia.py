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

	# Add date to the mood probability data 
	file1 = pd.read_csv('processed_tweets/AAPL_final_tweets.csv')
	tweet_file['Date'] = (pd.to_datetime(file1['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')

	# Group by date on tweet moods data
	# Use below line for POMS
	# newtestdf = tweet_file.groupby('Date', as_index=False)['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion'].mean()
	# Use below line for Ekman
	newtestdf = tweet_file.groupby('Date', as_index=False)['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'].mean()

	result = pd.merge(djia_pro, newtestdf, how = 'outer', on=['Date'])

	# Remove null values from the merged result
	# Use below code for POMS
	# result = result[pd.notnull(result['Anger'])]
	# result['Anger'] = result['Anger'].rolling(window = 3).mean()
	# result['Depression'] = result['Depression'].rolling(window = 3).mean()
	# result['Fatigue'] = result['Fatigue'].rolling(window = 3).mean()
	# result['Vigour'] = result['Vigour'].rolling(window = 3).mean()
	# result['Tension'] = result['Tension'].rolling(window = 3).mean()
	# result['Confusion'] = result['Confusion'].rolling(window = 3).mean()
	# result = result[pd.notnull(result['Anger'])]

	# Use below code for Ekman
	result = result[pd.notnull(result['Anger'])]
	result['Anger'] = result['Anger'].rolling(window = 3).mean()
	result['Disgust'] = result['Disgust'].rolling(window = 3).mean()
	result['Fear'] = result['Fear'].rolling(window = 3).mean()
	result['Joy'] = result['Joy'].rolling(window = 3).mean()
	result['Sadness'] = result['Sadness'].rolling(window = 3).mean()
	result['Surprise'] = result['Surprise'].rolling(window = 3).mean()
	result = result[pd.notnull(result['Anger'])]

	# Scale the adj_close values
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaled_djia = scaler.fit_transform(result['adj_close'].values.reshape(len(result['adj_close']), 1))
	# print (result)
	
	# pl.subplot(2, 1, 1)
	# pl.ylim(0, 1)
	# Use below code for POMS
	# pl.plot(result['Date'], result['Anger'], 'r')
	# pl.plot(result['Date'], result['Depression'], 'b')
	# pl.plot(result['Date'], result['Fatigue'], 'g')
	# pl.plot(result['Date'], result['Vigour'], 'm')
	# pl.plot(result['Date'], result['Tension'], 'k')
	# pl.plot(result['Date'], result['Confusion'], 'c')
	# pl.plot(result['Date'], rescaled_djia[:, 0], 'y')

	# pl.subplot(2, 1, 2)
	# pl.ylim(-0.5, 1.5)
	# Use below code for Ekman
	pl.xticks(rotation=45)
	# pl.ylim(0, 1)
	pl.plot(result['Date'], result['Anger'], 'r')
	pl.plot(result['Date'], result['Disgust'], 'c')
	pl.plot(result['Date'], result['Fear'], 'k')
	pl.plot(result['Date'], result['Joy'], 'm')
	pl.plot(result['Date'], result['Sadness'], 'b')
	pl.plot(result['Date'], result['Surprise'], 'g')
	pl.plot(result['Date'], rescaled_djia[:, 0], 'y')	
	
	pl.show()