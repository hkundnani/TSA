import numpy as np
import pandas as pd
import pylab as pl
import sys
from sklearn.preprocessing import MinMaxScaler

djia = pd.read_csv('processed_djia/CSCO.csv')
moods_sent = pd.read_csv('tweets_sentiment_score/CSCO_sentiment_score.csv')
moods_poms = pd.read_csv('tweets_emotion_class_probability/prob_CSCO_emotion.csv')
moods_ekman = pd.read_csv('tweets_ekman_scores/prob_CSCO_ekman.csv')

#Add columns to djia dataframe
column_names = ['Date', 'adj_close', 'label']
djia.columns = column_names

i = 1
djia_processed = []
while i < djia.shape[0]:
	djia_processed.append({
		'Date': djia['Date'][i],
		'adj_close': djia['adj_close'][i] - djia['adj_close'][i-1],
		'label': djia['label'][i]
		})
	i += 1
djia_pro = pd.DataFrame(djia_processed)

# Add date to the mood probability data 
moods_poms['Date'] = (pd.to_datetime(moods_sent['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')
moods_ekman['Date'] = (pd.to_datetime(moods_sent['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')

# Group by date on tweet moods data
# Use below line for POMS
moods_poms_mean = moods_poms.groupby('Date', as_index=False)['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion'].mean()
# Use below line for Ekman
moods_ekman_mean = moods_ekman.groupby('Date', as_index=False)['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'].mean()
# Use below line for sentiment
# moods_sent_mean = moods_sent.groupby('Date', as_index=False)['Compound'].mean()

result = pd.merge(djia_pro, moods_poms_mean, how = 'outer', on=['Date'])
result = pd.merge(result, moods_ekman_mean, how = 'outer', on=['Date'])
result = result[pd.notnull(result['Anger_x'])]
print (result)

# Remove null values from the merged result
# Use below code for POMS
result['Anger_x'] = result['Anger_x'].rolling(window = 3).mean()
result['Depression'] = result['Depression'].rolling(window = 3).mean()
result['Fatigue'] = result['Fatigue'].rolling(window = 3).mean()
result['Vigour'] = result['Vigour'].rolling(window = 3).mean()
result['Tension'] = result['Tension'].rolling(window = 3).mean()
result['Confusion'] = result['Confusion'].rolling(window = 3).mean()

# Use below code for Ekman
result['Anger_y'] = result['Anger_y'].rolling(window = 3).mean()
result['Disgust'] = result['Disgust'].rolling(window = 3).mean()
result['Fear'] = result['Fear'].rolling(window = 3).mean()
result['Joy'] = result['Joy'].rolling(window = 3).mean()
result['Sadness'] = result['Sadness'].rolling(window = 3).mean()
result['Surprise'] = result['Surprise'].rolling(window = 3).mean()

result = result[pd.notnull(result['Anger_x'])]

# Scale the adj_close values
scaler = MinMaxScaler(feature_range=(0, 1))
result['adj_close'] = scaler.fit_transform(result['adj_close'].values.reshape(len(result['adj_close']), 1))
result['Anger_x'] = scaler.fit_transform(result['Anger_x'].values.reshape(len(result['Anger_x']), 1))
result['Depression'] = scaler.fit_transform(result['Depression'].values.reshape(len(result['Depression']), 1))
result['Fatigue'] = scaler.fit_transform(result['Fatigue'].values.reshape(len(result['Fatigue']), 1))
result['Vigour'] = scaler.fit_transform(result['Vigour'].values.reshape(len(result['Vigour']), 1))
result['Tension'] = scaler.fit_transform(result['Tension'].values.reshape(len(result['Tension']), 1))
result['Confusion'] = scaler.fit_transform(result['Confusion'].values.reshape(len(result['Confusion']), 1))
result['Anger_y'] = scaler.fit_transform(result['Anger_y'].values.reshape(len(result['Anger_y']), 1))
result['Disgust'] = scaler.fit_transform(result['Disgust'].values.reshape(len(result['Disgust']), 1))
result['Fear'] = scaler.fit_transform(result['Fear'].values.reshape(len(result['Fear']), 1))
result['Joy'] = scaler.fit_transform(result['Joy'].values.reshape(len(result['Joy']), 1))
result['Sadness'] = scaler.fit_transform(result['Sadness'].values.reshape(len(result['Sadness']), 1))
result['Surprise'] = scaler.fit_transform(result['Surprise'].values.reshape(len(result['Surprise']), 1))

result['Date'] = pd.to_datetime(result['Date'])
mask = (result['Date'] > '2016-05-15') & (result['Date'] <= '2016-05-22')
result = result.loc[mask]
result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
print (result)


ax1 = pl.subplot(2, 1, 1)
ax2 = pl.subplot(2, 1, 2)
pl.suptitle('Moods vs DJIA value for Cisco tweets')
pl.title('POMS')
ax1.plot(result['Date'], result['Anger_x'], 'r', label='Anger')
ax1.plot(result['Date'], result['Depression'], 'b', label='Depression')
ax1.plot(result['Date'], result['Fatigue'], 'g', label='Fatigue')
ax1.plot(result['Date'], result['Vigour'], 'm', label='Vigour')
ax1.plot(result['Date'], result['Tension'], 'k', label='Tension')
ax1.plot(result['Date'], result['Confusion'], 'c', label='Confusion')
ax1.plot(result['Date'], result['adj_close'], 'y', label='DJIA')
ax1.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# pl.xticks(rotation=45)

ax2.plot(result['Date'], result['Anger_y'], 'r', label='Anger')
ax2.plot(result['Date'], result['Disgust'], 'c', label='Disgust')
ax2.plot(result['Date'], result['Fear'], 'k', label='Fear')
ax2.plot(result['Date'], result['Joy'], 'm', label='Joy')
ax2.plot(result['Date'], result['Sadness'], 'b', label='Sadness')
ax2.plot(result['Date'], result['Surprise'], 'g', label='Surprise')
ax2.plot(result['Date'], result['adj_close'], 'y', label='DJIA')
ax2.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# pl.xticks(rotation=45)	

pl.show()