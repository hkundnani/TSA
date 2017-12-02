from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import pylab as pl
from sklearn.metrics import classification_report,confusion_matrix

model = svm.SVC(kernel='rbf' , C=1, gamma=1)

url1 = 'tweets_emotion_class_probability/prob_MSFT_emotion.csv'
url2 = 'processed_djia/MSFT.csv'
df = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')

djia.columns= ['Date', 'Adjvalue', 'Label']

# Add date to the mood probability data 
file1 = pd.read_csv('processed_tweets/CSCO_final_tweets.csv')
df['Date'] = (pd.to_datetime(file1['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')

# Group by date on tweet moods data
newtestdf = df.groupby('Date', as_index=False)['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion'].mean()

result = pd.merge(djia, newtestdf, how = 'outer', on=['Date'])

# Remove null values from the merged result
result = result[pd.notnull(result['Anger'])]

train,test = train_test_split(result,test_size=0.2, random_state=42)# about random_state
train_label_list = train['Label'].values.tolist()
test_label_list = test['Label'].values.tolist()
train_mood_list = train[['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()
test_mood_list = test[['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()

fit = model.fit(train_mood_list, train_label_list)

y_pred = model.predict(test_mood_list)

print (model.score(test_mood_list, test_label_list))
# # print (confusion_matrix(test['Label'], y_pred))
# # print (classification_report(test['Label'], y_pred))