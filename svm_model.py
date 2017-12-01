from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import pylab as pl
from sklearn.metrics import classification_report,confusion_matrix

model = svm.SVC(kernel='rbf' , C=1, gamma=1)

url1 = 'tweets_sentiment_score/AAPL_sentiment_score.csv'
url2 = 'processed_djia/AAPL.csv'
df = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')

djia.columns= ['Date', 'Adjvalue', 'Label']

label = ['Date', 'Compound']
testdf = pd.DataFrame.from_records(df, columns = label)
# testdf['Date'].dropna(inplace = True)
# testdf[['Compound']] = testdf[['Compound']].astype(float)

testdf = testdf[testdf.Compound != 0.0000]
newtestdf = testdf.groupby('Date').mean().reset_index()

# newtestdf['Date'] = newtestdf['Date'].astype(str)

result = pd.merge(djia, newtestdf, how = 'outer', on=['Date'])

result['Compound'].fillna(0.0, inplace = True)
result = result[result.Compound != 0.0000]
# result['Label'].fillna(0.0, inplace = True)
# result['Label'].fillna(0.0, inplace = True)
# result.dropna(axis=0, inplace = True)


train,test = train_test_split(result,test_size=0.2, random_state=42)# about random_state


# compound_list = train['Compound'].tolist()
# label_list = train['Compound'].tolist()

fit = model.fit(train['Compound'].values.reshape(len(train['Compound']),1), train['Label'])

y_pred = model.predict(test['Label'].values.reshape(len(test['Label']),1))

print (model.score(test['Compound'].values.reshape(len(test['Compound']),1), test['Label']))
# print (confusion_matrix(test['Label'], y_pred))
# print (classification_report(test['Label'], y_pred))