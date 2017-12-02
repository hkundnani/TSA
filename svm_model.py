from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import pylab as pl
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

model = svm.SVC(kernel='rbf', C=1000000000.0, gamma=0.01)

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

c =[]

complist = result['Compound'].tolist()
for i in range(len(complist)):
    if i < 3:
        c.append(0.0)
    if i > 3:
        val= (complist[i-1]+complist[i-2]+complist[i-3])/3
        c.append(val)
        
date_list = result['Date'].tolist()
adj_list = result['Adjvalue'].tolist()
label_list = result['Label'].tolist()
final_list = zip(date_list, adj_list, label_list, complist, c)
label = ['Date', 'Adjvalue', 'Label', 'Compound', 'Prev3comp']
final_dframe = pd.DataFrame.from_records(final_list, columns = label)
result = final_dframe[final_dframe.Prev3comp != 0.0000]

# print (result)

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# grid.fit(result['Prev3comp'].values.reshape(len(result['Prev3comp']),1), result['Label'])

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

train,test = train_test_split(result,test_size=0.2, random_state=42)# about random_state


# compound_list = train['Compound'].tolist()
# label_list = train['Compound'].tolist()

fit = model.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])

y_pred = model.predict(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1))

print(y_pred)
print(test['Label'])

print (model.score(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1), test['Label']))

# pl.plot(train['Compound'].values, train['Label'].values)
# pl.show()
# print (confusion_matrix(test['Label'], y_pred))
# print (classification_report(test['Label'], y_pred))