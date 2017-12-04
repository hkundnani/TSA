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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


model = svm.SVC(kernel='rbf', C=1000000000.0, gamma=0.01) #Apple
#model = svm.SVC(kernel='rbf', C=0.01, gamma=0.000000001) #Intel
#model = svm.SVC(kernel='rbf', C=100000.0, gamma=1000.0) #MSFT


<<<<<<< Updated upstream
model = svm.SVC(kernel='rbf', C=1, gamma=0.0001)
=======
>>>>>>> Stashed changes

url1 = 'tweets_sentiment_score/MSFT_sentiment_score.csv'
url2 = 'processed_djia/MSFT.csv'
df = pd.read_csv(url1, header = 0,engine = 'python', sep = '\,')
djia = pd.read_csv(url2, header = None,engine = 'python', sep = '\,')

djia.columns= ['Date', 'Adjvalue', 'Label']

label = ['Date', 'Compound']
testdf = pd.DataFrame.from_records(df, columns = label)

testdf = testdf[testdf.Compound != 0.0000]
newtestdf = testdf.groupby('Date').mean().reset_index()

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
<<<<<<< Updated upstream
train, test = train_test_split(result, test_size=0.2, random_state=42)

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=[1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5], C=[1])
# param_grid = dict(gamma=gamma_range, C=[1])
# param_grid = dict(gamma=[2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], C=[1])
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# grid.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
=======
#
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
#grid.fit(result['Prev3comp'].values.reshape(len(result['Prev3comp']),1), result['Label'])
#
#print("The best parameters are %s with a score of %0.2f"
#   % (grid.best_params_, grid.best_score_))
>>>>>>> Stashed changes

# print(cross_val_score(grid, final_dframe['Prev3comp'].values.reshape(len(final_dframe['Prev3comp']),1), final_dframe['Label']))

fit = model.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])

y_pred = model.predict(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1))

<<<<<<< Updated upstream
print (model.score(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1), test['Label']))
=======
#print(y_pred)
#print(test['Label'])

#print (model.score(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1), test['Label']))
>>>>>>> Stashed changes

# print (confusion_matrix(test['Label'], y_pred))
# print (classification_report(test['Label'], y_pred))

y_test = test['Label'].reshape(len(test['Label']),1)
print (y_test.ravel())
print (y_pred)
fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_pred)
print (fpr)
print (tpr)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
