from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import pylab as pl
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# model = svm.SVC(kernel='rbf', C=1000000.0, gamma=10.0)
model = LogisticRegression(C=8192)

column_names = ['Date', 'adj_close', 'label']
djia_file = pd.read_csv('processed_djia/AAPL.csv', header=None)
djia_file.columns = column_names

tweet_file = pd.read_csv('tweets_ekman_scores/prob_AAPL_ekman.csv')
# tweet_file = pd.read_csv('tweets_emotion_class_probability/prob_AAPL_emotion.csv')
i = 1
djia_processed = []
while i < djia_file.shape[0]:
	djia_processed.append({
		'Date': djia_file['Date'][i],
		'adj_close': djia_file['adj_close'][i] - djia_file['adj_close'][i-1],
		'Label': djia_file['label'][i]
		})
	i += 1
djia_pro = pd.DataFrame(djia_processed)

# Add date to the mood probability data 
# Please change the file based on whichever company date one is passing
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

# print (result)
# # create a mesh to plot in
# result_compound[:, 1] = 0
# x_min, x_max = result_compound[:, 0].min() - 1, result_compound[:, 0].max() + 1
# y_min, y_max = result_compound[:, 1].min() - 1, result_compound[:, 1].max() + 1
# h = (x_max / x_min) / 100
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# # x_plot = np.c_[xx.ravel(), yy.ravel()]
# # print(x_plot)

# Use below code for POMS
# result_feature = result[['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()
# result_label = result['Label'].values.tolist()
# train,test = train_test_split(result, test_size=0.2, random_state=42)# about random_state
# train_label_list = train['Label'].values.tolist()
# test_label_list = test['Label'].values.tolist()
# train_mood_list = train[['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()
# test_mood_list = test[['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()

# Use below code for Ekman
result_feature = result[['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
result_label = result['Label'].values.tolist()
train,test = train_test_split(result, test_size=0.2, random_state=42)# about random_state
train_label_list = train['Label'].values.tolist()
test_label_list = test['Label'].values.tolist()
train_mood_list = train[['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
test_mood_list = test[['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=[1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5], C=[1, 10, 100, 1000])
# param_grid = dict(gamma=gamma_range, C=C_range)
# param_grid = dict(gamma=[2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], C=[2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15])
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv)
# grid.fit(train_mood_list, train_label_list)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

# print(cross_val_score(grid, result_feature, result_label))


fit = model.fit(train_mood_list, train_label_list)

y_pred = model.predict(test_mood_list)

print (model.score(test_mood_list, test_label_list))
# print (confusion_matrix(test['Label'], y_pred))
# print (classification_report(test['Label'], y_pred))

#AAPL 0.01
#MSFT 0.01
#CSCO 10000
#INTC 0.01

#AAPL 1
#MSFT 1
#CSCO 1000
#INTC 1

#AAPL 0.03125
#MSFT 128
#CSCO 32768
#INTC 0.03125