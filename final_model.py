import pip
pip.main(["install", "scikit-learn"])
pip.main(["install", "pandas"])
pip.main(["install", "scipy"])
pip.main(["install", "matplotlib"])

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd

# Function to read the CSV file
def read_file(file_name):
	return pd.read_csv(file_name, header = 0,engine = 'python', sep = '\,')

# Function to compute logistic regression and SVM for Compound and Label
def model_compound(data, model_val, param):
	if (model_val == 'logreg'):
		model = LogisticRegression(C=param['C'])
	elif (model_val == 'svm'):
		model = svm.SVC(kernel='rbf', C=param['C'], gamma=param['gamma'])

	train, test = train_test_split(data, test_size=0.2, random_state=42)
	fit = model.fit(train['Prev3comp'].values.reshape(len(train['Prev3comp']),1), train['Label'])
	y_pred = model.predict(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1))
	false_positive_rate, true_positive_rate, thresholds = roc_curve(test['Label'], y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	return {
		'accuracy_score': model.score(test['Prev3comp'].values.reshape(len(test['Prev3comp']),1), test['Label']),
		'false_positive_rate': false_positive_rate,
		'true_positive_rate': true_positive_rate,
		'roc_auc': roc_auc
	} 

# Function to compute logistic regression and SVM for POMS moods and Label
def model_poms(result, model_val, param):
	if (model_val == 'logreg'):
		model = LogisticRegression(C=param['C'])
	elif (model_val == 'svm'):
		model = svm.SVC(kernel='rbf', C=param['C'], gamma=param['gamma'])

	train,test = train_test_split(result, test_size=0.2, random_state=42)
	train_label_list = train['label'].values.tolist()
	test_label_list = test['label'].values.tolist()
	train_mood_list = train[['Anger_x', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()
	test_mood_list = test[['Anger_x', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']].values.tolist()

	fit = model.fit(train_mood_list, train_label_list)

	y_pred = model.predict(test_mood_list)

	false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label_list, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	return {
		'accuracy_score': model.score(test_mood_list, test_label_list),
		'false_positive_rate': false_positive_rate,
		'true_positive_rate': true_positive_rate,
		'roc_auc': roc_auc
	}

# Function to compute logistic regression and SVM for Ekman moods and Label
def model_ekman(result, model_val, param):
	if (model_val == 'logreg'):
		model = LogisticRegression(C=param['C'])
	elif (model_val == 'svm'):
		model = svm.SVC(kernel='rbf', C=param['C'], gamma=param['gamma'])

	train,test = train_test_split(result, test_size=0.2, random_state=42)
	train_label_list = train['label'].values.tolist()
	test_label_list = test['label'].values.tolist()
	train_mood_list = train[['Anger_y', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
	test_mood_list = test[['Anger_y', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
	fit = model.fit(train_mood_list, train_label_list)

	y_pred = model.predict(test_mood_list)

	false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label_list, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	return {
		'accuracy_score': model.score(test_mood_list, test_label_list),
		'false_positive_rate': false_positive_rate,
		'true_positive_rate': true_positive_rate,
		'roc_auc': roc_auc
	}	

# Function to draw ROC curve
def draw_roc_curve(file_data, title):
	for index, data in enumerate(file_data):
		for key, value in data.items():
			plt.subplot(2, 2, index+1)
			plt.title('Receiver Operating Characteristic for ' + companies[key])
			plt.plot(value['false_positive_rate'], value['true_positive_rate'], 'b', label='AUC = %0.2f'% value['roc_auc'])
			plt.legend(loc='lower right')
			plt.plot([0,1],[0,1],'r--')
			plt.xlim([-0.1,1.2])
			plt.ylim([-0.1,1.2])
			plt.ylabel('True Positive Rate')
			plt.xlabel('False Positive Rate')
	plt.suptitle(title)
	plt.tight_layout()
	plt.show()

files_compund = ['final_processed_compound/AAPL.csv', 'final_processed_compound/MSFT.csv', 'final_processed_compound/CSCO.csv', 'final_processed_compound/INTC.csv'] 
files_moods = ['final_processed_moods/AAPL.csv', 'final_processed_moods/MSFT.csv', 'final_processed_moods/CSCO.csv', 'final_processed_moods/INTC.csv']
compound_param = [{'C': 100.0, 'gamma': 1.0}, {'C': 100.0, 'gamma': 100.0}, {'C': 100.0, 'gamma': 10.0}, {'C': 10.0, 'gamma': 1.0}]
moods_param = [{'C': 1000000.0, 'gamma': 10.0}, {'C': 100.0, 'gamma': 100.0}, {'C': 100.0, 'gamma': 10.0}, {'C': 10.0, 'gamma': 1.0}]
companies = {
	'AAPL': 'Apple',
	'CSCO': 'Cisco',
	'MSFT': 'Microsoft',
	'INTC': 'Intel'
}
result_logreg_compound = []
result_svm_compound = []
result_logreg_poms = []
result_logreg_ekman = []
result_svm_poms = []
result_svm_ekman = []

for index, file in enumerate(files_compund):
	file_name = file[25:29]
	data = read_file(file)
	result_logreg_compound.append({
		file_name: model_compound(data, 'logreg', compound_param[index])
		})
	result_svm_compound.append({
		file_name: model_compound(data, 'svm', compound_param[index])
		})

for index, file in enumerate(files_moods):
	file_name = file[22:26]
	data = read_file(file)
	result_logreg_poms.append({
		file_name: model_poms(data, 'logreg', moods_param[index])
		})
	result_svm_poms.append({
		file_name: model_poms(data, 'svm', moods_param[index])
		})

for findex, file in enumerate(files_moods):
	file_name = file[22:26]
	data = read_file(file)
	result_logreg_ekman.append({
		file_name: model_ekman(data, 'logreg', moods_param[index])
		})
	result_svm_ekman.append({
		file_name: model_ekman(data, 'svm', moods_param[index])
		})
	
print ("%20s %20s %20s %20s %20s" % ('Company', 'Algorithm', 'Accuracy(Compound)', 'Accuracy(POMS)', 'Accuracy(Ekman)'))
print ("%20s %20s %20s %20s %20s" % ('Apple', 'Logistic Regression', result_logreg_compound[0]['AAPL']['accuracy_score'], result_logreg_poms[0]['AAPL']['accuracy_score'], result_logreg_ekman[0]['AAPL']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % (' ', 'SVM', result_svm_compound[0]['AAPL']['accuracy_score'], result_svm_poms[0]['AAPL']['accuracy_score'], result_svm_ekman[0]['AAPL']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % ('Microsoft', 'Logistic Regression', result_logreg_compound[1]['MSFT']['accuracy_score'], result_logreg_poms[1]['MSFT']['accuracy_score'], result_logreg_ekman[1]['MSFT']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % (' ', 'SVM', result_svm_compound[1]['MSFT']['accuracy_score'], result_svm_poms[1]['MSFT']['accuracy_score'], result_svm_ekman[1]['MSFT']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % ('Cisco', 'Logistic Regression', result_logreg_compound[2]['CSCO']['accuracy_score'], result_logreg_poms[2]['CSCO']['accuracy_score'], result_logreg_ekman[2]['CSCO']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % (' ', 'SVM', result_svm_compound[2]['CSCO']['accuracy_score'], result_svm_poms[2]['CSCO']['accuracy_score'], result_svm_ekman[2]['CSCO']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % ('Intel', 'Logistic Regression', result_logreg_compound[3]['INTC']['accuracy_score'], result_logreg_poms[3]['INTC']['accuracy_score'], result_logreg_ekman[3]['INTC']['accuracy_score']))
print ("%20s %20s %20s %20s %20s" % (' ', 'SVM', result_svm_compound[3]['INTC']['accuracy_score'], result_svm_poms[3]['INTC']['accuracy_score'], result_svm_ekman[3]['INTC']['accuracy_score']))
draw_roc_curve(result_logreg_compound, 'ROC curve for Logistic Regression Model - Compound')
draw_roc_curve(result_svm_compound, 'ROC curve for SVM Model - Compound')
draw_roc_curve(result_logreg_poms, 'ROC curve for Logistic Regression Model - POMS')
draw_roc_curve(result_svm_poms, 'ROC curve for SVM Model - POMS')
draw_roc_curve(result_logreg_ekman, 'ROC curve for Logistic Regression Model - Ekman')
draw_roc_curve(result_svm_ekman, 'ROC curve for SVM Model - Ekman')