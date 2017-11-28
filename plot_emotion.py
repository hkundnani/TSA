import numpy as np
import pandas as pd
import pylab as pl
import sys

if (len(sys.argv) < 3):
	print ('You have given wrong number of arguments.')
	print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:
	# axes = pl.gca()	
	# in_file = sys.argv[1]
	file = pd.read_csv(sys.argv[1])
	file1 = pd.read_csv('processed_tweets/CSCO_final_tweets.csv')
	file['Date'] = (pd.to_datetime(file1['Date'], errors='coerce')).dt.strftime('%Y-%m-%d')
	file2 = file.groupby('Date', as_index=False)['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion'].mean()
	# print (file2)
	# file2.to_csv('processed_emotions.csv', sep=',', encoding='utf-8')
	file_pb = file2.tail(10)
	file3 = pd.read_csv(sys.argv[2])
	file4 = file3.groupby('Date', as_index=False)['Compound'].mean()
	file_sc = file4.tail(10)

	pl.subplot(2, 1, 1)
	pl.ylim(0, 1)
	pl.plot(file_pb['Date'], file_pb['Anger'], 'r')
	pl.plot(file_pb['Date'], file_pb['Depression'], 'b')
	pl.plot(file_pb['Date'], file_pb['Fatigue'], 'g')
	pl.plot(file_pb['Date'], file_pb['Vigour'], 'm')
	pl.plot(file_pb['Date'], file_pb['Tension'], 'k')
	pl.plot(file_pb['Date'], file_pb['Confusion'], 'c')

	pl.subplot(2, 1, 2)
	pl.ylim(-1, 1)
	pl.plot(file_sc['Date'], file_sc['Compound'], 'y')
	
	pl.show()