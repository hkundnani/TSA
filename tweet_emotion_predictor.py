import pandas as pd
import sys
from emotion_predictor import EmotionPredictor

model = EmotionPredictor(classification='poms', setting='mc')

if (len(sys.argv) < 4):
	print ('You have given wrong number of arguments.')
	print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:	
	in_file = sys.argv[1]
	predict_out_file = sys.argv[2]
	prob_out_file = sys.argv[3]
	file = pd.read_csv(in_file)
	tweets = file['tweet_content']
	
	predictions = model.predict_classes(tweets)
	predictions.to_csv(predict_out_file, sep=',', encoding='utf-8')

	probabilities = model.predict_probabilities(tweets)
	probabilities.to_csv(prob_out_file, sep=',', encoding='utf-8')