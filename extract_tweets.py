import sys
import pandas
import csv
import re

if (len(sys.argv) < 3):
	print ('You have given wrong number of arguments.')
	print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:	
	in_file = sys.argv[1]
	out_file = sys.argv[2]
	file = pandas.read_excel(in_file)
	FORMAT = ['Date', 'Tweet content']
	tweet_data = file[FORMAT]

	punctuations = \
		[	#('',		['.', ] )	,\
			#('',		[',', ] )	,\
			#('',		['\'', '\"', ] )	,\
			('__PUNC_EXCL',		['!', '¡', ] )	,\
			('__PUNC_QUES',		['?', '¿', ] )	,\
			('__PUNC_ELLP',		['...', '…', ] )	,\
		]

	hash_regex = re.compile(r"#(\w+)")
	def hash_r(match):
		return '__HASH_'+match.group(1).upper()

	def processHashtags(text):
		return re.sub(hash_regex, hash_r, text)

	user_regex = re.compile(r"@(\w+)")
	def user_r(match):
		return '__USER'

	def processHandles(text):
		return re.sub(user_regex, user_r, text)

	url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")
	def processUrls(text):
		return re.sub( url_regex, ' __URL ', text )

	# Spliting by word boundaries
	word_bound_regex = re.compile(r"\W+")

	#For punctuation replacement
	def punctuations_repl(match):
		text = match.group(0)
		repl = []
		for (key, parr) in punctuations :
			for punc in parr :
				if punc in text:
					repl.append(key)
		if( len(repl)>0 ) :
			return ' '+' '.join(repl)+' '
		else :
			return ' '

	def processPunctuations(text):
		return re.sub(word_bound_regex, punctuations_repl, text)

	rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
	def rpt_r(match):
		return match.group(1)+match.group(1)

	def processRepeatings(text):
		return re.sub(rpt_regex, rpt_r, text)

	def processAll(text, subject='', query=[]):

		if(len(query)>0):
			query_regex = "|".join([ re.escape(q) for q in query])
			text = re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

		text = re.sub( hash_regex, hash_r, text )
		text = re.sub( user_regex, user_r, text )
		text = re.sub( url_regex, ' __URL ', text )

		text = text.replace('\'','')
		
		text = re.sub( word_bound_regex , punctuations_repl, text )
		text = re.sub( rpt_regex, rpt_r, text )

		return text

	tweet_list = []
	for tweet in tweet_data['Tweet content']:
		tweet1 = processAll(processRepeatings(processPunctuations(processUrls(processHandles(processHashtags(tweet))))))
		tweet_list.append(tweet1)
	tweet_data_new = tweet_data.drop('Tweet content', axis=1)
	tweet_data_new = tweet_data_new.assign(tweet_content=pandas.Series(tweet_list).values)
	tweet_data_new.to_csv(out_file, sep=',', encoding='utf-8')