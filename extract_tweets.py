import sys
import re
import string

# Function to write to a file 
def write_file(file_name, data):
	with open(file_name, 'w') as file:
		for rows in data:
			file.write(','.join(feature for feature in rows) + '\n')		

# Function to get file data into a list
def gen_list(data):
	count = 0
	index = [1, 6]
	pattern = re.compile('([^\s\w]|_)+')
	dates = ["20160425", "20160426", "20160427"]
	final_list = []
	final_new_list = []
	for row in data:
		feature_list = []
		if row:
			split_list = row.split(',')
			length = len(split_list)
			for i in index:
				if i <= length - 1 and split_list[i]:
					feature_list.append(pattern.sub('', split_list[i]))
				else:
					feature_list.append(' ')
			final_list.append(feature_list);

	for data in final_list:
		if data[0] in dates:
			final_new_list.append(data)
	return final_new_list 

# Read input file
rows_input = sys.stdin.read().splitlines()
filtered_data = [data for data in rows_input if data]
input_data = gen_list(filtered_data)
write_file("twitter_feeds_apple_processed.txt", input_data)