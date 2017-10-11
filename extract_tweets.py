import sys

# Function to write to a file 
def write_file(file_name, data):
	with open(file_name, 'w') as file:
		for rows in data:
			file.write(','.join(feature for feature in rows) + '\n')		

# Function to get file data into a list
def gen_list(data):
	index = [0, 1, 2, 6]
	final_list = []
	for row in data:
		feature_list = []
		if row:
			split_list = row.split(',')
			length = len(split_list)
			for i in index:
				if i <= length - 1 and split_list[i]:
					feature_list.append(split_list[i])
				else:
					feature_list.append(' ')
			final_list.append(feature_list);
	return final_list 

# Read input file
rows_input = sys.stdin.read().splitlines()
filtered_data = [data for data in rows_input if data]
input_data = gen_list(filtered_data)
write_file("twitter_feeds_apple_processed.csv", input_data)