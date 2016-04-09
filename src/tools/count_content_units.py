count = 0
with open('tmp/dblp/training_data.txt') as input:
	for line in input:
		elements = line.strip().split(' ')
		for element in elements:
			if int(element) >= 100000000:
				count += 1
print count
