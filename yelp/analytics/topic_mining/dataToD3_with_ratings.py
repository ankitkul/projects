import json
from pprint import pprint

sample_file_1 = 'sample_low_rating_topic.txt.json'
sample_file_2 = 'sample_high_rating_topic.txt.json'


def main():
	with open(sample_file_1, 'r') as f:
		data1 = json.load(f)

	with open(sample_file_2, 'r') as f:
		data2 = json.load(f)		

	convert_to_d3_cluster_data(data1, data2)	


def convert_to_d3_cluster_data(data1, data2):
	d3_json = {}
	d3_json['name'] = 'flare'
	d3_json['children'] = []
	d3_json['children'].append({'name':'Low Ratings', 'children': []})
	node = d3_json['children'][0]['children']

	for k, v in data1.iteritems():
		level1_dict = {}
		level1_dict['name'] = k
		level1_dict['children'] = []
		for item in v:
			topic = item.split(':')
			level1_dict['children'].append({'name':topic[0],'size':topic[1]})
		node.append(level1_dict)

	d3_json['children'].append({'name':'High Ratings', 'children': []})
	node = d3_json['children'][1]['children']

	for k, v in data2.iteritems():
		level1_dict = {}
		level1_dict['name'] = k
		level1_dict['children'] = []
		for item in v:
			topic = item.split(':')
			level1_dict['children'].append({'name':topic[0],'size':topic[1]})
		node.append(level1_dict)	

	#json export
	outputjson = 'd3_cluster_rating_topic.json'
	print "writing topics to file:", outputjson
	with open( outputjson, 'w') as f:
		f.write(json.dumps(d3_json))	


if __name__=="__main__":
	main()