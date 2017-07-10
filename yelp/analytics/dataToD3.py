import json
from pprint import pprint

sample_file = 'sample_topics.json'


def main():
	with open(sample_file, 'r') as f:
		data = json.load(f)

	convert_to_d3_cluster_data(data)	


def convert_to_d3_cluster_data(data):
	d3_json = {}
	d3_json['name'] = 'flare'
	d3_json['children'] = []

	for k, v in data.iteritems():
		level1_dict = {}
		level1_dict['name'] = k
		level1_dict['children'] = []
		even = int(k.split(':')[1]) % 2
		level1_dict['even'] = even
		for item in v:
			topic = item.split(':')
			level1_dict['children'].append({'name':topic[0],'size':topic[1]})
		d3_json['children'].append(level1_dict)

	#json export
	outputjson = 'd3_cluster_topic.json'
	print "writing topics to file:", outputjson
	with open( outputjson, 'w') as f:
		f.write(json.dumps(d3_json))	


if __name__=="__main__":
	main()