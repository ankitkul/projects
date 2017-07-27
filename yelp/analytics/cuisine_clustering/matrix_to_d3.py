import json
from pprint import pprint
import pandas as pd

cuisine_indices = 'cuisine_indices.txt'
sim_matrix = 'cuisine_sim_matrix.csv'


def main():
    c_indices = []
    with open(cuisine_indices, 'r') as f:
        for row in f.readlines():
            c_indices.append(row.replace("\n", ""))

    d3_json = {}
    d3_json['nodes'] = []
    d3_json['links'] = []


    for item in c_indices:
        d3_json['nodes'].append({"name":item, "group":1})

    with open(sim_matrix, 'r') as f1:
        i = 0
        for row in f1.readlines():
            line = row.replace("\n","").split(',')
            j = 0
            for l in line:
                d3_json['links'].append({"source":i, "target":j, "value": float(l)})
                j += 1
            i += 1
  
    #json export
    outputjson = 'd3_sim_matrix.json'
    print "writing topics to file:", outputjson
    with open( outputjson, 'w') as f:
        f.write(json.dumps(d3_json))               


if __name__=="__main__":
    main()