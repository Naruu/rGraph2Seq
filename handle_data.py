# import base64
import json
import os
import time
from collections import OrderedDict

import numpy as np

SOS_ID = 0
EOS_ID = 0

INPUT = 'input'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'

operation_dict = {
    CONV1X1 : 3,
    CONV3X3 : 4,
    MAXPOOL3X3 : 5,
    OUTPUT : 6,
    INPUT : 7 # Added. seminNAS does not encode INPUT
}

"""
0: sos/eos
1: no connection
2: connection
3: CONV1X1
4: CONV3X3
5: MAXPOOL3X3
6: OUTPUT
7: INPUT
"""


## TODO rename functions
def read_data(input_path, graph_size):
    seqs = []
    graphs = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            
            g_fw_adj = [list() for _ in range(graph_size)]
            g_bw_adj = [list() for _ in range(graph_size)]
            g_node_feature = []
            
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            
            seq = jo['sequence']
            seqs.append(seq)

            matrix = jo['module_adjacency']
            for row in range(graph_size):
                for col in range(row + 1, graph_size):
                    if matrix[row][col] :
                        g_fw_adj[row].append(col)
                        g_bw_adj[col].append(row)
            

            operations = jo['module_operations']
            for op in operations:
                if op == CONV1X1:
                    g_node_feature.append(3)
                elif op == CONV3X3:
                    g_node_feature.append(4)
                elif op == MAXPOOL3X3:
                    g_node_feature.append(5)
                elif op == OUTPUT:
                    g_node_feature.append(6)
                elif op == INPUT :
                    g_node_feature.append(7)

            graph = {}
            graph['g_node_features'] = g_node_feature
            graph['g_fw_adj'] = g_fw_adj
            graph['g_bw_adj'] = g_bw_adj
            graphs.append(graph)

    return seqs, graphs