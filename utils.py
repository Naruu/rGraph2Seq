import math
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import model_utils.configure as conf
## from nasbench import api

INPUT = 'input'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'

"""
0: sos/eos
1: no connection
2: connection
3: CONV1X1
4: CONV3X3
5: MAXPOOL3X3
6: OUTPUT
"""

MAX_EDGE = 9
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

        
class ControllerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(ControllerDataset, self).__init__()

        self.adjacency_matrices = []
        self.operations = []
        self.sequences = []

        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip()

                jo = json.loads(line, object_pairs_hook=OrderedDict)

                self.adjacency_matrices.append(jo['module_adjacency'])
                self.operations.append(jo['module_operations'])
                self.sequences.append(jo['sequence'])

    def __getitem__(self, index):
        operations = self.operations[index]
        num_nodes = len(operations)

        ops = []
        for op in operations:
            if op == CONV1X1:
                ops.append(3)
            elif op == CONV3X3:
                ops.append(4)
            elif op == MAXPOOL3X3:
                ops.append(5)
            elif op == OUTPUT:
                ops.append(6)
            if op == INPUT:
                ops.append(7)

        sample = {
            'matrix' : torch.LongTensor(self.adjacency_matrices[index]),
            'operations': torch.LongTensor(ops),
            'sequence': torch.LongTensor(self.sequences[index])
        }
        return sample
    
    def __len__(self):
        return len(self.sequences)

def collate_fn(samples):
    ## transform into batch samples

    ## create node -> global idnex
    ## global index -> neighbor's global index

    degree_max_size = conf.degree_max_size
    graph_size = conf.graph_size
    # degree_max_size = 5
    # graph_size = 7
    seq_max_length = int((graph_size+2)*(graph_size-1)/2)

    g_idxs = []
    g_fw_adjs = []
    g_bw_adjs = []
    g_operations = []
    g_sequence = []
    g_num_nodes = []

    g_idx_base = 0
    for g_idx, sample in enumerate(samples):
        matrix = sample['matrix']
        num_nodes = matrix.shape[0]
        g_num_nodes.append(num_nodes)

        for row in range(num_nodes):
            g_fw_adjs.append(list())
            g_bw_adjs.append(list())

        for row in range(num_nodes):
           for col in range(row+1, num_nodes):
            if matrix[row][col] :
                g_fw_adjs[g_idx_base + row].append(g_idx_base + col)
                g_bw_adjs[g_idx_base + col].append(g_idx_base + row)

        for op in sample['operations']:
            g_operations.append(op)

        sequence = sample['sequence']

        sequence = torch.cat([sequence, torch.LongTensor([0] * (seq_max_length - len(sequence)))])
        g_sequence.append(sequence)

        g_idx_base += num_nodes

    for idx in range(len(g_fw_adjs)):
        g_fw_adjs[idx].extend([g_idx_base] * (degree_max_size - len(g_fw_adjs[idx])))
        g_bw_adjs[idx].extend([g_idx_base] * (degree_max_size - len(g_bw_adjs[idx])))
        
    g_operations.append(0)

    g_num_nodes = torch.LongTensor(g_num_nodes)

    # [batch_size, conf.degree_max_size]
    g_fw_adjs = torch.LongTensor(g_fw_adjs)
    g_bw_adjs = torch.LongTensor(g_bw_adjs)

    # [batch_size +1] # due to padding
    g_operations = torch.LongTensor(g_operations)

    # [sum of sequence_length]
    g_sequence = torch.stack(g_sequence)

    return {
            'num_nodes' : g_num_nodes,
            'fw_adjs': g_fw_adjs,
            'bw_adjs': g_bw_adjs,
            'operations': g_operations,
            'sequence': g_sequence
            }

def convert_arch_to_seq(matrix, ops):
    seq = []
    n = len(matrix)
    assert n == len(ops)
    for col in range(1, n):
        for row in range(col):
            seq.append(matrix[row][col]+1)
        if ops[col] == CONV1X1:
            seq.append(3)
        elif ops[col] == CONV3X3:
            seq.append(4)
        elif ops[col] == MAXPOOL3X3:
            seq.append(5)
        if ops[col] == OUTPUT:
            seq.append(6)
    return seq

    assert len(seq) == (n+2)*(n-1)/2

def convert_seq_to_arch(seq):
    n = int(math.floor(math.sqrt((len(seq) + 1) * 2)))
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    ops = [INPUT]
    for i in range(n-1):
        offset=(i+3)*i//2
        for j in range(i+1):
            matrix[j][i+1] = seq[offset+j] - 1
        if seq[offset+i+1] == 3:
            op = CONV1X1
        elif seq[offset+i+1] == 4:
            op = CONV3X3
        elif seq[offset+i+1] == 5:
            op = MAXPOOL3X3
        elif seq[offset+i+1] == 6:
            op = OUTPUT
        ops.append(op)
    return matrix, ops

""""
def generate_arch(n, nasbench, need_perf=False):
    count = 0
    archs = []
    seqs = []
    valid_accs = []
    all_keys = list(nasbench.hash_iterator())
    np.random.shuffle(all_keys)
    for key in all_keys:
        fixed_stat, computed_stat = nasbench.get_metrics_from_hash(key)
        if len(fixed_stat['module_operations']) < 7:
            continue
        arch = api.ModelSpec(
            matrix=fixed_stat['module_adjacency'],
            ops=fixed_stat['module_operations'],
        )
        if need_perf:
            data = nasbench.query(arch)
            if data['validation_accuracy'] < 0.9:
                continue
            valid_accs.append(data['validation_accuracy'])
        archs.append(arch)
        seqs.append(convert_arch_to_seq(arch.matrix, arch.ops))
        count += 1
        if count >= n:
            return archs, seqs, valid_accs


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())
"""