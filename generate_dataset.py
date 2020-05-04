# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Before using this API, download the data files from the links in the README.

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


Adjacency matrices are expected to be upper-triangular 0-1 matrices within the
defined search space (7 vertices, 9 edges, 3 allowed ops). The first and last
operations must be 'input' and 'output'. The other operations should be from
config['available_ops']. Currently, the available operations are:
  CONV3X3 = "conv3x3-bn-relu"
  CONV1X1 = "conv1x1-bn-relu"
  MAXPOOL3X3 = "maxpool3x3"

When querying a spec, the spec will first be automatically pruned (removing
unused vertices and edges along with ops). If the pruned spec is still out of
the search space, an OutOfDomainError will be raised, otherwise the data is
returned.


The returned data object is a dictionary with the following keys:
  - module_adjacency
  - module_operations
  - sequence
"""

# import base64
import json
import os
import time
from collections import OrderedDict

# from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec as _model_spec
from utils import convert_arch_to_seq
import numpy as np
import tensorflow as tf

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


def transform_data(dataset_file, metrics_file_name='data/train_data.json'):
    """
    Read nasbench and generate dataset by dropping unneeded data :
    data : adjacency matrix, operations, targer sequence.

    - matrix_and_operation.json
    {
        module_hash : hash value of graph
        module_adjacency : numpy array object of adjacency matrix
        module_operations : list of operations
        sequence : list type object. matrix & operation expanded to sequence.
    }

    Args:
        dataset_file: path to .tfrecord file containing the dataset.
        metrics_file_name: name of file to store hash value, matrix, operation, sequence.
        sequence_file_name: name of file to store hash value, sequence.
    """

    metrics_file_name = "data/" + metrics_file_name

    print('Loading dataset from file... This may take a few minutes...')

    with open(metrics_file_name, "w+") as metrics_file:
        start = time.time()

        handled_hash = set()

        for serialized_row in tf.python_io.tf_record_iterator(dataset_file):
            # Parse the data from the data file.
            module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
                json.loads(serialized_row.decode('utf-8')))

            if module_hash in handled_hash:
                continue

            dim = int(np.sqrt(len(raw_adjacency)))
            adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
            adjacency = np.reshape(adjacency, (dim, dim))
            adjacency = adjacency.tolist() # ndarray can not be serialized.
            operations = raw_operations.split(',')
            # metrics = model_metrics_pb2.ModelMetrics.FromString(
            #    base64.b64decode(raw_metrics))
            sequence = convert_arch_to_seq(adjacency, operations)
            
            new_entry = {}
            new_entry['module_hash'] = module_hash
            new_entry['module_adjacency'] = adjacency
            new_entry['module_operations'] = operations
            new_entry['sequence'] = sequence
            metrics_file.write(json.dumps(new_entry)+"\n")
            
            handled_hash.add(module_hash)

        elapsed = time.time() - start

    print('Finisehd transforming data in %d seconds' % elapsed)


if __name__ == "__main__" :
    data = 'data/nasbench_only108.tfrecord'
    transform_data(data)