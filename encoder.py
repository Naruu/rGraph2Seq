import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.aggregators import MeanAggregator

class Encoder(nn.Module):
    
    def __init__(self,
                 mode,
                 vocab_size,
                 feature_embedding_dim,
                 graph_encode_direction,
                 hop_size,
                 concat,
                 dropout,
                 learning_rate
                 ):
        super(Encoder, self).__init__()

        ## graph2vec
        self.mode = mode
        self.vocab_size = vocab_size
        self.feature_embedding_dim = self.hidden_layer_dim = feature_embedding_dim
        # self.word_embedding_dim = conf.hidden_layer_dim
        self.node_feature_embedding = nn.Embedding(self.vocab_size, self.feature_embedding_dim)

        # the setting for the GCN
        self.graph_encode_direction = graph_encode_direction
        self.hop_size = hop_size
        self.concat = concat

        """
        # the following place holders are for the gcn
        self.feature_info = tf.placeholder(tf.int32, [None, None])              # the feature info for each node
        self.batch_nodes = tf.placeholder(tf.int32, [None, None])               # the nodes for each batch

        self.sample_size_per_layer = tf.shape(self.fw_adj_info)[1]

        self.single_graph_nodes_size = tf.shape(self.batch_nodes)[1]
        """
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.fw_aggregators = []
        self.bw_aggregators = []
    
    def forward(self, fw_adjs, bw_adjs, features):
        # [node_size, hidden_layer_dim]
        embedded_node_rep = self.encode_node_feature(features)

        # the fw_hidden and bw_hidden is the initial node embedding
        # [node_size, dim_size]
        fw_hidden = embedded_node_rep
        bw_hidden = embedded_node_rep

        fw_aggregators = []
        bw_aggregators = []

        # aggregate over hop size
        for hop in range(self.hop_size):
            if hop == 0:
                dim_mul = 1
            else:
                dim_mul = 2

            if hop > 6:
                fw_aggregator = self.fw_aggregators[6]
            else:
                fw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                fw_aggregators.append(fw_aggregator)

            # N^(k) as matrix
            # N^k는 k-1 때의 neighbors를 aggregate해서 만들어진다.
            # aggregator에 k-1 때의 neighbors를 넣어줘야한다.


            # [node_size, adj_size, word_embedding_dim]
            if hop == 0:

                # embedded_node_repo에서 neighbor만 뽑아내기
                neigh_vec_hidden = embedded_node_rep[fw_adjs]
                
                """
                # compute the neighbor size
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), dim=2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, dim=1)
                """

            else:
                print("fw_hidden_size", fw_hidden.size())
                neigh_vec_hidden = torch.concat([fw_hidden, torch.zeros([1, dim_mul * self.hidden_layer_dim])], 0)[fw_adjs]

            # fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            print("hop: {}, before aggregate : fw_hidden_size: {}".format(hop, fw_hidden.size()))
            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden))
            print("hop: {}, after aggregate: fw_hidden_size: {}".format(hop, fw_hidden.size()))
            

            if self.graph_encode_direction == "bi":
                if hop == 0:
                    dim_mul = 1
                else:
                    dim_mul = 2

                if hop > 6:
                    bw_aggregator = self.bw_aggregators[6]
                else:
                    bw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                    bw_aggregators.append(bw_aggregator)

                # N^(k) as matrix
                # N^k는 k-1 때의 neighbors를 aggregate해서 만들어진다.
                # aggregator에 k-1 때의 neighbors를 넣어줘야한다.


                # [node_size, adj_size, word_embedding_dim]
                if hop == 0:

                    # embedded_node_repo에서 neighbor만 뽑아내기
                    neigh_vec_hidden = embedded_node_rep[bw_adjs]
                    
                    """
                    # compute the neighbor size
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), dim=2)
                    tmp_mask = torch.sign(tmp_sum)
                    fw_sampled_neighbors_len = torch.sum(tmp_mask, dim=1)
                    """

                else:
                    print("bw_hidden_size", bw_hidden.size())
                    neigh_vec_hidden = torch.concat([bw_hidden, torch.zeros([1, dim_mul * self.hidden_layer_dim])], 0)[bw_adjs]

                # bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden))

        # hidden stores the representation for all nodes
        fw_hidden = torch.reshape(fw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
        if self.graph_encode_direction == "bi":
            bw_hidden = torch.reshape(bw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
            hidden = torch.concat([fw_hidden, bw_hidden], dim=2)
        else:
            hidden = fw_hidden

        hidden = F.relu(hidden)

        pooled = torch.max(hidden, dim=1)
        if self.graph_encode_direction == "bi":
            graph_embedding = torch.reshape(pooled, [-1, 4 * self.hidden_layer_dim])
        else:
            graph_embedding = torch.reshape(pooled, [-1, 2 * self.hidden_layer_dim])


        # graph_embedding = LSTMStateTuple(c=graph_embedding, h=graph_embedding)
        graph_embedding = (graph_embedding, graph_embedding)

        # shape of hidden: [batch_size, single_graph_nodes_size, 4 * hidden_layer_dim]
        # shape of graph_embedding: ([batch_size, 4 * hidden_layer_dim], [batch_size, 4 * hidden_layer_dim])
        return hidden, graph_embedding


    def encode_node_feature(self, features):
        return self.node_feature_embedding(features)