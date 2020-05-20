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
        self.node_feature_embedding = nn.Embedding(self.vocab_size+1, self.feature_embedding_dim)

        # the setting for the GCN
        self.graph_encode_direction = graph_encode_direction
        self.hop_size = hop_size
        self.concat = concat

        self.dropout = dropout
        self.learning_rate = learning_rate
    
    def forward(self, fw_adjs, bw_adjs, features, num_nodes):
        """
        fw_adjs : [ batch_size, max_degree_size] # info of adjs as global index
        bw_adjs : [ batch_size, max_degree_size] # info of adjs as global index
        features : [ (batch_size + 1) ] # operation range from 3 to 7 and also 0. 0 implies padding.
        # last index of features is padding for empty neighbor
        """

        # [ (batch_size + 1), hidden_layer_dim ]
        # last index is padding for neighbor
        embedded_node_rep = self.encode_node_feature(features)
        # print("enmbeded_node_rep : {}".format(embedded_node_rep.size()))

        # fw_hidden and bw_hidden is the initial node embedding
        # [ batch_size, hidden_layer_dim]
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
                fw_aggregator = fw_aggregators[6]
            else:
                fw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                fw_aggregators.append(fw_aggregator)

            # neigh_vec_hidden: [node_size, adj_size, word_embedding_dim]
            # same size with fw_adjs
            if hop == 0:
                # get embeddings of adjs
                neigh_vec_hidden = embedded_node_rep[fw_adjs]

            else:
                neigh_vec_hidden = torch.cat([fw_hidden, torch.zeros([1, dim_mul * self.hidden_layer_dim])], dim=0)[fw_adjs]

            # print("neigh_vec_hidden : {}".format(neigh_vec_hidden.size()))
            # print("fw_hidden : {}".format(fw_hidden.size()))
            
            fw_hidden = fw_aggregator(fw_hidden, neigh_vec_hidden)
            # print("hop: {}, after aggregate: fw_hidden_size: {}".format(hop, fw_hidden.size()))
            
            if self.graph_encode_direction == "bi":
                if hop == 0:
                    dim_mul = 1
                else:
                    dim_mul = 2

                if hop > 6:
                    bw_aggregator = bw_aggregators[6]
                else:
                    bw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                    bw_aggregators.append(bw_aggregator)

                # neigh_vec_hidden: [node_size, adj_size, word_embedding_dim]
                # same size with bw_adjs
                if hop == 0:
                    # get embeddings of adjs
                    neigh_vec_hidden = embedded_node_rep[bw_adjs]
                    
                else:
                    neigh_vec_hidden = torch.cat([bw_hidden, torch.zeros([1, dim_mul * self.hidden_layer_dim])], dim=0)[bw_adjs]

                bw_hidden = bw_aggregator(bw_hidden, neigh_vec_hidden)


        # split by number of nodes per graph
        fw_hiddens = torch.split(fw_hidden, num_nodes.tolist())
        # group by graph with padding(-100) to max pool
        fw_hidden = torch.nn.utils.rnn.pad_sequence(fw_hiddens, batch_first=True, padding_value= -100)

        if self.graph_encode_direction == "bi":
            bw_hiddens = torch.split(bw_hidden, num_nodes.tolist())
            bw_hidden = torch.nn.utils.rnn.pad_sequence(bw_hiddens, batch_first=True, padding_value= -100)
            hidden = torch.cat([fw_hidden, bw_hidden], dim=2)
        else:
            hidden = fw_hidden
        
        
        """
        # hidden stores the representation for all nodes
        fw_hidden = torch.reshape(fw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
        if self.graph_encode_direction == "bi":
            bw_hidden = torch.reshape(bw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
            hidden = torch.cat([fw_hidden, bw_hidden], dim=2)
        else:
            hidden = fw_hidden
        """

        hidden = F.relu(hidden)
        pooled = torch.max(hidden, dim=1).values

        ## print("pooled : {}".format(pooled.size()))
        ## print("pooled : {}".format(pooled))

        if self.graph_encode_direction == "bi":
            graph_embedding = torch.reshape(pooled, [-1, 4 * self.hidden_layer_dim])
        else:
            graph_embedding = torch.reshape(pooled, [-1, 2 * self.hidden_layer_dim])


        # print("graph_embedding: {}".format(graph_embedding.size()))
        # print(graph_embedding)
        graph_embedding = (graph_embedding, graph_embedding)

        """
        graph_embedding = LSTMStateTuple(c=graph_embedding, h=graph_embedding)
        """

        # shape of hidden: [batch_size, single_graph_nodes_size, 4 * hidden_layer_dim]
        # shape of graph_embedding: ([batch_size, 4 * hidden_layer_dim], [batch_size, 4 * hidden_layer_dim])
        return hidden, graph_embedding


    def encode_node_feature(self, features):
        return self.node_feature_embedding(features)