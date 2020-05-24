import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.aggregators import MeanAggregator
import model_utils.configure as conf
import utils

from decoder import Decoder
from encoder import Encoder

SOS_ID = 0
EOS_ID = 0


class Graph2Seq(nn.Module):

    def __init__(self, mode, conf):
        super(Graph2Seq, self).__init__()

        self.mode = mode
        # self.l2_lambda = conf.l2_lambda
        self.feature_embedding_dim = conf.hidden_layer_dim
        self.vocab_size = conf.vocab_size
        self.length = conf.length
        
        self.hop_size = conf.hop_size

        # the setting for the GCN
        self.graph_encode_direction = conf.graph_encode_direction
        self.concat = conf.concat

        # the setting for the decoder
        self.single_graph_nodes_size = conf.graph_size
        # self.attention = conf.attention
        self.decoder_layers = conf.decoder_layers

        self.dropout = conf.dropout
        self.learning_rate = conf.learning_rate

        self.if_pred_on_dev = False

        self.encoder = Encoder(
                                mode=mode,
                                vocab_size=self.vocab_size,
                                feature_embedding_dim=self.feature_embedding_dim,
                                graph_encode_direction=self.graph_encode_direction,
                                hop_size=self.hop_size,
                                concat=self.concat,
                                dropout=self.dropout,
                                learning_rate=self.learning_rate
                                )

        self.decoder = Decoder(
                                mode=mode,
                                hidden_dim=self.feature_embedding_dim * 4,
                                embedding_vocab_size=conf.embedding_vocab_size,
                                decoder_vocab_size=conf.decoder_vocab_size,
                                dropout=self.dropout,
                                length=self.length,
                                layers=self.decoder_layers
                                )



    def forward(self, fw_adjs, bw_adjs, operations, num_nodes, targets=None):
        encoder_hidden, graph_embedding = self.encoder(fw_adjs, bw_adjs, operations, num_nodes)

        # initail states has dimension of [num_layers * num_directions, batch, hidden_size]
        initial_states = graph_embedding[0].unsqueeze(0)
        # print("initial_states size : {}".format(initial_states.size()))
        initial_states = tuple([initial_states, initial_states])

        # batch_size = graph_embedding.size(0)

        # decoder input 수정?
        if self.mode == "train":
            # decoder_input = torch.Tensor().new_full((batch_size, 1), 0, dtype=torch.long, requires_grad=True)
            predicted_softmax, decoded_ids = self.decoder(graph_embedding[0], num_nodes, initial_states=initial_states, encoder_hidden=encoder_hidden, targets=targets)

        elif self.mode == "test":
            # decoder_input = torch.Tensor().new_full((batch_size, 1), 0, dtype=torch.long, requires_grad=True)
            predicted_softmax, decoded_ids = self.decoder(graph_embedding[0], num_nodes, initial_states=initial_states, encoder_hidden=encoder_hidden)

        return predicted_softmax, decoded_ids
