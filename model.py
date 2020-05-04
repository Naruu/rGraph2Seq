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


class Graph2Seq(nn.Module):

    def __init__(self, mode, conf):
        super(Graph2Seq, self).__init__()

        self.mode = mode
        self.l2_lambda = conf.l2_lambda
        self.feature_embedding_dim = conf.hidden_layer_dim
        self.vocab_size = conf.vocab_size
        self.length = conf.vocab_size -1
        self.hop_size = conf.hop_size

        # the setting for the GCN
        self.graph_encode_direction = conf.graph_encode_direction
        self.concat = conf.concat

        # the setting for the decoder
        self.single_graph_nodes_size = conf.graph_size
        self.attention = conf.attention
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
                                hidden_dim=self.feature_embedding_dim,
                                vocab_size=self.vocab_size,
                                dropout=self.dropout,
                                length=self.length,
                                layers=self.decoder_layers
                                )


        """
        encoder, decoder가 공유해야하는 것 : node feature embedding -> should it?
        encoder에서는 operation이 ndoe feature
        decoder에서는 operation + connection 이 node feature
        seminas에서는 다른 embedding 이용
        graph2sec 에서는 동일한 embedding 이용
        나는 다르게 써야겠당
        """

    def forward(self, fw_adjs, bw_adjs, operations, targets=None):
        encoded_nodes, graph_embedding = self.encoder(fw_adjs, bw_adjs, operations)
        # check dimensino
        decoder_input = torch.concat([graph_embedding, targets], dim=1)
        predicted_softmax, decoded_ids = self.decoder(graph_embedding, targets=targets)

        return predicted_softmax, decoded_ids
