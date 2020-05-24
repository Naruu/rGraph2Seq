import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_ID = 0
EOS_ID = 0

class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
    
    def forward(self, input, source_hids, mask=None):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        
        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)
        
        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        
        return output, attn


class Decoder(nn.Module):
    
    def __init__(self,
                mode,
                hidden_dim,
                embedding_vocab_size,
                decoder_vocab_size,
                dropout,
                length,
                layers
                ):
        super(Decoder, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.length = length
        self.embedding_vocab_size = embedding_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.layers = layers
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.embedding = nn.Embedding(self.embedding_vocab_size, self.hidden_dim)
        self.dropout = dropout
        self.attention = Attention(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.decoder_vocab_size)
        self.n = int(math.floor(math.sqrt((self.length + 1) * 2)))
        self.offsets=[]
        for i in range(self.n):
            self.offsets.append( (i + 3) * i // 2 - 1)
    
    def forward(self, x, num_nodes, initial_states=None, encoder_hidden=None, targets=None):

        """
        x는 initial input of each batch
        that is, [batch_size, graph_embeddindg_dimension]
        """
        ## train이든 test 이든
        ## graph embedding을 첫 input으로 받아서 -> 계속해서 전 단계의 embedding을 받겟지.

        ## encoder last state와 attention : 이때 graph embedding은 제외 -> decoder output

        ## train이면 output과 실제 target을 비교하고, 다음 input은 target의 element 값이 되고
        ## loss function을 비교해서 gradient update

        ## test이면 output이 다음 input이 되고, loss function로 정확도만 계산. no gradient update

        if self.mode == "train":
            batch_size = x.size(0)
            target_length = targets.size(1)
            # targets to decoder input
            x = torch.Tensor().new_full((batch_size, 1), 0, dtype=torch.long, requires_grad=True)
            # print("before cat : {}".format(x.size()))
            x = torch.cat([x, targets[:,:-1]], dim=1)
            
            # print("after cat : {}".format(x.size()))
            x = self.embedding(x)
            # print("after embedding : {}".format(x.size()))
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            """
            h_0: shape (num_layers * num_directions, batch, hidden_size): initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
            c_0: shape (num_layers * num_directions, batch, hidden_size): initial cell state for each element in the batch.
            """
            # print("initial states : {}".format(initial_states[0].size()))
            x, hidden = self.rnn(x, initial_states)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_hidden)
            x = (residual + x) * math.sqrt(0.5)

            predicted_softmax = F.log_softmax(self.out(x.view(-1, self.hidden_dim)), dim=-1)
            predicted_softmax = predicted_softmax.view(batch_size, self.decoder_vocab_size, -1)
            # print("predicted_softmax before -1: {}".format(predicted_softmax.size()))
            predicted_softmax = predicted_softmax.view(batch_size, target_length, -1)
            # predicteed_softmax_list = torch.split(predicted_softmax, num_nodes.tolist())
            # predicted_softmax = torch.nn.utils.rnn.pad_sequence(predicteed_softmax_list, batch_first=True, padding_value=0)
        
            return predicted_softmax, None

        # x : list of graph embeddings
        elif self.mode == "test":
            batch_size = x.size(0)
            length = max(num_nodes)
            decoder_hidden = initial_states
            
            decoded_ids = torch.Tensor().new_full((batch_size, 1), 0, dtype=torch.long, requires_grad=False)
            
            def decode(step, output):
                if step in self.offsets:  # sample operation, should be in [3, 7]
                    if step != (self.n + 2) * (self.n - 1) / 2 - 1:
                        symbol = output[:, 3:6].topk(1)[1] + 3
                    else:
                        symbol = output[:, 6:].topk(1)[1] + 6
                else:  # sample connection, should be in [1, 2]
                    symbol = output[:, 1:3].topk(1)[1] + 1
                return symbol
            
            for i in range(length):
                x = self.embedding(decoded_ids[:, i:i+1])
                x = F.dropout(x, self.dropout, training=self.training)
                residual = x
                x, decoder_hidden = self.rnn(x, decoder_hidden)
                x = (residual + x) * math.sqrt(0.5)
                residual = x
                x, _ = self.attention(x, encoder_hidden)
                x = (residual + x) * math.sqrt(0.5)
                output = self.out(x.squeeze(1))
                symbol = decode(i, output)
                decoded_ids = torch.cat((decoded_ids, symbol), axis=-1)
                x = self.embedding(symbol)

            return None, decoded_ids