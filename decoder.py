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
                vocab_size,
                dropout,
                length,
                layers
                ):
        super(Decoder, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.length = length
        self.vocab_size = vocab_size
        self.layers = layers
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.dropout = dropout
        self.attention = Attention(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.n = int(math.floor(math.sqrt((self.length + 1) * 2)))
        self.offsets=[]
        for i in range(self.n):
            self.offsets.append( (i + 3) * i // 2 - 1)
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None, target=None):
        """
        Each decode cell:
            input : embedded sequence of node
            output : node
        thus, should change node to embedded representation in next input
        """
        ## Always, exist :  encoder_hidden, encoder_output

        ## train이든 test 이든
        ## graph embedding을 첫 input으로 받아서 -> 계속해서 전 단계의 embedding을 받겟지.

        ## encoder last state와 attention을 해서 : 이때 graph embedding은 제외해야해.
        ## output을 만들어내야해

        ## train이면 output과 실제 target을 비교하고, 다음 input은 target의 element 값이 되어야할 거고
        ## loss function을 비교해서 gradient update를 해야돼


        ## test이면 output이 다음 input이 되고, loss function만 비교하면 되지

        decoder_hidden = self._init_state(encoder_outputs)

        # x : concat of [graph_embeddindg, node embeddings at last step]
        if self.mode == "train":
            ## there would be target.
            ## train to reduce the target
            bsz = x.size(0)
            tgt_len = x.size(1)
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            predicted_softmax = F.log_softmax(self.out(x.view(-1, self.hidden_dim)), dim=-1)
            predicted_softmax = predicted_softmax.view(bsz, tgt_len, -1)
        
            return predicted_softmax, None

        # x : list of graph embeddings
        elif self.mode == "test":
            bsz = x.size(0)
            length = self.length
            decoded_ids = x.new_tensor(bsz, 0).fill_(0).long()
            
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
    
    def _init_state(self, encoder_outputs):
        # tuple of (initial hidden state, initial cell state)
        return tuple([encoder_outputs, encoder_outputs])
