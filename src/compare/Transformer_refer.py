import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import math
# ## 功能函数定义

torch.manual_seed(0)
np.random.seed(0)


#### positional encoding ####
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=168):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


#### model stracture ####
class TransAm(nn.Module):
    def __init__(self, feature_size=10, num_layers=1, dropout=0.3):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.trans = nn.Linear(3,10)
        # self.decoder = nn.Sequential(
        #     nn.Linear(32*5, 40),
        #     nn.ReLU(),
        #     # nn.Linear(80, 20),
        #     # nn.ReLU(), 
        #     nn.Dropout(p=0.1),
        #     nn.Linear(40, 6))
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     DEVICE = src.DEVICE
        #     mask = self._generate_square_subsequent_mask(len(src)).to(DEVICE)
        #     self.src_mask = mask
        # if self.src_key_padding_mask is None:
        src = self.trans(src)
        src = src.transpose(0, 1)
        mask_key = src_padding.bool()
        self.src_key_padding_mask = mask_key
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output = self.decoder(output)
        output = output[-1]
        output = torch.flatten(output)
        # print(output.shape)
        # output = output.transpose(0, 1)
        return output