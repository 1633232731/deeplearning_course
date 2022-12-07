import json
import logging
import os

import pickle as pkl
import time

import numpy as np
from numpy import average
from pandas import NA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from utils import sequence_mask, dataset_name_to_charges, set_random_seed


class LSTM(nn.Module):
    def __init__(self, device, embeddings_matrix, num_classes=None):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1])
        self.embeddings.weight.data.copy_(embeddings_matrix)
        self.embeddings.weight.requires_grad = False

        self.embed_size = embeddings_matrix.shape[1]
        self.device = device
        self.num_classes = num_classes

        self.hidden_size = 200

        self.lstm = nn.LSTM(self.embed_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.CE_loss = nn.CrossEntropyLoss()

        self.linear = nn.Linear(in_features=200 * 2, out_features=num_classes)

    def forward(self, inputs=None, inputs_seq_lens=None, labels=None):
        x_embed = self.embeddings(inputs)  # B * S * H
        x_embed = x_embed.to(self.device)

        # if <PAD> is fed into lstm encoder, it may be cause the error.
        x_embed_packed = pack_padded_sequence(x_embed, inputs_seq_lens, batch_first=True, enforce_sorted=False)

        encoder_outputs_packed, _ = self.lstm(x_embed_packed)  # B * S * 2H
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        x_max_pool, _ = torch.max(encoder_outputs, 1)
        logits = self.linear(x_max_pool)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits


class Transformer(nn.Module):
    def __init__(self, device, embeddings_matrix, num_classes=None):
        super(Transformer, self).__init__()

        # use pre_trained word embedding to init.
        self.embeddings = nn.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1])
        self.embeddings.weight.data.copy_(embeddings_matrix)
        self.embeddings.weight.requires_grad = True

        # config
        self.hidden_size = 200
        self.embed_size = 200
        self.max_sent_len = 500

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)

        self.linear = nn.Linear(self.embed_size, num_classes)

        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, inputs_id, labels=None):

        x_embed = self.embeddings(inputs_id)
        x_transformer = self.transformer_encoder(x_embed)  # [B, S, H]

        out, _ = torch.max(x_transformer, dim=1)  # [B, H]
        logits = self.linear(out)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits
