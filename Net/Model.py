import torch
import torch.nn as nn
from torch.nn import functional as F
from Net import CNN, PCNN, BiGRU
import numpy as np

class Model(nn.Module):
    def __init__(self, pre_word_vec, rel_num, opt, pos_dim=5, hidden_size=230):
        super(Model, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        pos_len = opt['max_pos_length']
        emb_dim = word_embedding.shape[1] + 2 * pos_dim
        self.encoder_name = opt['encoder']
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.drop = nn.Dropout(opt['dropout'])

        if self.encoder_name == 'CNN':
            self.encoder = CNN(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size, rel_num)

        elif self.encoder_name == 'BiGRU':
            self.encoder = BiGRU(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size * 2, rel_num)

        else:
            self.encoder = PCNN(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size * 3, rel_num)

        self.init_weight()

    def forward(self, X, X_Pos1, X_Pos2, X_Mask, X_Len):
        X = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        if self.encoder_name == 'CNN':
            X = self.encoder(X)
        elif self.encoder_name == 'BiGRU':
            X = self.encoder(X, X_Len)
        else:
            X = self.encoder(X, X_Mask)
        X = self.drop(X)
        X = self.rel(X)
        return X

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.zeros_(self.rel.bias)

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X = self.word_embedding(X)
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        X = torch.cat([X, X_Pos1, X_Pos2], -1)
        return X
