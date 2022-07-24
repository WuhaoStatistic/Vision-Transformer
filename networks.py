import torch
import torch.nn as nn
import torch.functional as f
import torch.nn.functional as nf
import numpy as np
import math
from config import Config
from PIL import Image
import matplotlib.pyplot as plt
import time
from data import img2patch


class Embedding(nn.Module):
    def __init__(self, W, D):
        """
        :param W: dimension of input tensor
        :param D: fixed dimension of transformer
        """
        super(Embedding, self).__init__()
        self.W = W
        self.D = D
        self.head = torch.rand((1, 1, D))

    def forward(self, patches):
        res = nn.Linear(self.W, self.D)(patches)
        res = nn.GELU()(res)
        self.head = self.head.expand(res.shape[0], -1, -1)
        return torch.cat((self.head, res), 1)


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.Embed = Embedding(opt.W, opt.D)
        self.positional = torch.rand(size=(opt.batch_size, opt.N + 1, opt.D))

    def forward(self, x):
        x = self.Embed(x)
        x = x + self.positional
        return x


class SelfAttention(nn.Module):

    def __init__(self, input_dim, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        """
            input: batch_size * sequence_length * input_dim
            q,k : batch_size * input_dim * dim_k
            v : batch_size * input_dim * dim_v
        """
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.scale = 1 / (dim_k ** (1 / 2))

    def forward(self, X):
        """
            X: batch_size * sequence_length * input_dim
            Q,K : batch_size * sequence_length * dim_k
            v : batch_size * sequence_length * dim_v
        """
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)
        A = nn.Softmax(dim=2)(torch.bmm(Q, K.permute(0, 2, 1)) / self.scale)
        res = torch.bmm(A, V)
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, h):
        """
           h: number of Head
        """
        super(MultiHeadAttention, self).__init__()
        assert dim_k % h == 0, 'dim_k should be divisible by h'
        assert dim_v % h == 0, 'dim_k should be divisible by h'
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

    def forward(self, X):
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)
