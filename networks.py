import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, W, D):
        """
        :param W: dimension of input tensor
        :param D: fixed dimension of transformer
        """
        super(Embedding, self).__init__()
        self.W = W
        self.D = D
        self.cls = torch.rand((1, 1, D))

    def forward(self, patches):
        res = nn.Linear(self.W, self.D)(patches)
        res = nn.GELU()(res)
        self.cls = self.cls.expand(res.shape[0], -1, -1)
        return torch.cat((self.cls, res), 1)


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.Embed = Embedding(opt.D1, opt.D2)
        self.positional = torch.rand(size=(opt.batch_size, opt.N + 1, opt.D2))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=opt.D2, nhead=opt.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=opt.n_layers)
        self.MLPhead = nn.Linear(opt.D2, opt.n_hidden)
        self.MLPhead2 = nn.Linear(opt.hidden, opt.n_class)
        self.fine_tuning = opt.fine_tuning

    def forward(self, x):
        x = self.Embed(x)
        x = x + self.positional
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.MLPhead(x)
        if not self.fine_tuning:
            x = nn.Tanh()(x)
            x = self.MLPhead2(x)
        return x
