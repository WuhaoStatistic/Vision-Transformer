import argparse


class Config(object):
    def __init__(self):
        self.N = 14
        self.D = 512
        self.isTrain = True
        self.fineTuning = False
        self.batch_size = 2
        self.n_heads = 8
        self.n_layers = 4
