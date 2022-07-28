import data as md
import networks
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.img_size = (224, 224)
        self.patch_size = (16, 16)
        assert self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[
            1] == 0, 'patch size is not ' \
                     'aligned with img_size '
        self.N = int(self.img_size[0] * self.img_size[1] / self.patch_size[0] / self.patch_size[1])  # number of patches
        self.D1 = self.patch_size[0] * self.patch_size[1] * 3  # Dimensions before linear projection
        self.D2 = 768  # Dimensions after linear projection

        self.isTrain = True
        self.batch_size = 2

        ## transformer parameters
        self.n_heads = 12
        assert self.D2 % self.n_heads == 0, 'transformer input dimensions should be divisible by n_heads'
        self.n_layers = 4

        ## MLP head parameters
        self.n_hidden = 768  # not fine tuning stage, vit require a hidden layer in the MLP layer
        self.n_class = 3  # number of class in the data
        self.fine_tuning = False  # if it is fine tuning stage


opt = Config()
tr = networks.Transformer(opt)
Data = md.AttDataset('data.csv', opt)
dataloader = md.get_dataloader(Data, opt)
bar = tqdm(enumerate(dataloader), total=len(dataloader))
for step, data in bar:
    imgs = data[0]
    labels = data[1]
    res = tr.forward(imgs)
    print(res)
