import data as md
import networks
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.img_size = (224, 224)
        self.patch_size = 16
        assert self.img_size[0] % self.patch_size == 0 and self.img_size[1] % self.patch_size == 0, 'patch size is not ' \
                                                                                                    'aligned with img_size '
        self.N = int(self.img_size[0] * self.img_size[1] / self.patch_size ** 2)
        self.W = 768
        self.D = 512
        self.isTrain = True
        self.fineTuning = False
        self.batch_size = 2


opt = Config()
tr = networks.Transformer(opt)
Data = md.AttDataset('data.csv', opt)
dataloader = md.get_dataloader(Data, opt)
bar = tqdm(enumerate(dataloader), total=len(dataloader))
for step, data in bar:
    imgs = data[0]
    labels = data[1]
    imgs = tr.forward(imgs)
    print(imgs.shape)


