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

        # training parameters
        self.isTrain = True
        self.batch_size = 2
        self.learning_rate = 0.002
        self.lr_policy = 'linear'  # linear | step | plateau | cosine
        self.n_epochs = 200  # total epoch
        self.n_epochs_decay = 30  # after n_epoch_decay, learning rate will be linearly reduced,only valid when using linear policy
        self.label_smoothing = 0  # label smoothing in cross entrophy loss
        # continue train parameters
        self.continue_train = False
        self.epoch_count = 10 if self.continue_train else 0

        # transformer parameters
        self.N = int(self.img_size[0] * self.img_size[1] / self.patch_size[0] / self.patch_size[1])  # number of patches
        self.D1 = self.patch_size[0] * self.patch_size[1] * 3  # Dimensions before linear projection
        self.D2 = 768  # Dimensions after linear projection
        self.n_heads = 12
        assert self.D2 % self.n_heads == 0, 'transformer input dimensions should be divisible by n_heads'
        self.n_layers = 4

        # MLP head parameters
        self.n_hidden = 768  # not fine tuning stage, vit require a hidden layer in the MLP layer
        self.n_class = 3  # number of class in the data
        self.fine_tuning = False  # if it is fine tuning stage


opt = Config()
tr = networks.Transformer(opt)
Data = md.AttDataset('data.csv', opt)

dataloader = md.get_dataloader(Data, opt)
optimizer = md.get_optmizer(tr.parameters(), opt)
schedular = md.get_scheduler(optimizer, opt)

bar = tqdm(enumerate(dataloader), total=len(dataloader))
for epoch in range(opt.n_epochs-opt.epoch_count):
    for step, data in bar:
        imgs = data[0]
        labels = data[1]
        res = tr.forward(imgs)
        loss = networks.loss(res, labels, opt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        schedular.step()
