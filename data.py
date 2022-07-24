import torch
import torch.nn.functional as nf
import numpy as np
import torch.utils.data as data
import os
import pandas as pd
from PIL import Image
from torch.optim import lr_scheduler


class Config(object):
    def __init__(self):
        self.N = 16
        self.isTrain = True
        self.fineTuning = False
        self.batch_size = 2


@torch.no_grad()
def img2patch(img, patch):
    """
    img : The input image, need to be numpy array

    patch:  N in the paper

    return : a 4D tensor

    example: If a image is 224*224*3 and patch = 14. Then we will have 14*14 = 196 patches in
    total.Since we need one more vector to do output, so there would be 197 vectors in total. And each vector will
    have (224/14)^2 * 3= 768 dimensions.
    So the shape of result is [197*16*16*3]
    """
    patch = int(patch**(1/2))
    assert isinstance(img, np.ndarray), 'img should be numpy array'
    assert img.shape[0] % patch == 0 and img.shape[1] % patch == 0, 'H and W should be divisible by patch'
    res = torch.zeros((1, img.shape[0] // patch, img.shape[1] // patch, 3))
    for i in range(patch):
        for j in range(patch):
            temp = torch.from_numpy(img[(i * int(img.shape[0] / patch)):((i + 1) * int(img.shape[0] / patch)),
                                    (j * int(img.shape[1] / patch)):((j + 1) * int(img.shape[1] / patch)),
                                    :]).unsqueeze(0)
            res = torch.cat((res, temp), 0)
            res = res.squeeze(0)
    res = res.flatten(start_dim=1)
    res = res[1:, :]
    res = nf.normalize(res)
    return res


class AttDataset(data.Dataset):
    def __init__(self, name, opt):
        super(AttDataset, self).__init__()
        self.data = pd.read_csv('./' + name)
        self.opt = opt

    def __getitem__(self, index):
        p, l = self.data.loc[index, :]
        p = './train_img/' + p
        p = Image.open(p)
        p = np.array(p.resize(self.opt.img_size))
        p = img2patch(p, self.opt.N)
        return p, l

    def get(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return self.data.shape[0]

    def len(self):
        return self.__len__()


def get_dataloader(dataset, opt):
    return data.DataLoader(dataset, opt.batch_size, shuffle=True)


def get_optmizer(model, opt):
    optimizer = torch.optim.SGD(model, lr=opt.lr, nesterov=True, momentum=0.9)
    return optimizer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# opt = Config()
# a = AttDataset('data.csv', opt)
# print(a.len())
# print(a.get(0))
