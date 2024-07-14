import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, isize, nz=100):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = 64
        self.isize = isize
        kernel_dict = {31: 5, 45: 4, 55: 5, 65: 6, 75:4}
        pad_dict = {31: [1, 1, 1], 45:[1, 1, 1], 55: [0, 1, 1], 65: [1, 1, 1], 75: [0, 0, 0]}

        if self.isize == 31:
            self.layer1 = self._block(self.nz, self.ngf * 4, 3, 1, 0)
        elif self.isize == 45:
            self.layer1 = self._block(self.nz, self.ngf * 4, 5, 1, 0)
        elif self.isize == 55:
            self.layer1 = self._block(self.nz, self.ngf * 4, 5, 1, 0)
        elif self.isize == 65:
            self.layer1 = self._block(self.nz, self.ngf * 4, 7, 1, 0)
        elif self.isize == 75:
            self.layer1 = self._block(self.nz, self.ngf * 4, 7, 1, 0)
        else:
            raise ValueError(f'Invalid input size: {self.isize}. Supported sizes are 31, 45, 55, 65, 75')
        
        self.layers = nn.Sequential(self._block(self.ngf * 4, self.ngf * 2, 5, 2, pad_dict[self.isize][0]), 
                                   self._block(self.ngf * 2, self.ngf, kernel_dict[self.isize], 2, pad_dict[self.isize][1]), 
                                   nn.ConvTranspose2d(self.ngf, 1, 5, 2, pad_dict[self.isize][2], bias=False))

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel, 0.8),
            nn.ReLU(True),
        )

    def forward(self, x):
        w = self.g1(x)
        x = self.Gk(w)
        return x

    def g1(self, x):
        return self.layer1(x)

    def Gk(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), 1, self.isize * self.isize)
        x = F.softmax(x, dim=2)
        x = x.view(x.size(0), 1, self.isize, self.isize)
        return x


class Discriminator(nn.Module):
    def __init__(self, isize):
        super(Discriminator, self).__init__()
        self.isize = isize
        self.nc = 1
        self.ndf = 64
        kernel_dict = {31: [5, 5, 5, 3], 45: [5, 4, 5, 5], 55: [5, 5, 5, 5], 65: [5, 6, 5, 7], 75:[5, 4, 5, 7]}
        pad_dict = {31: [1, 1, 1, 0], 45:[1, 1, 1, 0], 55: [1, 1, 0, 0], 65: [1, 1, 1, 0], 75: [0, 0, 0, 0]}
        
        if self.isize not in kernel_dict.keys():
            raise ValueError(f'Invalid input size: {self.isize}. Supported sizes are 31, 45, 55, 65, 75')
        
        self.layers = nn.Sequential(self._block(self.nc, self.ndf, kernel_dict[self.isize][0], 2, pad_dict[self.isize][0], False),
                                    self._block(self.ndf, self.ndf * 2, kernel_dict[self.isize][1], 2, pad_dict[self.isize][1]),
                                    self._block(self.ndf * 2, self.ndf * 4, kernel_dict[self.isize][2], 2, pad_dict[self.isize][2]),
                                    nn.Conv2d(self.ndf * 4, 1, kernel_dict[self.isize][3], 1, pad_dict[self.isize][3], bias=False),
                                    nn.Sigmoid())

    def _block(self, in_channel, out_channel, kernel_size, stride, padding, bn=True):
        block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
        if bn:
            block.add_module('1', nn.BatchNorm2d(out_channel, 0.8))
        block.add_module('2', nn.LeakyReLU(0.2, inplace=True))
        return block

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=nz)
        self.nz = nz
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        #x = 0.299 * x[:,0,:,:] + 0.587 * x[:,1,:,:] + 0.114 * x[:,2,:,:] # rgb to y
        x = self.model(x)
        x = x.reshape(x.size()[0], self.nz, 1, 1)
        return x

