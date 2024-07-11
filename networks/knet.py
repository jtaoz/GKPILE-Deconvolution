import torch.nn as nn
import torch.utils.data
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

        self.main1 = nn.Sequential()
        if self.isize == 31:
            self.main1.add_module('0', self._block(self.nz, self.ngf * 4, 3, 1, 0))
        elif self.isize == 55:
            self.main1.add_module('0', self._block(self.nz, self.ngf * 4, 5, 1, 0))
        elif self.isize == 65:
            self.main1.add_module('0', self._block(self.nz, self.ngf * 4, 7, 1, 0))
        elif self.isize == 75:
            self.main1.add_module('0', self._block(self.nz, self.ngf * 4, 7, 1, 0))
        
        self.main2 = nn.Sequential()
        self.net()

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel, 0.8),
            nn.ReLU(True),
        )

    def net(self):
        if self.isize == 31:
            self.main2.add_module('1', self._block(self.ngf * 4, self.ngf * 2, 5, 2, 1))
            self.main2.add_module('2', self._block(self.ngf * 2, self.ngf, 5, 2, 1))
            self.main2.add_module('3', nn.ConvTranspose2d(self.ngf, 1, 5, 2, 1, bias=False))
        elif self.isize == 55:
            self.main2.add_module('1', self._block(self.ngf * 4, self.ngf * 2, 5, 2, 0))
            self.main2.add_module('2', self._block(self.ngf * 2, self.ngf, 5, 2, 1))
            self.main2.add_module('3', nn.ConvTranspose2d(self.ngf, 1, 5, 2, 1, bias=False))
        elif self.isize == 65:
            self.main2.add_module('1', self._block(self.ngf * 4, self.ngf * 2, 5, 2, 1))
            self.main2.add_module('2', self._block(self.ngf * 2, self.ngf, 6, 2, 1))
            self.main2.add_module('3', nn.ConvTranspose2d(self.ngf, 1, 5, 2, 1, bias=False))
        elif self.isize == 75:
            self.main2.add_module('1', self._block(self.ngf * 4, self.ngf * 2, 5, 2, 0))
            self.main2.add_module('2', self._block(self.ngf * 2, self.ngf, 4, 2, 0))
            self.main2.add_module('3', nn.ConvTranspose2d(self.ngf, 1, 5, 2, 0, bias=False))
        else:
            print('Incorrect input of kernel size!')
            assert False

    def forward(self, x):
        fe = self.net_part1(x)
        x = self.net_part2(fe)
        return x

    def net_part1(self, x):
        return self.main1(x)

    def net_part2(self, x):
        x = self.main2(x)
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
        self.main = nn.Sequential()
        self.net()

    def _block(self, in_channel, out_channel, kernel_size, stride, padding, bn=True):
        block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
        if bn:
            block.add_module('1', nn.BatchNorm2d(out_channel, 0.8))
        block.add_module('2', nn.LeakyReLU(0.2, inplace=True))
        return block

    def net(self):

        if self.isize == 31:
            self.main.add_module('0', self._block(self.nc, self.ndf, 5, 2, 1, False))
            self.main.add_module('1', self._block(self.ndf, self.ndf * 2, 5, 2, 1))
            self.main.add_module('2', self._block(self.ndf * 2, self.ndf * 4, 5, 2, 1))
            self.main.add_module('3', nn.Conv2d(self.ndf * 4, 1, 3, 1, 0, bias=False))
            self.main.add_module('4', nn.Sigmoid())
        elif self.isize == 55:
            self.main.add_module('0', self._block(self.nc, self.ndf, 5, 2, 1, False))
            self.main.add_module('1', self._block(self.ndf, self.ndf * 2, 5, 2, 1))
            self.main.add_module('2', self._block(self.ndf * 2, self.ndf * 4, 5, 2, 0))
            self.main.add_module('3', nn.Conv2d(self.ndf * 4, 1, 5, 1, 0, bias=False))
            self.main.add_module('4', nn.Sigmoid())
        elif self.isize == 65:
            self.main.add_module('0', self._block(self.nc, self.ndf, 5, 2, 1, False))
            self.main.add_module('1', self._block(self.ndf, self.ndf * 2, 6, 2, 1))
            self.main.add_module('2', self._block(self.ndf * 2, self.ndf * 4, 5, 2, 1))
            self.main.add_module('3', nn.Conv2d(self.ndf * 4, 1, 7, 1, 0, bias=False))
            self.main.add_module('4', nn.Sigmoid())
        elif self.isize == 75:
            self.main.add_module('0', self._block(self.nc, self.ndf, 5, 2, 0, False))
            self.main.add_module('1', self._block(self.ndf, self.ndf * 2, 4, 2, 0))
            self.main.add_module('2', self._block(self.ndf * 2, self.ndf * 4, 5, 2, 0))
            self.main.add_module('3', nn.Conv2d(self.ndf * 4, 1, 7, 1, 0, bias=False))
            self.main.add_module('4', nn.Sigmoid())
        else:
            print('Incorrect input of kernel size!')
            assert False

    def forward(self, x):
        x = self.main(x)
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

