import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from networks.knet import Generator, ResNet18
import itertools as it
from torch.optim.lr_scheduler import MultiStepLR
from DataSets import *
from utils.common_utils import blurring
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed, seed= random.randint(0, 10000)')
parser.add_argument('--clean_img_path', default='./datasets/open_val', type=str, help="path to save the clean images")
parser.add_argument('--num_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations for latent codes per optimization step')
parser.add_argument('--batch_size', type=int, default=128, help='size of training batch')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of netE')
parser.add_argument('--lr_z', type=float, default=0.01, help='learning rate of latent code z')
parser.add_argument('--weight_lambda', type=float, default=0.1, help='the weight of the loss on latent code')
parser.add_argument('--kernel_size', type=int, default=31, help='size of blur kernel')
parser.add_argument('--kernel_path', type=str, default='', help='path to save blur kernel file')
parser.add_argument('--save_path', type=str, default='./models', help='path to save the model file')
parser.add_argument('--log_dir', default='./log', type=str, help="path to save the log file")
opt = parser.parse_args()

print(opt)

random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

os.makedirs(opt.save_path, exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)

def train(kernel_size, kernel_path):

    model_file_name = 'netE_{}.pth'.format(kernel_size)
    if os.path.exists(opt.save_path + '/' + model_file_name):
        print('The file %s already exists !' % model_file_name)
        return
    
    log_path = os.path.join(opt.log_dir, 'netE%d' % kernel_size)
    writer = SummaryWriter(log_path)
    netD_path = os.path.join(opt.save_path, './netG_{}.pth'.format(kernel_size))

    img_set = Openimage(opt.clean_img_path, 256)
    loader_img = DataLoader(dataset=img_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    kernel_set = Kernel(kernel_path)
    loader_kernel = DataLoader(dataset=kernel_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    netE = ResNet18().cuda()
    netG = Generator(kernel_size).cuda()
    netG.load_state_dict(torch.load(netD_path))
    netG.eval()
    for p in netG.parameters(): p.requires_grad = False

    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    scheduler = MultiStepLR(optimizerE, milestones=[40, 50], gamma=0.1)

    print("Starting Training Loop...")
    for epoch in range(opt.num_epochs):

        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []

        for i, data in enumerate(zip(it.cycle(loader_img), loader_kernel), 0):

            netE.zero_grad()
            kernel = data[1].cuda()
            clean_img = data[0].cuda()
            blur_img = blurring(clean_img, kernel, kernel_size)

            ze = netE(blur_img)
            z0 = ze.clone().detach().requires_grad_(True)
            optimizerI = optim.Adam([z0], lr=opt.lr_z)

            for _ in range(opt.num_iters):
                optimizerI.zero_grad()
                output_z0 = netG(z0)
                loss_iter = l1(output_z0, kernel)
                loss_iter.backward()
                optimizerI.step()
            zi = z0.clone().detach()
            output = netG(ze)

            loss1 = l1(output, kernel)
            loss2 = mse(ze, zi)
            loss = loss1 + opt.weight_lambda * loss2
            loss.backward()

            epoch_loss.append(loss.item())
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())

            optimizerE.step()

        print('[%d/%d]\tLoss: %.6f' % (epoch, opt.num_epochs, np.mean(epoch_loss)))
        writer.add_image('groundtruth', vutils.make_grid(kernel[:16], padding=2, normalize=True, nrow=4), epoch)
        writer.add_image('result', vutils.make_grid(output[:16], padding=2, normalize=True, nrow=4), epoch)
        writer.add_scalar("loss", np.mean(epoch_loss), epoch)
        writer.add_scalar("loss1", np.mean(epoch_loss1), epoch)
        writer.add_scalar("loss2", np.mean(epoch_loss2), epoch)
        scheduler.step()

    torch.save(netE.state_dict(), os.path.join(opt.save_path, model_file_name))
    writer.close()


if __name__ == '__main__':
    
    #train(kernel_size = 31, kernel_path='./datasets/kernel/lai31.npz')
    #train(kernel_size = 55, kernel_path='./datasets/kernel/lai55.npz')
    #train(kernel_size = 75, kernel_path='./datasets/kernel/lai75.npz')

    train(opt.kernel_size, opt.kernel_path)