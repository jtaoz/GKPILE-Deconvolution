
import random
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from networks.knet import weights_init, Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from DataSets import Kernel

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed, seed= random.randint(0, 10000)')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of training batch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
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

    model_file_name = 'netG_{}.pth'.format(kernel_size)
    if os.path.exists(opt.save_path + '/' + model_file_name):
        print('The file %s already exists !' % model_file_name)
        return
    
    log_path = os.path.join(opt.log_dir, 'netG%d' % kernel_size)
    writer = SummaryWriter(log_path)
    kernel_dataset = Kernel(kernel_path)
    loader = DataLoader(dataset=kernel_dataset, batch_size=opt.batch_size, shuffle=True)

    netG = Generator(kernel_size).cuda()
    netG.apply(weights_init)

    netD = Discriminator(kernel_size).cuda()
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.cuda.FloatTensor(64, 100, 1, 1).normal_()

    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    print("Starting Training Loop...")
    for epoch in range(opt.num_epochs):
        for i, data in enumerate(loader, 0):
            # Update D network
            netD.zero_grad()
            real_cpu = data.cuda()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label).cuda()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.cuda.FloatTensor(b_size, 100, 1, 1).normal_()
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, opt.num_epochs, i, len(loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i == len(loader) - 1:
                with torch.no_grad():
                    fake = netG(fixed_noise)
                writer.add_image('fake', vutils.make_grid(fake, nrow=8, padding=2, normalize=True), epoch)
                #writer.add_image('real', vutils.make_grid(real_cpu[:64], nrow=8, padding=2, normalize=True), epoch)

    torch.save(netG.state_dict(), os.path.join(opt.save_path, model_file_name))
    writer.close()


if __name__ == '__main__':
    
    #train(kernel_size = 31, kernel_path='./datasets/kernel/lai31.npz')
    #train(kernel_size = 55, kernel_path='./datasets/kernel/lai55.npz')
    #train(kernel_size = 75, kernel_path='./datasets/kernel/lai75.npz')

    train(opt.kernel_size, opt.kernel_path)
    
    
