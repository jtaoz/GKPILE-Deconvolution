from __future__ import print_function

import argparse
import os

from networks.skip import skip
import glob
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *

import torch.nn.functional as F
from SSIM import SSIM
from networks.knet import Generator, ResNet18

parser = argparse.ArgumentParser()

parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel')
parser.add_argument('--data_path', type=str, default="./datasets/lai/uniform", help='path to blurry image')
parser.add_argument('--models_path', type=str, default='./models', help='path to save the model file')
parser.add_argument('--save_path', type=str, default="./results/lai", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=1000, help='lfrequency to save results')

opt = parser.parse_args()

#print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)


def get_kernel_network(kernel_size):

    netG_path = opt.models_path + '/' + 'netG_{}.pth'.format(kernel_size)
    netE_path = opt.models_path + '/' + 'netE_{}.pth'.format(kernel_size)

    netG = Generator(kernel_size).cuda()
    netG.load_state_dict(torch.load(netG_path))
    for p in netG.parameters(): p.requires_grad = False
    netG.eval()

    netE = ResNet18().cuda()
    netE.load_state_dict(torch.load(netE_path))
    for p in netE.parameters(): p.requires_grad = False
    netE.eval()

    return netE, netG



for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    print(imgname)

    if imgname.find('kernel_01') != -1:
        opt.kernel_size = 31
    if (imgname.find('kernel_02') != -1) or (imgname.find('kernel_03') != -1):
        opt.kernel_size = 55
    if imgname.find('kernel_04') != -1:
        opt.kernel_size = 75

    netE, netG = get_kernel_network(opt.kernel_size)

    
    new_path = os.path.join(opt.save_path, '%s' % imgname)
    os.makedirs(new_path, exist_ok=True)
    imgs, y = get_color_image(path_to_image, -1)  # load image and convert to np.
    img_blur = np_to_torch(imgs).type(dtype)
    y = np_to_torch(y).type(dtype)

    img_size = imgs.shape
    padh, padw = opt.kernel_size - 1, opt.kernel_size - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw

    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip(input_depth, 3,
                num_channels_down=[128, 128, 128, 128, 128],
                num_channels_up=[128, 128, 128, 128, 128],
                num_channels_skip=[16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    z = netE(y.unsqueeze(0))
    w = netG.g1(z)
    w.requires_grad = True
    out_k = netG.Gk(w)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    optimizerI = torch.optim.Adam([{'params': net.parameters()}, {'params': [w], 'lr': 5e-4}], lr=LR)
    schedulerI = MultiStepLR(optimizerI, milestones=[2000, 3000, 4000], gamma=0.5)

    net_input_saved = net_input.detach().clone()

    save_path = os.path.join(new_path, 'initialization_k1.png')
    out_k_np = torch_to_np(out_k)
    out_k_np = out_k_np.squeeze()
    out_k_np /= np.max(out_k_np)
    imsave(save_path, out_k_np)

    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(
            net_input_saved.data).normal_()

        # change the learning rate
        schedulerI.step(step)
        optimizerI.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = netG.Gk(w)
        out_img = F.conv2d(out_x, out_k.repeat(3, 1, 1, 1), groups=3)

        if step < 500:
            total_loss = mse(out_img, img_blur)
        else:
            total_loss = 1 - ssim(out_img, img_blur)
        total_loss.backward()
        optimizerI.step()

        if (step + 1) % opt.save_frequency == 0:

            out_x_np = torch_to_np(out_x).transpose(1, 2, 0)
            out_x_np = out_x_np[padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2], 0:3]
            save_path = os.path.join(new_path, '%d_x.png' % (step+1))
            imsave(save_path, out_x_np)
            save_path = os.path.join(new_path, '%d_k.png' % (step+1))
            out_k_np = torch_to_np(out_k)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            imsave(save_path, out_k_np)
