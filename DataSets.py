from torch.utils.data import Dataset
import cv2
import numpy as np
import random

from torch.utils.data import DataLoader
import torchvision.transforms.functional as TVTF
from pathlib import Path


def random_crop_patch(im, patch_size):
    H = im.shape[0]
    W = im.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im = cv2.resize(im, (W, H))
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch = im[ind_H : ind_H + patch_size, ind_W : ind_W + patch_size]
    return patch


class Kernel(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = np.load(data_path)
        self.kernel_list = data['arr_0']

    def __len__(self):
        return len(self.kernel_list)

    def __getitem__(self, index):
        kernel = self.kernel_list[index]
        kernel = TVTF.to_tensor(kernel.copy())
        return kernel
    

class Openimage(Dataset):
    def __init__(self, data_path, patch_size=256):
        super().__init__()
        data_path = Path(data_path)
        a = list(data_path.glob('*.jpg'))
        self.image_list = sorted([str(x) for x in a])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image, _, _ = cv2.split(image)
        image = random_crop_patch(image, self.patch_size)
        image = TVTF.to_tensor(image.copy())
        return image