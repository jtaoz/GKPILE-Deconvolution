# GKPILE-Deconvolution

Code for reproducing the result of paper [Blind Image Deconvolution by Generative-based Kernel Prior and Initializer via Latent Encoding]

## Test

### Prerequisites
- PyTorch >= 1.10.0
- Requirements: opencv-python, tqdm

Download the pretrained models of kernel Generator(netG) and Initializer(netE) from [Google Drive](https://drive.google.com/drive/folders/1IXRYuf2ekyUObp1_tRc0aBotZxGZP-kx?usp=sharing) to the `models` folder. 

### Test on the synthetic images from Lai dataset

Put the test images in the `./datasets/Lai/uniform` folder. Reproduce results reported in the paper.
```bash
python deblur_lai.py
```

## Train

### Prepare datasets

#### Kernel datasets

The generated N blur kernels are combined into a three-dimensional array of shape (N, kernel_size, kernel_size), and stored as an npz file in the `./datasets/kernel` folder.

#### Clean image dataset

To train the kernel initializer, a clean image dataset is used for convolving with the blur kernels to produce blurred images. We used the OpenImage dataset and place the folder `open_val` into `./datasets`. Other clean image datasets could also be considered.

### Train kernel Generator
```bash
python train_generator.py --kernel_size [size of kernel] --kernel_path [path to kernel] --save_path [path to save model]
```

### Train kernel Initializer
```bash
python train_initializer.py --kernel_size [size of kernel] --kernel_path [path to kernel] --save_path [path to save model]
```

## Citation

If our work is useful for your research, please cite our paper:
```bash
@inproceedings{zhang2024blind,
  title={Blind Image Deconvolution by Generative-Based Kernel Prior and Initializer via Latent Encoding},
  author={Zhang, Jiangtao and Yue, Zongsheng and Wang, Hui and Zhao, Qian and Meng, Deyu},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
