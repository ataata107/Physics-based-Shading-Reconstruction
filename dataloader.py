from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import math
import random
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import scipy.io
from os import path


def toInt1(elem):
    
    return int(elem)

def toInt2(elem):
    val  = elem.split("_")
    val = val[1].split('.')
    return int(val[0])

class AlbShadDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.base_path = "./Dataset/SN_envmap_mat_files_full"
        self.dirs = sorted(os.listdir(self.base_path))
        self.data = []
        for i in self.dirs:
            path1 = os.path.join(self.base_path,i)
            data = os.listdir(path1)
            for j in data:
                final_dir_path = os.path.join(path,j)
                final_img_path = os.path.join(final_dir_path,"Camera_front000001.mat")
                if(path.exists(final_img_path)):
                    self.data.append(final_img_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_dir = self.data[idx]
        mat =  scipy.io.loadmat(img_dir)
        image_rgb = mat["Image_R"]
        image_albedo = mat["Diffuse_color"]
        image_shading = mat["Z"]
        image_rgb *= 255.0/image_rgb.max()
        image_albedo *= 255.0/image_albedo.max()
        image_shading *= 255.0/image_shading.max()
        sample = {'RGB': image_rgb, 'albedo' : image_albedo, 'shading': image_shading}
        if self.transform:
          sample = self.transform(sample)
        # print(sample['first_image'].shape)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image_rgb, image_albedo, image_shading = sample['RGB'], sample['albedo'], sample['shading']
        new_h, new_w = self.output_size
        image_rgb = transforms.Resize((new_h, new_w))(image_rgb)
        image_albedo = transforms.Resize((new_h, new_w))(image_albedo)
        image_shading = transforms.Resize((new_h, new_w))(image_shading)
        return {'RGB': image_rgb, 'albedo' : image_albedo, 'shading': image_shading}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image_rgb, image_albedo, image_shading = sample['RGB'], sample['albedo'], sample['shading']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_rgb = image_rgb.transpose((2, 0, 1))
        image_albedo = image_albedo.transpose((2, 0, 1))
        image_shading = np.expand_dims(image_shading, 0)      
        return {'RGB': torch.from_numpy(image_rgb), 'albedo' : torch.from_numpy(image_albedo), 'shading': torch.from_numpy(image_shading)}