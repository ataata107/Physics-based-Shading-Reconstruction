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

def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    # if is_best:
    #     best_fpath = best_model_path
    #     # copy that checkpoint file to best path given, best_model_path
    #     shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    # valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch']#, valid_loss_min.item()


# import cv2
# import numpy as np
# import scipy.io
# mat = scipy.io.loadmat('//home/ataata107/Desktop/Ata/paper2/download (1)/SN_envmap_mat_files_full/02871439/1ab8202a944a6ff1de650492e45fb14f/Camera_front000001.mat')
# cap = cv2.imread("/home/ataata107/Desktop/Ata/paper2/download (1)/SN_envmap_mat_files_full/02871439/1ab8202a944a6ff1de650492e45fb14f/Camera_front00.png")
# gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
# # print(cap.size)
# lapRgb = cv2.Laplacian(cap,cv2.CV_16S,ksize=3)
# lapGray = cv2.Laplacian(gray,cv2.CV_16S, ksize=3)
# rByG = cap[:,:,2]/(cap[:,:,1]+0.000001)
# rByB = cap[:,:,2]/(cap[:,:,0]+0.000001)
# gByB = cap[:,:,1]/(cap[:,:,0]+0.000001)

# rByG = np.log(rByG + 0.000001)
# rByB = np.log(rByB + 0.000001)
# gByB = np.log(gByB + 0.000001)

# # rByG = np.abs(rByG)
# # rByG *= 255.0/rByG.max()

# laprByG = cv2.Laplacian(np.uint8(rByG),cv2.CV_16S,ksize=3)
# laprByB = cv2.Laplacian(np.uint8(rByB),cv2.CV_16S,ksize=3)
# lapgByB = cv2.Laplacian(np.uint8(gByB),cv2.CV_16S,ksize=3)

# laprByG = cv2.convertScaleAbs(laprByG)
# laprByB = cv2.convertScaleAbs(laprByB)
# lapgByB = cv2.convertScaleAbs(lapgByB)

# # print(laprByG.dtype)
# agi = np.sqrt(laprByG**2+laprByB**2+lapgByB**2)
# # agi = cv2.convertScaleAbs(agi)
# # cv2.imshow("sdc",cv2.convertScaleAbs(np.uint8(agi)))
# # print(agi.dtype)
# # print(cv2.convertScaleAbs(np.uint8(agi)).max())
# agi *= 255.0/agi.max()
# # print(agi<0.01)
# mask = np.copy(agi)
# mask[agi<0.01] = 255
# # print((agi.max()))
# mask[agi>=0.01] = 0
# # res = cap*mask
# mask = np.uint8(mask)
# res = cv2.bitwise_and(cap,cap,mask=mask)
# # print((mask.min()))
# # print(agi<0.01)
# ########Shading
# lapr = cv2.Laplacian((res[:,:,2]),cv2.CV_16S,ksize=3)
# lapg = cv2.Laplacian((res[:,:,1]),cv2.CV_16S,ksize=3)
# lapb = cv2.Laplacian((res[:,:,0]),cv2.CV_16S,ksize=3)
# lap = cv2.Laplacian(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),cv2.CV_16S,ksize=3)
# lapr = cv2.convertScaleAbs(lapr)
# lapg = cv2.convertScaleAbs(lapg)
# lapb = cv2.convertScaleAbs(lapb)
# lap = cv2.convertScaleAbs(lap)
# sgi = np.sqrt(lapr**2+lapg**2+lapb**2)
# sgi *= 255.0/sgi.max()
# # lap *= 255.0/lap.max()
# to_show_1 = mat["Diffuse_color"]
# to_show_2 = mat["Image_R"]
# to_show_3 = mat["Image"]

# print(np.sum(to_show_3==to_show_2)==cap.size)
# to_show_1 *= 255.0/to_show_1.max()
# to_show_2 *= 255.0/to_show_2.max()
# to_show_3 *= 255.0/to_show_3.max()
# # print(to_show.max())
# cv2.imshow("sdc1",cap)
# cv2.imshow("sdc2",np.uint8(to_show_1))
# cv2.imshow("sdc3",np.uint8(to_show_2))
# cv2.imshow("sdc4",np.uint8(to_show_3))
# cv2.waitKey(0)
# cv2.destroyAllWindows() 