from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import random
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import AlbShadDataset, Rescale, ToTensor
from models import FinalModel
from util import load_ckp, save_ckp

loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('ASPPConv') == -1 and classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight.data, 0.0, math.sqrt(2. / n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def toInt3(elem):
    val  = elem.split("_")
    val = val[1].split('.')
    return int(val[0])

# Gradient loss
def grad_loss(pdt, gt, device, direction = "x"):
  if(direction=="x"):
    filter_1 = torch.from_numpy(np.array([[-1,0,1],[-2,0,2],[-1,0,1]])).to(device).view(1, 1, 3, 3)
  else:
    filter_1 = torch.from_numpy(np.array([[-1,-2,-1],[0,0,0],[1,2,1]])).to(device).view(1, 1, 3, 3)
  pdt_grad = F.conv2d(pdt, filter_1, stride=1)
  gt_grad = F.conv2d(gt, filter_1, stride=1)
  return loss_l2(pdt_grad, gt_grad)

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 2

#Start Epoch
start_epoch = 0

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0004

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9
beta2 = 0.999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Rescale size
cnvrt_size = 256

#Checkpoint Path Inpainter
checkpoint_model_path = "./checkpoint/model/"

pretrained_model = None

if not os.path.exists(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
else:
  a1 = sorted(os.listdir(checkpoint_model_path),key = toInt3,reverse= True)
  if(len(a1)>0):
    pretrained_model = a1[0]

flow_dataset = AlbShadDataset(transform = transforms.Compose([ToTensor(),Rescale((cnvrt_size,cnvrt_size))]))
# flow_dataset = AlbShadDataset(transform = transforms.Compose([ToTensor()]))

dataloader = DataLoader(flow_dataset, batch_size=batch_size,shuffle=True, num_workers=workers)

net_model = FinalModel().to(device) 
# net_impainter.apply(weights_init)
alpha_mse_albedo = (torch.rand(1, requires_grad=True, dtype=torch.float, device=device))
alpha_mse_shading = (torch.rand(1, requires_grad=True, dtype=torch.float, device=device))
optimizerM = optim.Adam([
      {'params': net_model.parameters()},
      {'params': alpha_mse_albedo},
      {'params': alpha_mse_shading}
  ], lr=lr, betas=(beta1, beta2))
# optimizerM.param_groups.append({'params': alpha_mse_albedo})
# optimizerM.param_groups.append({'params': alpha_mse_shading})

if(pretrained_model!=None):
  net_model, optimizerM, start_epoch, alpha_mse_albedo, alpha_mse_shading = load_ckp(checkpoint_model_path+pretrained_model, net_model, optimizerM)
  print("Loaded pretrained: " + pretrained_model)




I_losses = []
iters = 0

print("Starting Training Loop... from" + str(start_epoch))
net_model.train()


step = 1
for epoch in range(start_epoch,num_epochs):
  step = 1
  for i, data in enumerate(dataloader, 0):
    # if(step%4!=0):
    optimizerM.zero_grad()

    image_rgb = data['RGB'].to(device)#.float()
    image_albedo = data['albedo'].to(device)
    image_shading = data['shading'].to(device)

    albedo_pred, shading_pred = net_model(image_rgb, image_shading)

    mse_loss_1 = loss_l2(albedo_pred, image_albedo)
    mse_loss_2 = loss_l2(shading_pred, image_shading)

    smse_loss_1 = loss_l2(albedo_pred*alpha_mse_albedo, image_albedo)
    smse_loss_2 = loss_l2(shading_pred*alpha_mse_shading, image_shading)

    grad_loss_x_1 = grad_loss(albedo_pred,image_albedo, device)
    grad_loss_y_1 = grad_loss(albedo_pred,image_albedo, device,"y")
    grad_loss_x_2 = grad_loss(shading_pred,image_shading, device)
    grad_loss_y_2 = grad_loss(shading_pred,image_shading, device,"y")
    grad_loss_1 = grad_loss_x_1 + grad_loss_y_1
    grad_loss_2 = grad_loss_x_2 + grad_loss_y_2

    albedo_loss = 2*(0.95*smse_loss_1 + 0.05*mse_loss_1) + 1*grad_loss_1
    shading_loss = 2*(0.95*smse_loss_2 + 0.05*mse_loss_2) + 1*grad_loss_2

    err = 1*albedo_loss + 1*shading_loss

    err.backward()
    optimizerM.step()
    
    print("Epoch"+str(epoch),"Step"+str(step),abs(err.item()),alpha_mse_albedo, alpha_mse_shading)
    if(step%200==0):
      checkpoint_model = {
          'epoch': epoch + 1,
          'state_dict': net_model.state_dict(),
          'alpha_mse_albedo' : alpha_mse_albedo,
          'alpha_mse_shading' : alpha_mse_shading,
          'optimizer': optimizerM.state_dict(),
      }

      save_ckp(checkpoint_model, checkpoint_model_path+"checkpoint_"+str(epoch+1)+".pt")
    step+=1
    
    # break
  checkpoint_model = {
        'epoch': epoch + 1,
        'state_dict': net_model.state_dict(),
        'alpha_mse_albedo' : alpha_mse_albedo,
        'alpha_mse_shading' : alpha_mse_shading,
        'optimizer': optimizerM.state_dict(),
  }
  save_ckp(checkpoint_model, checkpoint_model_path+"checkpoint_"+str(epoch+1)+".pt")

  # break