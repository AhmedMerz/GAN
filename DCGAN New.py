# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:49:04 2023

@author: am224745
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
import os, sys
from PIL import Image

import matplotlib.pyplot as plt


MODEL_NAME = 'DCGAN'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                std=[0.5])]
)

def get_sample_image(G, n_noise):
    """
        save sample 3 images
    """
    z = torch.randn(3, n_noise).to(DEVICE)  # Generate 3 noise samples
    y_hat = G(z).view(3, 128, 128)  # (3, 127, 127)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([3*128, 128])  # 3 images vertically stacked
    for j in range(3):
        img[j*128:(j+1)*128] = result[j]
    return img



class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # Feature map 1: (128x128x1) --> (64x64x8)
            nn.Conv2d(in_channel, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Feature map 2: (64x64x8) --> (32x32x16)
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Feature map 3: (32x32x16) --> (16x16x32)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            
            # Feature map 4: (16x16x32) --> (8x8x64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            


            
        )
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1

            nn.Linear(64*8*8, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_
    
    
    
class Generator(nn.Module):
    def __init__(self, input_size=100):
        super(Generator, self).__init__()
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64 * 8 * 8),
            nn.ReLU()
        )
        
        # Transposed Convolutional Layers
        self.deconv = nn.Sequential(
            # Upscale to 8x8x256
            # Feature map 1: (8x8x64) --> (16x16x32)
           nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(32),
           nn.ReLU(True),

           # Feature map 2: (16x16x32) --> (32x32x16)
           nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(16),
           nn.ReLU(True),

           # Feature map 3: (32x32x16) --> (64x64x8)
           nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(8),
           nn.ReLU(True),
       

           # Feature map 4: (64x64x8) --> (128x128x4)
           nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=False),
           nn.Tanh()
           
       )
        
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 8, 8)  # Reshape to appropriate shape for convolutional layers
        x = self.deconv(x)

        return x

    
    
D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)
# D.load_state_dict('D_dc.pkl')
# G.load_state_dict('G_dc.pkl')
    


    
Loaded_data = np.load('MPS_Training_image_and_Realizations_500.npz')
TI = Loaded_data['array1']
MPS_real = Loaded_data['array2']


# Visualizae MPS realizations from the above training image:
print('Total Number of realizations: %d \nThe dimension of each reservoir model: %d x %d in X and Y' %(MPS_real.shape[2],MPS_real.shape[0], MPS_real.shape[1]))

# Visualizae MPS realizations from the above training image:
print('Total Number of realizations: %d \nThe dimension of each reservoir model: %d x %d in X and Y' %(MPS_real.shape[2],MPS_real.shape[0], MPS_real.shape[1]))

plt.figure(figsize = (10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(MPS_real[:,:,i], cmap='binary')
    plt.title('Realization # %d' %(i))
    plt.xlabel('X axis, 100 ft')
    plt.ylabel('Y axis, 100 ft')
plt.tight_layout()

criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

batch_size = 100
max_epoch = 5000 # need more than 20 epochs for training generator
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_noise = 100


D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

X_train = MPS_real.reshape(128,128,1,500)
X_train = X_train.transpose(3,2,0,1)
tensor_train = torch.tensor(X_train, dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(tensor_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



for epoch in range(max_epoch):
    for idx, (images,) in enumerate(train_loader):
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels)

        z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()
        
        if step % 500 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))
        
        if epoch % 100 == 0:
            sample_images = get_sample_image(G, n_noise)
            im = Image.fromarray((sample_images * 255).astype(np.uint8))
            im.save(f'generated_epoch_{epoch}.png')




Z = get_sample_image(G, n_noise)[:128,:128]