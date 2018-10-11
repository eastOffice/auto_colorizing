import os

# set up data

for i, file in enumerate(os.listdir('testSet_resize')):
    if i < 1000:
        os.rename('./testSet_resize/' + file, './images/val/class/' + file)
    else:
        os.rename('./testSet_resize/' + file, './images/train/class/' + file)


# from Ipython.display import Image, display
# display(Image(filename='./images/val/class/84b3ccd8209a4db1835988d28adfed4c.jpg'))

####### End of data set up #######

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision import datasets, transforms

import os, shutil, time

from model import *
from utils import *

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def validate(val_loader, model, criterion, save_images, epoch):
    print('starting validation...')

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch model to validation mode
    model.eval()

    # Run through validation set
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):

        # Use GPU if available
        target = target.to(device)
        with torch.no_grad():
            input_gray_variable = input_gray.to(device)
            input_ab_variable = input_ab.to(device)
            target_variable = target.to(device)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable)

        # Record loss and measure accuracy
        losses.update(loss.data[0], input_gray.size(0))

        # Save images to file
        if save_images:
            for j in range(len(output_ab)):
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path, save_name=save_name)

        # Record the time to do forward pass and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the colde below, val refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation')
    return losses.avg

def train(train_loader, model, criterion, optimizer, epoch):
    '''Train model on data in train_loader for a single epoch'''
    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to train mode
    model.train()
    
    # Train for single eopch
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        
        # Use GPU if available
        input_gray_variable = input_gray.to(device)
        input_ab_variable = input_ab.to(device)
        target_variable = target.to(device)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable) # MSE
        
        # Record loss and measure accuracy
        losses.update(loss.data[0], input_gray.size(0))
        
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))

# Make folders and set parameters
best_losses = 1e10
epochs = 100

model = ColorizationNet()

if use_gpu:
    model.cuda()
    print('Loaded model onto GPU')

##### Start Training ########

## Mean Squared Error loss
criterion = nn.MSELoss()
# A problemetic loss function, making the model more likely to predict
# less strong or bright colors

## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

## Transform training and validation data

# Training
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = GrayscaleImageFolder('images/train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

# Validation
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder('images/val', val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

### Train model
for epoch in range(epochs):
    # Train for one epoch, then validate
    train(train_loader, model, criterion, optimizer, epoch)
    save_images = True
    losses = validate(val_loader, model, criterion, save_images, epoch)

    # Save checkpoint
    is_best_so_far = losses < best_losses
    best_losses = max(losses, best_losses)
    save_checkpoint({
            'epoch': epoch + 1,
            'best_losses': best_losses,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best_so_far, 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))
    


    






