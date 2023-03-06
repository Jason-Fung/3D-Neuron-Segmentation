#!/usr/bin/env python
# coding: utf-8

# ## Import Modules and Check GPU

# In[1]:

# import utility libraries
import os
import numpy as np
import gc
from datetime import datetime
from pytz import timezone
import pytz
import random
import yaml
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help="configuration file *.yml", type=str)
parser.add_argument('-n', '--net', help="neural network type: 'convnets' or 'transformers' ", type=str)
parser.add_argument('-a', '--arch', help="model architecture: 'ResUNET', 'SwinUNeTr', 'UNETR' ", type=str)
parser.add_argument('-m', '--mode', help=" 'preprocessed_subvolume' or 'sequential_subvolume", type=str)
parser.add_argument('-td', '--trainingdir', help="directory for training type", type=str)
args = vars(parser.parse_args())
print(args)

from processing.processing_functions import *
from train_engine.models import *

# import deep learning libraries
from torchvision import transforms, utils
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import tensorflow as tf
import torch
from torch.multiprocessing import set_start_method
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from ignite.engine import Engine, Events
from ignite.metrics import Metric, Loss
import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
# from ignite.metrics import ConfusionMatrix, DiceCoefficient
from monai.handlers.ignite_metric import IgniteMetric
import pprint

# from monai.losses import DiceLoss
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss
from monai.inferers.inferer import SliceInferer, SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet, UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

# import processing libraries
from patchify import unpatchify
import skimage

global patch_size
global batch_size

# In[3]:

# # cloud server
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
#device = idist.device()
device = torch.device("cuda:0")

if use_cuda == True:
    print("Using Cuda")
else:
    print("Using CPU")

torch.cuda.empty_cache()
# macbook
# # use_mps = torch.has_mps
# use_mps = False
# device = torch.device("mps" if use_mps else "cpu")


# ## Read TIF images into Tensors

# In[5]:

import os
import torchvision
from torch.utils.data import Dataset
import tifffile
import glob



# yaml experiment file location
# project_path = os.path.abspath(os.path.abspath(os.path.dirname(__vsc_ipynb_file__)) + "/../") # macbook
# project_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../") # cedar
project_path = "/home/jsfung/projects/def-haas/jsfung/"
training_dir = args['trainingdir']
net_type = args['net']
model_arch = args['arch']
exp_path = f"/config/{net_type}/{model_arch}/{training_dir}/"
exp_file = args['config']
# exp_file = "post_hptune_exp.yml"

with open(project_path + exp_path + exp_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# subvolume configurations
patch_x = config['DATASET']['x_patch']
patch_y = config['DATASET']['y_patch']
patch_z = config['DATASET']['z_patch']
spatial_dim = config['MODEL']['spatial_dim']
batch_size = config['DATASET']['batch_size']

# read in segmentation_exp
segmentation_exp = config['DATASET']['exp'] # +s_+d_+f

# Define shifting windows inferencing conditions
batch_size = config['DATASET']['batch_size'] # integer
lateral_steps = config['DATASET']['lateral_steps'] # integer
axial_steps = config['DATASET']['axial_steps'] # integer

# results path
model_name = config['model_name']
date = datetime.now(tz=pytz.utc).strftime('%Y%m%d')
time = datetime.now(tz=pytz.utc).strftime('%H%M%S')

# Path to Raw images and Masks
raw_path = config['raw_path']
mask_path = config['mask_path']

model_directory = project_path + f"results/{model_name}/"
date_directory = f"{date}/"
exp_directory = f"{patch_z}_{patch_y}_{patch_x}_{segmentation_exp}/"
time_directory = f"{date}_{time}/"

# In[20]:
experimental_path = model_directory+date_directory+exp_directory+f'{date}_{time}_{model_name}_experiment.yml'

if not os.path.exists(experimental_path):
    os.makedirs(os.path.dirname(experimental_path), exist_ok=True)
    with open(experimental_path, 'w') as outfile:
        config['date'] = f"{date}_{time}"
        yaml.dump(config, outfile, default_flow_style=False)

# # Jason's Desktop
# raw_path = "I:\My Drive\Raw\*.tif"
# mask_path = "I:\My Drive\Mask\*.tif"

# # Jason's Macbook
# raw_path = "/Users/jasonfung/haaslabdataimages@gmail.com - Google Drive/My Drive/Images/Raw/*.tif"
# mask_path = "/Users/jasonfung/haaslabdataimages@gmail.com - Google Drive/My Drive/Images/Mask/*.tif"

raw_filename_list = glob.glob(raw_path) 
mask_filename_list = glob.glob(mask_path)

# set up number of classes being used

if segmentation_exp == "+s_+d_+f":
    num_classes = 4 # classes: soma, dendrite, filopodia
if segmentation_exp == '+s_+d_-f':
    num_classes = 3 # classes: soma, dendrite
if segmentation_exp == '-s_+d_-f':
    num_classes = 2 # classes: neuron and background

# remove artifacts
remove_artifacts = config['DATASET']['remove_artifacts']
artifacts = config['DATASET']['artifacts']

# define experimental conditions
ex_autofluor = config['DATASET']['ex_autofluorescence'] # True/False
ex_melanocytes = config['DATASET']['ex_melanocytes'] # True/False

if ex_autofluor and ex_melanocytes:
    num_classes += 2
elif ex_autofluor == False and ex_melanocytes == True:
    num_classes += 1
elif ex_autofluor == True and ex_melanocytes == False:
    num_classes += 1
else:
    pass

print("Number of Classes: ", num_classes)

# In[11]:

log_directory = model_directory + date_directory + exp_directory + f"{time}_log_fold"
results_txt = model_directory+date_directory+exp_directory+f'{time}_results.txt'

os.makedirs(os.path.dirname(results_txt), exist_ok=True)
os.makedirs(log_directory)

with open(experimental_path, 'w') as outfile:
    config['RESULTS'] = {}
    config['RESULTS']['log_file_path'] = log_directory    
    yaml.dump(config, outfile, default_flow_style=False)


# create writer to log results into tensorboard
log_writer = SummaryWriter(log_directory)

if args["trainingdir"] == "changing_patch_experiments":
    patch_name = f"z_{patch_z}_y_{patch_y}_x_{patch_x}"
    # Locate raw sub volume image and sub volume mask file locations and return its list of file names for images 
    parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/{training_dir}/{segmentation_exp}/{patch_name}/"
else:
    parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/"

training_raw_folder = os.path.join(parent_folder,"Raw/*.tif")
training_mask_folder = os.path.join(parent_folder,"Mask/*.tif")
training_raw_list = glob.glob(training_raw_folder)
training_mask_list = glob.glob(training_mask_folder)

# # Define Model and Parameters

model = Build_Model(cfg = config, num_classes=num_classes)
model = model.to(device)

# build the dataset and dataloaders using the newly calculated mean and standard deviation for normalization
# and also set the sliding window infereencing based on the training paradigm

if spatial_dim == 2:
    dim_order = (0,3,1,2)
    patch_size = (patch_y, patch_x)
    inferer = SliceInferer(roi_size=patch_size, sw_batch_size=1, cval=-1, progress=True)

    image_transform = transforms.Compose([MinMaxScalerVectorized(),
                                        #   transforms.Normalize(mean, std),
                                        ])
    label_transform = transforms.Compose([process_masks(exp = segmentation_exp,
                                                        ex_autofluor=ex_autofluor,
                                                        ex_melanocytes=ex_melanocytes,
                                                        )])

    training_dataset = SubVolumeDataset(raw_directory = training_raw_list,
                                        mask_directory = training_mask_list,
                                        num_classes = num_classes,
                                        device = device)

if spatial_dim == 3:
    dim_order = (0,4,1,2,3) # define the image and mask dimension order
    patch_size = (patch_z, patch_y, patch_x)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=batch_size)

    subpatch_transform = transforms.Compose([
                                        MinMaxScalerVectorized(),
                                        patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = False)])

    # define transforms for labeled masks
    subpatch_label_transforms = transforms.Compose([
                                        process_masks(exp = segmentation_exp,
                                                        ex_autofluor=ex_autofluor,
                                                        ex_melanocytes=ex_melanocytes,
                                                        ),
                                        patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = True)
                                        ])


    subpatch_training_data = MyImageDataset(
                                raw_list = training_raw_list,
                                mask_list = training_mask_list,
                                remove_artifacts=remove_artifacts,
                                artifacts = artifacts,
                                transform = subpatch_transform,
                                label_transform = subpatch_label_transforms,
                                device = device,
                                img_order = dim_order,
                                mask_order = dim_order,
                                num_classes = num_classes,
                                train=True
                                )
    training_dataloader = DataLoader(subpatch_training_data, batch_size=batch_size, shuffle=False)

test_training_dataset = WholeVolumeDataset(raw_directory = raw_filename_list,
                                           mask_directory = mask_filename_list,
                                           num_classes = num_classes,
                                           raw_transform = image_transform,
                                           label_transform = label_transform,
                                           mask_order=(0,4,1,2,3),
                                           device = device)
test_training_dataloader = DataLoader(test_training_dataset, batch_size=1, shuffle=False)

# calculate mean and standard deviation of the dataset before training
pixel_tensor = torch.tensor([])
depth_tensor = torch.tensor([])

for batch in test_training_dataloader:
    raw_img, _ = batch
    #print(raw_img.shape)
    batch_size = raw_img.size(0)
    pixel_tensor = torch.cat((pixel_tensor, torch.flatten(raw_img)))
    depth_tensor = torch.cat((depth_tensor, torch.tensor([raw_img.shape[0]])))

mean = torch.mean(pixel_tensor*depth_tensor)/torch.mean(depth_tensor)
std = torch.sqrt(torch.mean((pixel_tensor - mean)**2 * depth_tensor) / torch.mean(depth_tensor))

print("mean pixel: ", mean)
print("std pixel: ", std)

with open(experimental_path, 'w') as outfile:
    config['DATASET']['mean'] = mean.item()
    config['DATASET']['std'] = std.item()
    config['DATASET']['AUGMENTATION']['mean_noise'] = mean.item()
    config['DATASET']['AUGMENTATION']['std_noise'] = std.item()
    yaml.dump(config, outfile, default_flow_style=False)

# Augmentation Function parameters using resnet parameters
z_deg = config['DATASET']['AUGMENTATION']['z_deg']
y_deg = config['DATASET']['AUGMENTATION']['y_deg']
x_deg = config['DATASET']['AUGMENTATION']['x_deg']
gamma_lower = config['DATASET']['AUGMENTATION']['gamma_lower']
gamma_upper = config['DATASET']['AUGMENTATION']['gamma_upper']

gamma = (gamma_lower,gamma_upper)
degree = (z_deg, y_deg, x_deg)
# translate = (10,10,10)
transform_rotate = torchio.RandomAffine(degrees=degree,
#                                         translation=translate, 
                                        image_interpolation="bspline")
transform_noise = torchio.RandomNoise(mean=mean.item(), std=std.item())
transform_gamma = torchio.RandomGamma(log_gamma=gamma)
transform_flip = torchio.RandomFlip(axes=('ap',),
                                    flip_probability=1,
                                    )
all_transform = torchio.Compose([
                                transform_gamma,
                                transform_rotate,
                                transform_flip,
                                transform_noise,
                                ])

# ## Loss, Metric, Schedulers

# In[13]:

config_loss = {"dice": DiceLoss, "dice_ce": DiceCELoss, "focal": FocalLoss, "dice_focal": DiceFocalLoss, "generalized_dice_focal": GeneralizedDiceFocalLoss, "generalized_dice": GeneralizedDiceLoss}
weight_config = config['weights']
loss_type = config['loss']
if weight_config == "balanced":
    print("Calculating Focal Weights")
    class_weights = torch.from_numpy(get_class_weights(mask_filename_list,classes = num_classes))
elif weight_config == None:
    class_weights = None
else:
    class_weights = weight_config

with open(results_txt, 'w') as results_file:
    results_file.write("Using {}".format(loss_type))

if loss_type == "dice_focal":
    #calculate weights based on relative frequency of pixels for each class
    print("Calculating Focal Weights")
    class_weights = torch.from_numpy(get_class_weights(mask_filename_list,classes = num_classes))
    loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_dice=1.0, lambda_focal=0.2)
    
elif loss_type == "generalized_dice_focal":
    #calculate weights based on relative frequency of pixels for each class
    loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_gdl=1.0, lambda_focal=0.2)
elif loss_type == "focal":
    print("Calculating Focal Weights")
    class_weights = torch.from_numpy(get_class_weights(mask_filename_list, classes = num_classes))
    loss_function = config_loss[loss_type](weight = class_weights)
else:
    loss_function = config_loss[loss_type]()

# Set up Optimizer configurations
max_epochs = config['max_epochs']
learning_rate = config['MODEL']['learning_rate']
l2 = config['MODEL']['l2']

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = l2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs, verbose = True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience = 10, threshold=1e-5, threshold_mode= 'abs', verbose=True)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# discretize = Compose([Activations(softmax = True), 
#                       AsDiscrete(logit_thresh=0.5)])


# ## Define Training and Validation Functions

# In[15]:

def train_2D(engine, batch):
    augment = config['DATASET']['AUGMENTATION']['augment']
    training_mode = args['mode']
    model.train()
    running_loss = 0
    count_loss = 0
    
    raw_img, mask_img = batch
    raw_img, mask_img = torch.permute(torch.unsqueeze(raw_img,-1),dim_order), torch.permute(mask_img,dim_order)

    raw_dict = dict(zip(list(range(len(raw_img))), raw_img))
    mask_dict = dict(zip(list(range(len(mask_img))), mask_img))

    # train on upper images
    if augment:
        #print("Augmenting Images")
        # Extract index of non-zero subvolumes
        non_empty_indices = get_index_nonempty_cubes(mask_dict)
        
        # Augment on non-zero subvolumes based on their location in the volume (by index)
        if non_empty_indices != []:
            aug_imgs, aug_masks = augmentation(all_transform, training_mode, spatial_dim, raw_dict, mask_dict, non_empty_indices)
            aug_imgs, aug_masks = aug_imgs.to(device), aug_masks.to(device)
            optimizer.zero_grad()
            output = model(aug_imgs)
            probabilities = torch.softmax(output, 1)
            # prediction = torch.argmax(probabilities, 1)
            current_loss = loss_function(probabilities, aug_masks)
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            count_loss += 1
    
    optimizer.zero_grad()
    output = model(raw_img)
    probabilities = torch.softmax(output, 1)
    # prediction = torch.argmax(probabilities, 1)
    
    current_loss = loss_function(probabilities, mask_img)
    current_loss.backward()
    optimizer.step()
    running_loss += current_loss.item()
    count_loss += 1
        
    return {"batch_loss":running_loss/count_loss}

def train_3D(engine, batch):
    augment = config['DATASET']['AUGMENTATION']['augment']
    shuffle = config['TRAINING']['shuffle']
    training_mode = args['mode']

    model.train()
    running_loss = 0
    count_loss = 0
    # Instantiate the dice sum for each class
    
    upper_img, upper_shape, lower_img, lower_shape, full_mask, upper_mask, lower_mask = batch
    # upper_img: dict() of {"index 1": upper_raw_tensor_1, "index 2": upper_raw_tensor_2, ..., "index n": upper_raw_tensor_n} raw_tensor = FloatTensor(Z,Y,X)
    # lower_img: dict() of {"index 1": lower_raw_tensor_1, "index 2": lower_raw_tensor_2", ..., "index n": lower_raw_tensor_n} raw_tensor = FloatTensor(Z,Y,X)
    # upper_shape: tuple() representing shape of the upper volume (z,y,x) Note: this is used for reconstruction
    # lower_shape: tuple() representing shape of the lower volume (z,y,x) Note: this is used for reconstruction
    # full_mask: torch.FloatTensor of size (B,C,Z,Y,X) B = batch, C = Class Channel 
    # upper_mask: dict() of {"index 1": upper_mask_tensor_1, "index 2": upper_mask_tensor_2, ..., "index n": upper_mask_tensor_n} mask_tensor = FloatTensor(C,Z,Y,X)
    # lower_mask: dict() of {"index 1": upper_mask_tensor_1, "index 2": upper_mask_tensor_2, ..., "index n": lower_mask_tensor_n} mask_tensor = FloatTensor(C,Z,Y,X)
    # Empty list to place subvolumes in
    
    tmp_upper_dict = {}
    tmp_lower_dict = {}    

    if shuffle == True:
        # shuffle the batches
        upper_key_list = list(range(len(upper_img)))
        random.shuffle(upper_key_list)
        
        # check if lower img exists, otherwise perform shuffling
        if lower_img == None:
            pass
        else:
            lower_key_list = list(range(len(lower_img)))
            random.shuffle(upper_key_list)
    else:
        upper_key_list = list(range(len(upper_img)))
        lower_key_list = list(range(len(lower_img)))
    
    
    # Only train on evenly split images
    if lower_img == []:
        num_upper_subvolumes = len(upper_img)
        if augment:
            # Extract index of non-zero subvolumes
            upper_indexes = get_index_nonempty_cubes(upper_mask)
            
            # Augment on non-zero subvolumes based on their location in the volume (by index)
            for bindex in range(0, len(upper_indexes), batch_size):
                # for augmentation
                if bindex + batch_size > len(upper_indexes):
                    upper_batch = upper_indexes[bindex:len(upper_indexes)]
                else:
                    upper_batch = upper_indexes[bindex:bindex+batch_size]

                sub_imgs, sub_masks = augmentation(all_transform, training_mode, spatial_dim, upper_img, upper_mask, upper_batch)
                sub_imgs, sub_masks = sub_imgs.to(device), sub_masks.to(device)
                optimizer.zero_grad()
                output = model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                prediction = torch.argmax(probabilities, 1)
                
                current_loss = loss_function(probabilities, sub_masks)
                current_loss.backward()
                optimizer.step()
                running_loss += current_loss.item()
                count_loss += 1

        for bindex in range(0, num_upper_subvolumes, batch_size):
            if bindex + batch_size > num_upper_subvolumes:
                # if the bindex surpasses the number of number of sub volumes
                batch_keys = upper_key_list[bindex:num_upper_subvolumes]
            else:
                batch_keys = upper_key_list[bindex:bindex+batch_size]
            
            sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0) 
            sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
            
            optimizer.zero_grad()
            output = model(sub_imgs) # predict the batches
            probabilities = torch.softmax(output, 1) 
            prediction = torch.argmax(probabilities,1)
            
            # update the upper img dictionary
            tmp_upper_dict.update(dict(zip(batch_keys,prediction)))
            
            current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
            
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            count_loss += 1
        
        # lower list does not exist
        tmp_lower_list = None
            
    # train on both 
    else:
        # train on upper images
        num_upper_subvolumes = len(upper_img)
        if augment:
            # Extract index of non-zero subvolumes
            upper_indexes = get_index_nonempty_cubes(upper_mask)
            
            # Augment on non-zero subvolumes based on their location in the volume (by index)
            for bindex in range(0, len(upper_indexes), batch_size):
                # for augmentation
                if bindex + batch_size > len(upper_indexes):
                    upper_batch = upper_indexes[bindex:len(upper_indexes)]
                else:
                    upper_batch = upper_indexes[bindex:bindex+batch_size]
                
                
                sub_imgs, sub_masks = augmentation(all_transform, training_mode, spatial_dim, upper_img, upper_mask, upper_batch)
                sub_imgs, sub_masks = sub_imgs.to(device), sub_masks.to(device)
                optimizer.zero_grad()
                output = model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                prediction = torch.argmax(probabilities, 1)
                
                current_loss = loss_function(probabilities, sub_masks)
                current_loss.backward()
                optimizer.step()
                running_loss += current_loss.item()
                count_loss += 1
        
        for bindex in range(0, num_upper_subvolumes, batch_size):
            if bindex + batch_size > num_upper_subvolumes:
                # if the bindex surpasses the number of number of sub volumes
                batch_keys = upper_key_list[bindex:num_upper_subvolumes]
            else:
                batch_keys = upper_key_list[bindex:bindex+batch_size]
            
            sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0) 
            sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
            optimizer.zero_grad()
            output = model(sub_imgs) # predict the batches
            probabilities = torch.softmax(output, 1) 
            prediction = torch.argmax(probabilities,1)
            
            # update the upper img dictionary
            tmp_upper_dict.update(dict(zip(batch_keys,prediction)))
            
            current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
            
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            count_loss += 1
        
        # train on lower image
        num_lower_subvolumes = len(lower_img)
        if augment:
            lower_indexes = get_index_nonempty_cubes(lower_mask)
            
            for bindex in range(0, len(lower_indexes), batch_size):
                # for augmentation
                if bindex + batch_size > len(lower_indexes):
                    lower_batch = lower_indexes[bindex:len(lower_indexes)]
                else:
                    lower_batch = lower_indexes[bindex:bindex+batch_size]
                    
                sub_imgs, sub_masks = augmentation(all_transform, training_mode, spatial_dim, lower_img, lower_mask, lower_batch)
                sub_imgs, sub_masks = sub_imgs.to(device), sub_masks.to(device)
                optimizer.zero_grad()
                output = model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                prediction = torch.argmax(probabilities, 1)
                
                current_loss = loss_function(probabilities, sub_masks)
                current_loss.backward()
                optimizer.step()
                running_loss += current_loss.item()
                count_loss += 1
                
        for bindex in range(0, num_lower_subvolumes, batch_size):
            if bindex + batch_size > num_lower_subvolumes:
                # if the bindex surpasses the number of number of sub volumes
                batch_keys = lower_key_list[bindex:num_lower_subvolumes]
            else:
                batch_keys = lower_key_list[bindex:bindex+batch_size]
            
            sub_imgs = torch.squeeze(torch.stack([lower_img.get(key) for key in batch_keys], dim=1), dim = 0) 
            sub_masks = torch.squeeze(torch.stack([lower_mask.get(key) for key in batch_keys], dim=1), dim = 0)            
            optimizer.zero_grad()
            output = model(sub_imgs)
            probabilities = torch.softmax(output, 1)
            prediction = torch.argmax(probabilities,1)
            
            # update the lower dictionary
            tmp_lower_dict.update(dict(zip(batch_keys,prediction)))

            current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            count_loss += 1

        #orig_shape = full_mask.shape[1:-1]
        #reconstructed_mask_order = (3,0,1,2)

        #upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
        #lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])

        #reconstructed = reconstruct_training_masks(upper_values, lower_values, upper_shape, 
        #                                        lower_shape, patch_size, orig_shape) # returns (z,y,x)
        #reconstructed = to_categorical_torch(reconstructed, num_classes = num_classes) # returns (z,y,x,c)
        #reconstructed = reconstructed.type(torch.int16)
        #reconstructed = torch.permute(reconstructed, reconstructed_mask_order)
        #reconstructed = torch.unsqueeze(reconstructed, 0) # make reconstructed image into (Batch,c,z,y,x)

#         full_mask = full_mask.type(torch.int16)
        #gt_mask = torch.permute(full_mask, dim_order).cpu() # roll axis of grount truth mask into (batch,c,z,y,x)
        
    return {"batch_loss":running_loss/count_loss}
# In[16]:

def validate(engine, batch):

    model.eval()
    with torch.no_grad():
        running_loss = 0
        count_loss = 0

        raw, mask = batch
        mask = torch.squeeze(mask, dim=0)

        output = inferer(inputs = raw, network = model)
        probabilities = torch.softmax(output, 1)
        prediction = torch.argmax(probabilities,1)
        prediction = torch.permute(to_categorical_torch(prediction, num_classes), (0,4,1,2,3))
        
        current_loss = loss_function(probabilities, mask) # + dice_loss(predictions, patch_gt)
        running_loss += current_loss.item()
        count_loss += 1

        mask = mask.to("cpu")

    return {"batch_loss":running_loss/count_loss, "y_pred":prediction, "y": mask}


# In[17]:

if training_mode == "preprocessed_subvolumes":
    trainer = Engine(train_2D)
    evaluator = Engine(validate)
if training_mode == "sequential_subvolumes":
    trainer = Engine(train_3D)
    evaluator = Engine(validate)

# ## Metrics and Progress Bars

# In[18]:

# set up progress bar

def metric_output_transform(output):
    y_pred, y = output["y_pred"], output["y"]
    return y_pred, y

# cm = ConfusionMatrix(num_classes=4, output_transform = metric_output_transform)
# dice_metric = RunningAverage(DiceCoefficient(cm))
dice_metric = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans = True)
metric = IgniteMetric(dice_metric, output_transform=metric_output_transform)

# progress output transform
def loss_output_transform(output):
    loss = output["batch_loss"]
    return loss

# Attach both metric to trainer and evaluator engine
# metric.attach(trainer,"Dice")
metric.attach(evaluator,"Dice")

RunningAverage(output_transform=loss_output_transform).attach(trainer, "batch_loss")
RunningAverage(output_transform=loss_output_transform).attach(evaluator, "batch_loss")
pbar = ProgressBar(persist=True)
pbar.attach(trainer, metric_names=["batch_loss"])
pbar.attach(evaluator, metric_names=["batch_loss"])

# ## Setup Model and Log Saving Directories

# In[19]:
import copy

@trainer.on(Events.STARTED)
def print_start(trainer):
    print("Training Started")
    with open(model_directory+date_directory+exp_directory+f'{time}_results.txt', 'w') as results_file:
            results_file.write(f'Starting Training')

@trainer.on(Events.EPOCH_STARTED)
def print_epoch(trainer):
    print("Epoch : {}".format(trainer.state.epoch))

@trainer.on(Events.TERMINATE)
def save_results_to_yaml(trainer):
    with open(model_directory+date_directory+exp_directory+f'{time}_exp_configs.yml', 'w') as exp_configs:
        yaml.dump(config, exp_configs, default_flow_style = False)

@trainer.on(Events.EPOCH_COMPLETED)
def save_model(trainer):
    global best_dice
    global best_epoch
    global best_epoch_file
    global best_loss
    
    epoch = trainer.state.epoch
    def get_saved_model_path(epoch):
        return model_directory + date_directory + exp_directory+f"{segmentation_exp}_{spatial_dim}D_{model_name}_{epoch}.pth"

    # initialize global values
    best_dice = -torch.inf if epoch == 1 else best_dice
    best_loss = torch.inf if epoch == 1 else best_loss
    best_epoch = 1 if epoch == 1 else best_epoch
    best_epoch_file = '' if epoch == 1 else best_epoch_file
    
    def log_training_results(trainer):
        evaluator.run(test_training_dataloader)
        # Get engine metrics and losses
        training_metrics = copy.deepcopy(evaluator.state.metrics)
        pbar.log_message(
            "Training Results - Epoch: {} \nMetrics\n{}"
            .format(trainer.state.epoch, pprint.pformat(training_metrics)))
        with open(model_directory+date_directory+exp_directory+f'{time}_results.txt', 'a') as results_file:
            results_file.write("Training Results - Epoch: {} \nMetrics\n{}\n".format(trainer.state.epoch, pprint.pformat(training_metrics)))
        return training_metrics
    
    training_metrics= log_training_results(trainer)
    
    train_dice = training_metrics['Dice']

    train_loss = training_metrics['batch_loss']
    
    # log results based off experiment
    log_writer.add_scalars('Training vs. Validation Loss',
                    {f'Training Loss' : train_loss}, epoch)

    if segmentation_exp == '+s_+d_+f':
        train_mean_dice = torch.mean(train_dice[0:2])

        log_writer.add_scalars('Training vs. Validation Soma Dice ',
                        {f'Training Soma Dice' : train_dice[0]}, epoch)
        
        log_writer.add_scalars('Training vs. Validation Dendrite Dice ',
                        {f'Training Dendrite Dice' : train_dice[1]}, epoch)
        
        log_writer.add_scalars('Training vs. Validation Filopodias Dice ',
                        {f'Training Filopodias Dice' : train_dice[2]}, epoch)
        
        log_writer.add_scalars('Training vs. Validation Mean Dice ',
                        {f'Training Mean Dice' : train_mean_dice}, epoch)
        
        log_writer.flush()

    if segmentation_exp == '+s_+d_-f':
        train_mean_dice = torch.mean(train_dice[0:1])

        log_writer.add_scalars('Training vs. Validation Soma Dice Fold',
                        {f'Training Soma Dice' : train_dice[0]}, epoch)
        
        log_writer.add_scalars(f'Training vs. Validation Dendrite Dice',
                        {f'Training Dendrite Dice' : train_dice[1]}, epoch)

        log_writer.add_scalars('Training vs. Validation Mean Dice ',
                        {f'Training Mean Dice' : train_mean_dice}, epoch)

        log_writer.flush()
    
    if segmentation_exp == '-s_+d_-f':
        log_writer.add_scalars('Training vs. Validation Neuron Dice ',
                        {f'Training Neuron Dice' : train_dice}, epoch)
        
        log_writer.flush()

    if (training_metrics['batch_loss'] < best_loss):
        
        # if there was a previous model saved, delete that one
        prev_best_epoch_file = get_saved_model_path(best_epoch)
        if os.path.exists(prev_best_epoch_file):
            os.remove(prev_best_epoch_file)

        # update the best mean dice and loss and save the new model state
#         best_dice = val_mean_dice
        best_loss = training_metrics['batch_loss']
        best_epoch = epoch
        best_epoch_file = get_saved_model_path(best_epoch)
#         print(f'\nEpoch: {best_epoch} - New best Dice and Loss! Mean Dice: {best_dice} Loss: {best_loss}\n\n\n')
        # print(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
        with open(model_directory+date_directory+exp_directory+f'{time}_results.txt', 'a') as results_file:
            results_file.write(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': best_epoch
                    },best_epoch_file)
        with open(experimental_path, 'w') as outfile:
            config['RESULTS']['model_states_path'] = best_epoch_file
            yaml.dump(config, outfile, default_flow_style=False)


trainer.run(training_dataloader, max_epochs = max_epochs)

# if __name__ == "__main__":
#     set_start_method("spawn")
#     backend = "nccl"
#     with idist.Parallel(backend=backend) as parallel:
#         parallel.run(training, config)

# %%
