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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help="configuration file *.yml", type=str)
args = vars(parser.parse_args())


from processing.processing_functions import *

# import deep learning libraries
from torchvision import transforms, utils
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from ignite.engine import Engine, Events
from ignite.metrics import Metric, Loss
import pprint

# from monai.losses import DiceLoss
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss
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
device = torch.device("cuda:0" if use_cuda else "cpu")
#device = idist.device()
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
project_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + "/../") # cedar
exp_path = "/config/convnets/ResUNet/kfold_training/"
exp_file = args['config']
# exp_file = "post_hptune_exp.yml"

with open(project_path + exp_path + exp_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)  

# Cloud Server
raw_path = config['raw_path']
mask_path = config['mask_path']

# # Jason's Desktop
# raw_path = "I:\My Drive\Raw\*.tif"
# mask_path = "I:\My Drive\Mask\*.tif"

# # Jason's Macbook
# raw_path = "/Users/jasonfung/haaslabdataimages@gmail.com - Google Drive/My Drive/Images/Raw/*.tif"
# mask_path = "/Users/jasonfung/haaslabdataimages@gmail.com - Google Drive/My Drive/Images/Mask/*.tif"

raw_filename_list = glob.glob(raw_path) 
mask_filename_list = glob.glob(mask_path)

# Pre Shuffle
raw_filename_list.sort()
mask_filename_list.sort()

# Shuffle the filename list
from sklearn.utils import shuffle
raw_filename_list, mask_filename_list = shuffle(raw_filename_list, mask_filename_list, random_state = 42)

raw_filename_list, mask_filename_list = np.array(raw_filename_list), np.array(mask_filename_list)

# results path
model_name = config['model_name']
date = datetime.now(tz=pytz.utc).strftime('%Y%m%d')
time = datetime.now(tz=pytz.utc).strftime('%H%M%S')

model_directory =  project_path + f"/results/{model_name}/"
date_directory = f"/{date}/"
time_directory = f"{date}_{time}/"
log_directory = model_directory + date_directory + time_directory + "log"
os.makedirs(log_directory)

# create writer to log results into tensorboard
log_writer = SummaryWriter(log_directory)


# In[20]:
output_txt = open(model_directory+date_directory+time_directory+'results.txt', 'w')
exp_configs = open(model_directory+date_directory+time_directory+'experiment.yml', 'w')

with open(model_directory+date_directory+time_directory + 'experiment.yml', 'w') as outfile:
     yaml.dump(config, outfile, default_flow_style=False)

# configure kFolds
from sklearn.model_selection import KFold
num_folds = config['DATASET']['folds']
splits = KFold(n_splits=num_folds, shuffle = True, random_state=42)

# remove artifacts
remove_artifacts = config['DATASET']['remove_artifacts']
artifacts = config['DATASET']['artifacts']

# define patching parameters

batch_size = config['DATASET']['batch_size'] # integer
lateral_steps = config['DATASET']['lateral_steps'] # integer
axial_steps = config['DATASET']['axial_steps'] # integer
patch_size = (axial_steps, lateral_steps, lateral_steps)
#split_size = config['DATASET']['split_size'] # integer 
segmentation_exp = config['DATASET']['exp'] # +s+d+f
ex_autofluor = config['DATASET']['ex_autofluorescence'] # True/False
ex_melanocytes = config['DATASET']['ex_melanocytes'] # True/False
dim_order = (0,4,1,2,3) # define the image and mask dimension order

# Augmentation Function parameters using resnet parameters
z_deg = config['DATASET']['AUGMENTATION']['z_deg']
y_deg = config['DATASET']['AUGMENTATION']['y_deg']
x_deg = config['DATASET']['AUGMENTATION']['x_deg']
gamma_lower = config['DATASET']['AUGMENTATION']['gamma_lower']
gamma_upper = config['DATASET']['AUGMENTATION']['gamma_upper']
noise_mean = config['DATASET']['AUGMENTATION']['mean_noise']
noise_std = config['DATASET']['AUGMENTATION']['std_noise']

gamma = (gamma_lower,gamma_upper)
gauss_mean = (0,noise_mean)
gauss_std = (0,noise_std)
degree = (z_deg, y_deg, x_deg)
# translate = (10,10,10)
transform_rotate = torchio.RandomAffine(degrees=degree, 
#                                         translation=translate, 
                                        image_interpolation="bspline")
transform_noise = torchio.RandomNoise(mean=gauss_mean, std=gauss_std)
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

# set up number of classes being used

if segmentation_exp == "+s_+d_+f":
    num_classes = 4 # classes: soma, dendrite, filopodia
if segmentation_exp == '+s_+d_-f':
    num_classes = 3 # classes: soma, dendrite
if segmentation_exp == '-s_+d_-f':
    num_classes = 2 # classes: neuron and background

if ex_autofluor and ex_melanocytes:
    num_classes += 2
elif ex_autofluor == False and ex_melanocytes == True:
    num_classes += 1
elif ex_autofluor == True and ex_melanocytes == False:
    num_classes += 1
else:
    pass

print("Number of Classes: ", num_classes)

def setup_dataflow(cfg, raw_filename_list,mask_filename_list, train_idx, val_idx, remove_artifacts, artifacts_list, num_classes, device):
    # define patching parameters

    lateral_steps = cfg['DATASET']['lateral_steps'] # integer
    axial_steps = cfg['DATASET']['axial_steps'] # integer
    patch_size = (axial_steps, lateral_steps, lateral_steps)
    dim_order = (0,4,1,2,3) # define the image and mask dimension order

    segmentation_exp = cfg['DATASET']['exp'] # +s+d+f
    ex_autofluor = cfg['DATASET']['ex_autofluorescence'] # True/False
    ex_melanocytes = cfg['DATASET']['ex_melanocytes'] # True/False

    patch_transform = transforms.Compose([
    #                                       new_shape(new_xy = (600,960)),
                                        MinMaxScalerVectorized(),
                                        patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = False)])

    # define transforms for labeled masks
    label_transforms = transforms.Compose([
    #                                        new_shape(new_xy = (600,960)),
                                        process_masks(exp = segmentation_exp,
                                                      ex_autofluor=ex_autofluor,
                                                      ex_melanocytes=ex_melanocytes,
                                                      ),
                                        patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = True)])

    raw_training_list, mask_training_list = raw_filename_list[train_idx], mask_filename_list[train_idx]
    raw_testing_list, mask_testing_list = raw_filename_list[val_idx], mask_filename_list[val_idx]

    training_data = MyImageDataset(
                                raw_training_list,
                                mask_training_list,
                                remove_artifacts=remove_artifacts,
                                artifacts = artifacts_list,
                                transform = patch_transform,
                                label_transform = label_transforms,
                                device = device,
                                img_order = dim_order,
                                mask_order = dim_order,
                                num_classes = num_classes,
                                train=True
                                )

    testing_data = MyImageDataset(
                                raw_testing_list,
                                mask_testing_list,
                                remove_artifacts=remove_artifacts,
                                artifacts = artifacts_list,
                                transform = patch_transform,
                                label_transform = label_transforms,
                                device = device,
                                img_order = dim_order,
                                mask_order = dim_order,
                                num_classes = num_classes,
                                train=False
                                )

    training_dataloader = DataLoader(training_data, batch_size = 1, shuffle = False)
    testing_dataloader = DataLoader(testing_data, batch_size = 1, shuffle = False)
    return training_dataloader, testing_dataloader



# In[11]:

for fold_idx, (train_idx, val_idx) in enumerate(splits.split(raw_filename_list, mask_filename_list)):

    training_dataloader, testing_dataloader = setup_dataflow(config,
                                                             raw_filename_list, 
                                                             mask_filename_list, 
                                                             train_idx, 
                                                             val_idx, 
                                                             remove_artifacts,
                                                             artifacts,
                                                             num_classes, 
                                                             device)

    # set up loss and optimizer
    max_epochs = config['max_epochs']
    dropout = config['dropout']
    learning_rate = config['learning_rate']
    l2 = config['l2']
    norm_type = config['norm']
    # decay = 1e-5
    input_chnl = 1
    output_chnl = num_classes

    # # Define Model and Parameters

    # ### Model: ResUNet

    model = UNet(spatial_dims=3, 
                in_channels = input_chnl,
                out_channels = output_chnl,
                channels = (32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm = norm_type,
                dropout = dropout)

    model = model.to(device)


    # ## Loss, Metric, Schedulers

    # In[13]:

    config_loss = {"dice": DiceLoss, "dice_ce": DiceCELoss, "focal": FocalLoss, "dice_focal": DiceFocalLoss, "generalized_dice_focal": GeneralizedDiceFocalLoss, "generalized_dice": GeneralizedDiceLoss}
    weight_config = config['weights']
    loss_type = config['loss']
    if weight_config == "balanced":
        print("Calculating Focal Weights")
        class_weights = torch.from_numpy(get_class_weights(mask_filename_list[train_idx],classes = num_classes))
    elif weight_config == None:
        class_weights = None
    else:
        class_weights = weight_config

    with open(model_directory+date_directory+time_directory+'results.txt', 'w') as results_file:
        results_file.write("Using {}".format(loss_type))

    if loss_type == "dice_focal":
        #calculate weights based on relative frequency of pixels for each class
        print("Calculating Focal Weights")
        class_weights = torch.from_numpy(get_class_weights(mask_filename_list[train_idx],classes = num_classes))
        loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_dice=1.0, lambda_focal=0.2)
        
    elif loss_type == "generalized_dice_focal":
        #calculate weights based on relative frequency of pixels for each class
        loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_gdl=1.0, lambda_focal=0.2)
    elif loss_type == "focal":
        print("Calculating Focal Weights")
        class_weights = torch.from_numpy(get_class_weights(mask_filename_list[train_idx], classes = num_classes))
        loss_function = config_loss[loss_type](weight = class_weights)
    else:
        loss_function = config_loss[loss_type]()

    # loss_function = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs, verbose = True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience = 10, threshold=1e-5, threshold_mode= 'abs', verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # discretize = Compose([Activations(softmax = True), 
    #                       AsDiscrete(logit_thresh=0.5)])


    # ## Define Training and Validation Functions

    # In[15]:

    def train(engine, batch):
        augment = config['DATASET']['AUGMENTATION']['augment']
        shuffle = config['TRAINING']['shuffle']

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
        if lower_img == None:
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
                        
                    sub_imgs, sub_masks = augmentation(all_transform, upper_img, upper_mask, upper_batch)
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
                        
                    sub_imgs, sub_masks = augmentation(all_transform, upper_img, upper_mask, upper_batch)
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
                        
                    sub_imgs, sub_masks = augmentation(all_transform, lower_img, lower_mask, lower_batch)
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

            orig_shape = full_mask.shape[1:-1]
            reconstructed_mask_order = (3,0,1,2)

            upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
            lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])

            reconstructed = reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                    lower_shape, patch_size, orig_shape) # returns (z,y,x)
            reconstructed = to_categorical_torch(reconstructed, num_classes = num_classes) # returns (z,y,x,c)
            reconstructed = reconstructed.type(torch.int16)
            reconstructed = torch.permute(reconstructed, reconstructed_mask_order)
            reconstructed = torch.unsqueeze(reconstructed, 0) # make reconstructed image into (Batch,c,z,y,x)

    #         full_mask = full_mask.type(torch.int16)
            gt_mask = torch.permute(full_mask, dim_order).cpu() # roll axis of grount truth mask into (batch,c,z,y,x)
            
        return {"batch_loss":running_loss/count_loss, "y_pred":reconstructed, "y":gt_mask}

    # In[16]:

    def validate(engine, batch):

        model.eval()
        with torch.no_grad():
            running_loss = 0
            count_loss = 0

            upper_img, upper_shape, lower_img, lower_shape, full_mask, upper_mask, lower_mask = batch
            # Empty list to place subvolumes in
            tmp_upper_dict = {}
            tmp_lower_dict = {}
            
            
            upper_key_list = list(range(len(upper_img)))
            lower_key_list = list(range(len(lower_img)))

            # Only train on evenly split images
            if lower_img == None:
                num_subvolumes = len(upper_img)
                for bindex in range(0, num_subvolumes, batch_size):
                    if bindex + batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = upper_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = upper_key_list[bindex:bindex+batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0)
                    sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                    
                    optimizer.zero_grad()
                    output = model(sub_imgs)
                    probabilities = torch.softmax(output, 1)

                    # discretize probability values 
                    prediction = torch.argmax(probabilities, 1)
                    tmp_upper_dict.update(dict(zip(batch_keys,prediction)))

                    # calculate the loss for the current batch, save the loss per epoch to calculate the average running loss
                    current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                    running_loss += current_loss.item()
                    count_loss += 1

                # lower list does not exist
                tmp_lower_list = None

            # train on both 
            else:
                num_subvolumes = len(upper_img)
                for bindex in range(0, num_subvolumes, batch_size):
                    if bindex + batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = upper_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = upper_key_list[bindex:bindex+batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                    sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)

                    optimizer.zero_grad()
                    output = model(sub_imgs) # predict the batches
                    probabilities = torch.softmax(output, 1) 
                    prediction = torch.argmax(probabilities,1)

                    current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                    running_loss += current_loss.item()
                    count_loss += 1

                    # update the upper img dictionary
                    tmp_upper_dict.update(dict(zip(batch_keys,prediction)))

                num_subvolumes = len(lower_img)
                for bindex in range(0, num_subvolumes, batch_size):
                    if bindex + batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = lower_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = lower_key_list[bindex:bindex+batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([lower_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                    sub_masks = torch.squeeze(torch.stack([lower_mask.get(key) for key in batch_keys], dim=1), dim = 0)

                    output = model(sub_imgs)
                    probabilities = torch.softmax(output, 1)
                    prediction = torch.argmax(probabilities,1)
                    current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                    running_loss += current_loss.item()
                    count_loss += 1

                    # update the lower dictionary
                    tmp_lower_dict.update(dict(zip(batch_keys,prediction)))

                # return tmp_upper_list, tmp_lower_list, running_loss / count
        
            # neuron reconstruction to calculate the dice metric.
            orig_shape = full_mask.shape[1:-1]
            reconstructed_mask_order = (3,0,1,2)


            upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
            lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])


            reconstructed = reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                        lower_shape, patch_size, orig_shape) # returns (z,y,x)
            reconstructed = to_categorical_torch(reconstructed, num_classes = num_classes) # returns (z,y,x,c)
            reconstructed = torch.permute(reconstructed, reconstructed_mask_order)
            reconstructed = torch.unsqueeze(reconstructed, 0) # make reconstructed image into (Batch,c,z,y,x)

    #         full_mask = full_mask.type(torch.int16).cpu()
            gt_mask = torch.permute(full_mask, dim_order).cpu() # roll axis of grount truth mask into (batch,c,z,y,x)
            
        return {"batch_loss":running_loss/count_loss, "y_pred":reconstructed, "y":gt_mask}


    # In[17]:

    trainer = Engine(train)
    evaluator = Engine(validate)

    # ## Metrics and Progress Bars

    # In[18]:

    # set up progress bar
    from ignite.contrib.handlers import ProgressBar
    from ignite.metrics import RunningAverage
    # from ignite.metrics import ConfusionMatrix, DiceCoefficient
    from monai.handlers.ignite_metric import IgniteMetric

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
    metric.attach(trainer,"Dice")
    metric.attach(evaluator,"Dice")

    # RunningAverage(output_transform=loss_output_transform).attach(trainer, "batch_loss")
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
        with open(model_directory+date_directory+time_directory+'results.txt', 'a') as results_file:
                results_file.write(f'Starting Fold #{fold_idx}')

    @trainer.on(Events.EPOCH_STARTED)
    def print_epoch(trainer):
        print("Epoch : {}".format(trainer.state.epoch))

    @trainer.on(Events.TERMINATE)
    def save_results_to_yaml(trainer):
        with open(model_directory+date_directory+time_directory+'exp_configs.yml', 'w') as exp_configs:
            yaml.dump(config, exp_configs, default_flow_style = False)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(trainer):
        global best_dice
        global best_epoch
        global best_epoch_file
        global best_loss
        
        epoch = trainer.state.epoch
        def get_saved_model_path(epoch):
            return model_directory + date_directory + time_directory + f"{model_name}_{fold_idx}_{epoch}.pth"

        # initialize global values
        best_dice = -torch.inf if epoch == 1 else best_dice
        best_loss = torch.inf if epoch == 1 else best_loss
        best_epoch = 1 if epoch == 1 else best_epoch
        best_epoch_file = '' if epoch == 1 else best_epoch_file
        
        def log_training_results(trainer):
            evaluator.run(training_dataloader)
            # Get engine metrics and losses
            training_metrics = copy.deepcopy(evaluator.state.metrics)
            # pbar.log_message(
            #     "Training Results - Epoch: {} \nMetrics\n{}"
            #     .format(trainer.state.epoch, pprint.pformat(training_metrics)))
            with open(model_directory+date_directory+time_directory+'results.txt', 'a') as results_file:
                results_file.write("Training Results - Epoch: {} \nMetrics\n{}\n".format(trainer.state.epoch, pprint.pformat(training_metrics)))
            return training_metrics
        
        def log_testing_results(trainer):
            evaluator.run(testing_dataloader)
            testing_metrics = copy.deepcopy(evaluator.state.metrics)
            # scheduler.step(testing_metrics["batch_loss"])
            scheduler.step()
            # pbar.log_message(
            #     "Validation Results - Epoch: {} \nMetrics\n{}"
            #     .format(trainer.state.epoch, pprint.pformat(testing_metrics)))
            with open(model_directory+date_directory+time_directory+'results.txt', 'a') as results_file:
                results_file.write("Validation Results - Epoch: {} \nMetrics\n{}\n".format(trainer.state.epoch, pprint.pformat(testing_metrics)))
            return testing_metrics
        
        training_metrics= log_training_results(trainer)
        testing_metrics= log_testing_results(trainer)
        
        train_dice = training_metrics['Dice']
        val_dice = testing_metrics['Dice']

        train_loss = training_metrics['batch_loss']
        val_loss = testing_metrics['batch_loss']
        
        # log results based off experiment
        log_writer.add_scalars('Training vs. Validation Loss',
                        {f'Training Fold {fold_idx}' : train_loss, f'Validation Fold {fold_idx}' : val_loss}, epoch)

        if segmentation_exp == '+s_+d_+f':
            train_mean_dice = torch.mean(train_dice[0:2])
            val_mean_dice = torch.mean(val_dice[0:2])

            log_writer.add_scalars('Training vs. Validation Soma Dice ',
                            {f'Training Soma Dice Fold {fold_idx}' : train_dice[0], f'Validation Soma Dice Fold {fold_idx}' : val_dice[0]}, epoch)
            
            log_writer.add_scalars('Training vs. Validation Dendrite Dice ',
                            {f'Training Dendrite Dice Fold {fold_idx}' : train_dice[1], f'Validation Dendrite Dice {fold_idx}' : val_dice[1]}, epoch)
            
            log_writer.add_scalars('Training vs. Validation Filopodias Dice ',
                            {f'Training Filopodias Dice Fold {fold_idx}' : train_dice[2], f'Validation Filopodias Dice {fold_idx}' : val_dice[2]}, epoch)
            
            log_writer.add_scalars('Training vs. Validation Mean Dice ',
                            {f'Training Mean Dice Fold {fold_idx}' : train_mean_dice, f'Validation Mean Dice Fold {fold_idx}' : val_mean_dice}, epoch)
            
            log_writer.flush()

        if segmentation_exp == '+s_+d_-f':
            train_mean_dice = torch.mean(train_dice[0:1])
            val_mean_dice = torch.mean(val_dice[0:1])

            log_writer.add_scalars('Training vs. Validation Soma Dice Fold',
                            {f'Training Soma Dice Fold {fold_idx}' : train_dice[0], f'Validation Soma Dice Fold {fold_idx}' : val_dice[0]}, epoch)
            
            log_writer.add_scalars(f'Training vs. Validation Dendrite Dice',
                            {f'Training Dendrite Dice Fold {fold_idx}' : train_dice[1], f'Validation Dendrite Dice {fold_idx}' : val_dice[1]}, epoch)

            log_writer.add_scalars('Training vs. Validation Mean Dice ',
                            {f'Training Mean Dice Fold {fold_idx}' : train_mean_dice, f'Validation Mean Dice Fold {fold_idx}' : val_mean_dice}, epoch)

            log_writer.flush()
        
        if segmentation_exp == '-s_+d_-f':
            log_writer.add_scalars('Training vs. Validation Neuron Dice ',
                            {f'Training Neuron Dice Fold {fold_idx}' : train_dice, f'Validation Neuron Dice Fold {fold_idx}' : val_dice}, epoch)
            
            log_writer.flush()

        if (testing_metrics['batch_loss'] < best_loss):
            
            # if there was a previous model saved, delete that one
            prev_best_epoch_file = get_saved_model_path(best_epoch)
            if os.path.exists(prev_best_epoch_file):
                os.remove(prev_best_epoch_file)

            # update the best mean dice and loss and save the new model state
    #         best_dice = val_mean_dice
            best_loss = testing_metrics['batch_loss']
            best_epoch = epoch
            best_epoch_file = get_saved_model_path(best_epoch)
    #         print(f'\nEpoch: {best_epoch} - New best Dice and Loss! Mean Dice: {best_dice} Loss: {best_loss}\n\n\n')
            # print(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
            with open(model_directory+date_directory+time_directory+'results.txt', 'a') as results_file:
                results_file.write(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
            torch.save(model.state_dict(), best_epoch_file)


    # ## Early Stopping
    # In[21]:

    # from ignite.handlers import EarlyStopping

    # def score_function(engine):
    #     val_loss = engine.state.metrics['batch_loss']
    #     return -val_loss

    # handler = EarlyStopping(patience=10, score_function=score_function, min_delta=1e-6, trainer=trainer)
    # evaluator.add_event_handler(Events.COMPLETED, handler)

    # In[ ]:

    # Running Training Engine

    trainer.run(training_dataloader, max_epochs = max_epochs)

# %%
