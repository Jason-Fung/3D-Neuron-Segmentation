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
parser.add_argument('-k', '--kfold', help="kfold file *.yml", type=str)
parser.add_argument('-d', '--dir', help="directory of training data", type=str)
parser.add_argument('-td', '--trainingdir', help="directory for training type", type=str)
args = vars(parser.parse_args())
print(args)

from processing.processing_functions import *

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
device = idist.device()
#device = torch.device("cuda:0")

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
training_type = args['trainingdir']
exp_path = f"/config/convnets/ResUNet/{training_type}/"
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

# read in segmentation_exp
segmentation_exp = config['DATASET']['exp'] # +s_+d_+f

# results path
model_name = config['model_name']
date = datetime.now(tz=pytz.utc).strftime('%Y%m%d')
time = datetime.now(tz=pytz.utc).strftime('%H%M%S')

model_directory = project_path + f"results/{model_name}/"
date_directory = f"{date}/"
exp_directory = f"{segmentation_exp}/"
time_directory = f"{date}_{time}/"

# In[20]:

experimental_path = model_directory+date_directory+exp_directory+'experiment.yml'

if not os.path.exists(experimental_path):
    os.makedirs(os.path.dirname(experimental_path), exist_ok=True)
    with open(experimental_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

# Define shifting windows inferencing conditions
batch_size = config['DATASET']['batch_size'] # integer
lateral_steps = config['DATASET']['lateral_steps'] # integer
axial_steps = config['DATASET']['axial_steps'] # integer

patch_x = config['DATASET']['x_patch']
patch_y = config['DATASET']['y_patch']
patch_z = config['DATASET']['z_patch']


if config['MODEL']['spatial_dim'] == 3:
    patch_size = (patch_z, patch_y, patch_x)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=batch_size)
if config['MODEL']['spatial_dim'] == 2:
    patch_size = (patch_y, patch_x)
    inferer = SliceInferer(roi_size=patch_size, sw_batch_size=1, spatial_dim=0)

# remove artifacts
remove_artifacts = config['DATASET']['remove_artifacts']
artifacts = config['DATASET']['artifacts']

# define experimental conditions
ex_autofluor = config['DATASET']['ex_autofluorescence'] # True/False
ex_melanocytes = config['DATASET']['ex_melanocytes'] # True/False
if config['MODEL']['spatial_dim'] == 3:
    dim_order = (0,4,1,2,3) # define the image and mask dimension order
if config['MODEL']['spatial_dim'] == 2:
    dim_order = (0,3,1,2)

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

# In[11]:

# set up kfold experiments, read in yml files with their fold_idx, train_idx, and val_idx, 
# indicating which sets of images to use for training
kfold_path = "config/convnets/ResUNet/kfold_training/kfold_indices/"
kfold_file = args['kfold']

with open(project_path + kfold_path + kfold_file, "r") as stream:
    try:
        fold_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc) 

fold_idx = fold_config['fold_idx']
train_idx = fold_config['train_idx']
val_idx = fold_config['val_idx']

# set up loss and optimizer
max_epochs = config['max_epochs']
dropout = config['MODEL']['dropout']
learning_rate = config['MODEL']['learning_rate']
l2 = config['MODEL']['l2']
norm_type = config['MODEL']['norm']
input_chnl = config['MODEL']['input_dim']
output_chnl = num_classes
spatial_dim = config['MODEL']['spatial_dim']
enc_dec_channels = config['MODEL']['channel_layers']
conv_strides = config['MODEL']['strides']
res_units = config['MODEL']['num_res_units']

# Locate raw sub volume image and sub volume mask file locations and return its list of file names for images 
parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/{training_type}/{segmentation_exp}/"
fold_folder = os.path.join(parent_folder,f"fold_{fold_idx}/")
training_raw_folder = os.path.join(fold_folder,"Raw/*.tif")
training_mask_folder = os.path.join(fold_folder,"Mask/*.tif")
training_raw_list = glob.glob(training_raw_folder)
training_mask_list = glob.glob(training_mask_folder)

# locate whole volume raw and masked image file locations and return its list of filenames according to the 
# predetermined fold train_idx and val_idx.
validation_raw_folder = "/home/jsfung/projects/def-haas/jsfung/Images/new_labels/Raw/*.tif"
validation_mask_folder = "/home/jsfung/projects/def-haas/jsfung/Images/new_labels/Mask/*.tif"
validation_raw_list = np.array(glob.glob(validation_raw_folder))[val_idx]
validation_mask_list = np.array(glob.glob(validation_mask_folder))[val_idx]

test_training_raw_list = np.array(glob.glob(validation_raw_folder))[train_idx]
test_training_mask_list = np.array(glob.glob(validation_mask_folder))[train_idx]

#from pathlib import Path

#for i in range(len(validation_raw_list)):
#    word_of_interest = Path(validation_raw_list[i]).stem
#
#    if any(word_of_interest in word for word in training_raw_list):
#        print(True)
#    else:
#        print(False)

# define transforms for labeled masks
image_transform = transforms.Compose([MinMaxScalerVectorized()])
label_transform = transforms.Compose([process_masks(exp = segmentation_exp,
                                                     ex_autofluor=ex_autofluor,
                                                     ex_melanocytes=ex_melanocytes,
                                                     )])

def training(rank, config):

    log_directory = model_directory + date_directory + exp_directory + f"{time}_worker_{rank}_log_fold_{fold_idx}"
    results_txt = model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt'

    os.makedirs(os.path.dirname(results_txt), exist_ok=True)
    os.makedirs(log_directory)

    # create writer to log results into tensorboard
    log_writer = SummaryWriter(log_directory)

    # # Define Model and Parameters

    # ### Model: ResUNet

    max_epochs = config['max_epochs']
    dropout = config['MODEL']['dropout']
    learning_rate = config['MODEL']['learning_rate']
    l2 = config['MODEL']['l2']
    norm_type = config['MODEL']['norm']
    input_chnl = config['MODEL']['input_dim']
    output_chnl = num_classes
    spatial_dim = config['MODEL']['spatial_dim']
    enc_dec_channels = config['MODEL']['channel_layers']
    conv_strides = config['MODEL']['strides']
    res_units = config['MODEL']['num_res_units']

    model = UNet(spatial_dims=spatial_dim,
                in_channels = input_chnl,
                out_channels = output_chnl,
                channels = enc_dec_channels,
                strides=conv_strides,
                num_res_units=res_units,
                norm = norm_type,
                dropout = dropout)

    model = idist.auto_model(model)


    training_dataset = SubVolumeDataset(raw_directory = training_raw_list,
                                        mask_directory = training_mask_list,
                                        num_classes = num_classes,
                                        device = device)
    
    test_training_dataset = WholeVolumeDataset(raw_directory = test_training_raw_list,
                                            mask_directory = test_training_mask_list,
                                            num_classes = num_classes,
                                            raw_transform = image_transform,
                                            label_transform = label_transform,
                                            mask_order=(0,4,1,2,3),
                                            model_spatial_dim = spatial_dim,
                                            device = device)


    validation_dataset = WholeVolumeDataset(raw_directory = validation_raw_list,
                                            mask_directory = validation_mask_list,
                                            num_classes = num_classes,
                                            raw_transform = image_transform,
                                            label_transform = label_transform,
                                            mask_order=(0,4,1,2,3),
                                            model_spatial_dim = spatial_dim,
                                            device = device)

    training_dataloader = idist.auto_dataloader(training_dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory = False)
    test_training_dataloader = idist.auto_dataloader(test_training_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory = False)
    validation_dataloader = idist.auto_dataloader(validation_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory = False)

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

    with open(model_directory+date_directory+f'{time}_results_fold_{fold_idx}.txt', 'w') as results_file:
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
    optimizer = idist.auto_optim(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs, verbose = True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience = 10, threshold=1e-5, threshold_mode= 'abs', verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # discretize = Compose([Activations(softmax = True), 
    #                       AsDiscrete(logit_thresh=0.5)])


    # ## Define Training and Validation Functions

    # In[15]:

    def train(engine, batch):
        augment = config['DATASET']['AUGMENTATION']['augment']

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
                aug_imgs, aug_masks = augmentation(all_transform, spatial_dim, raw_dict, mask_dict, non_empty_indices)
                
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

# In[16]:

    def validate(engine, batch):

        model.eval()
        with torch.no_grad():
            running_loss = 0
            count_loss = 0

            raw, mask = batch
            mask = torch.squeeze(mask, dim=0)
            
            print(raw.max())

            output = inferer(inputs = raw, network = model)
            probabilities = torch.softmax(output, 1)
            prediction = torch.argmax(probabilities,1)
            print(torch.unique(prediction))
            prediction = torch.permute(to_categorical_torch(prediction, num_classes), (0,4,1,2,3))
            

            current_loss = loss_function(probabilities, mask) # + dice_loss(predictions, patch_gt)
            running_loss += current_loss.item()
            count_loss += 1

            mask = mask.to("cpu")

        return {"batch_loss":running_loss/count_loss, "y_pred":prediction, "y": mask}


# In[17]:

    trainer = Engine(train)
    evaluator = Engine(validate)

# ## Metrics and Progress Bars

# In[18]:

# set up progress bar

    def metric_output_transform(output):
        y_pred, y = output["y_pred"], output["y"]
        print(y_pred.shape)
        print(y.shape)
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
        with open(model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt', 'w') as results_file:
                results_file.write(f'Starting Fold #{fold_idx}')

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
            return model_directory + date_directory + exp_directory+f"{segmentation_exp}_{model_name}_{fold_idx}_{epoch}.pth"

        # initialize global values
        best_dice = -torch.inf if epoch == 1 else best_dice
        best_loss = torch.inf if epoch == 1 else best_loss
        best_epoch = 1 if epoch == 1 else best_epoch
        best_epoch_file = '' if epoch == 1 else best_epoch_file
        
        def log_training_results(trainer):
            evaluator.run(test_training_dataloader)
            # Get engine metrics and losses
            training_metrics = copy.deepcopy(evaluator.state.metrics)
            #pbar.log_message("Training Results - Epoch: {} \nMetrics\n{}".format(trainer.state.epoch, pprint.pformat(training_metrics)))
            with open(model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt', 'a') as results_file:
                results_file.write("Training Results - Epoch: {} \nMetrics\n{}\n".format(trainer.state.epoch, pprint.pformat(training_metrics)))
            return training_metrics
        
        def log_testing_results(trainer):
            evaluator.run(validation_dataloader)
            testing_metrics = copy.deepcopy(evaluator.state.metrics)
            # scheduler.step(testing_metrics["batch_loss"])
            scheduler.step()
            #pbar.log_message("Validation Results - Epoch: {} \nMetrics\n{}".format(trainer.state.epoch, pprint.pformat(testing_metrics)))
            with open(model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt', 'a') as results_file:
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
            with open(model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt', 'a') as results_file:
                results_file.write(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
            torch.save({'model_state_dict': model.state_dict(), 
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': best_epoch
                        },best_epoch_file)

    trainer.run(training_dataloader, max_epochs = max_epochs)

if __name__ == "__main__":
    set_start_method("spawn")
    backend = "nccl"
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(training, config)

# %%
