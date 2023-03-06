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
parser.add_argument('-k', '--kfold', help="kfold file *.yml", type=str)
parser.add_argument('-td', '--trainingdir', help="directory for training type", type=str)
args = vars(parser.parse_args())
print(args)

from processing.processing_functions import *
from src.train_engine.models import *

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

training_mode = "preprocessed_subpatch_image"

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

# Define shifting windows inferencing conditions
batch_size = config['DATASET']['batch_size'] # integer
lateral_steps = config['DATASET']['lateral_steps'] # integer
axial_steps = config['DATASET']['axial_steps'] # integer

patch_x = config['DATASET']['x_patch']
patch_y = config['DATASET']['y_patch']
patch_z = config['DATASET']['z_patch']
spatial_dim = config['MODEL']['spatial_dim']
batch_size = config['DATASET']['batch_size']

if config['MODEL']['spatial_dim'] == 3:
    patch_size = (patch_z, patch_y, patch_x)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=batch_size)
if config['MODEL']['spatial_dim'] == 2:
    patch_size = (patch_y, patch_x)
    inferer = SliceInferer(roi_size=patch_size, sw_batch_size=1, cval=-1, progress=True)


# results path
model_name = config['model_name']
date = datetime.now(tz=pytz.utc).strftime('%Y%m%d')
time = datetime.now(tz=pytz.utc).strftime('%H%M%S')

model_directory = project_path + f"results/{model_name}/"
date_directory = f"{date}/"
exp_directory = f"{patch_z}_{patch_y}_{patch_x}_{segmentation_exp}/"
time_directory = f"{date}_{time}/"

# In[20]:
# set up kfold experiments, read in yml files with their fold_idx, train_idx, and val_idx, 
# indicating which sets of images to use for training
kfold_path = f"config/{net_type}/{model_arch}/{training_dir}/kfold_indices/"
kfold_file = args['kfold']

with open(project_path + kfold_path + kfold_file, "r") as stream:
    try:
        fold_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc) 

fold_idx = fold_config['fold_idx']
train_idx = fold_config['train_idx']
val_idx = fold_config['val_idx']

experimental_path = model_directory+date_directory+exp_directory+f'{date}_{time}_{model_name}_experiment.yml'

if not os.path.exists(experimental_path):
    os.makedirs(os.path.dirname(experimental_path), exist_ok=True)
    with open(experimental_path, 'w') as outfile:
        config['date'] = f"{date}_{time}"
        yaml.dump(config, outfile, default_flow_style=False)

# remove artifacts
remove_artifacts = config['DATASET']['remove_artifacts']

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
# set up log directory:

log_directory = model_directory + date_directory + exp_directory + f"{time}_log_fold_{fold_idx}"
results_txt = model_directory+date_directory+exp_directory+f'{time}_results_fold_{fold_idx}.txt'

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
    parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/{training_dir}/{segmentation_exp}/"

fold_folder = os.path.join(parent_folder,f"fold_{fold_idx}/")
training_raw_folder = os.path.join(fold_folder,"Raw/*.tif")
training_mask_folder = os.path.join(fold_folder,"Mask/*.tif")
training_raw_list = glob.glob(training_raw_folder)
training_mask_list = glob.glob(training_mask_folder)


# define transforms for labeled masks
image_transform = transforms.Compose([MinMaxScalerVectorized()])
label_transform = transforms.Compose([process_masks(exp = segmentation_exp,
                                                     ex_autofluor=ex_autofluor,
                                                     ex_melanocytes=ex_melanocytes,
                                                     )])

training_dataset = SubVolumeDataset(raw_directory = training_raw_list,
                                    mask_directory = training_mask_list,
                                    num_classes = num_classes,
                                    raw_transform = image_transform,
                                    device = device)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

minibatch_mean = []
minibatch_std = []
for i, batch in enumerate(training_dataloader):
    raw_img, _ = batch
    data = raw_img.view(raw_img.size(0), raw_img.size(1), -1)

    # calculate the mean and std over the minibatch for each channel
    mean = torch.mean(data, dim=2)
    std = torch.std(data, dim=2)

    # take the mean of the means and stds over the batch dimension
    mean = torch.mean(mean, dim=0)
    std = torch.mean(std, dim=0)
    
    # store the population m
    minibatch_mean.append(mean)
    minibatch_std.append(std)

population_mean = np.mean(np.array(minibatch_mean))
population_std = np.mean(np.array(minibatch_std))

print(population_mean)
print(population_std)

    

