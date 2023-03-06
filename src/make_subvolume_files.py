#!/usr/bin/env python
# coding: utf-8

# ## Import Modules

# In[1]:


import os
import sys

from processing import processing_functions as pf

import glob
import os
import sys
import tifffile
import numpy as np
import h5py
import torch
from torchvision import transforms, utils
import yaml
import itertools
from torch.utils.data import DataLoader
from patchify import patchify
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help="configuration file *.yml", type=str)
parser.add_argument('-k', '--kfold', help="kfold file *.yml", type=str)
parser.add_argument('-td', '--training', help="directory for training method, i.e., 2D_training", type=str)
args = vars(parser.parse_args())
print(args)

# ## Create Raw and Image tiffs

# In[2]:


#dir = "C:/Users/Fungj/My Drive (haaslabdataimages@gmail.com)/Completed"
#file_type = "/*.h5"
#h5_list = glob.glob(dir+file_type)
#filename = [os.path.basename(f) for f in h5_list]


# ## Read in Completed h5 files

# In[3]:


#raw_folder = 'E:/Image_Folder/Raw/'
#mask_folder = 'E:/Image_Folder/Mask/'
#if not os.path.isdir(raw_folder) and not os.path.isdir(mask_folder):
#    os.mkdir(raw_folder)
#    os.mkdir(mask_folder)


#for file in filename:
#    root_ext = os.path.splitext(file) # split basename and extension
#    new_ext = '.tif'
#    if not os.path.exists(raw_folder + root_ext[0] + new_ext):
#        print('tif doesnt exist, writing new tif file')
#        hf = h5py.File(os.path.join(dir,file),'r')
#        raw = np.array(hf['project_data'].get('raw_image'))
#        mask = np.array(hf['project_data'].get('label'))
#        tifffile.imwrite(raw_folder + root_ext[0] + new_ext,raw)
#        tifffile.imwrite(mask_folder + root_ext[0] + new_ext,mask)
#    else:
#        print('tif file exists')


# ## Read In Configurations

# In[4]:


project_path = os.path.dirname(os.getcwd())
training_type = args['training']
exp_path = f"/config/convnets/ResUNet/{training_type}/"
exp_file = args['config']
# exp_file = "soma_dendrite.yml"
print(exp_file)
# exp_file = "post_hptune_exp.yml"

with open(project_path + exp_path + exp_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)  


# In[5]:


# get list of h5 files
#dir = "C:/Users/Fungj/My Drive (haaslabdataimages@gmail.com)/Completed"
#file_type = "/*.h5"
#h5_list = glob.glob(dir+file_type)
#filename = [f for f in h5_list]


# In[6]:


# read in segmentation_exp
segmentation_exp = config['DATASET']['exp'] # +s+d+f
lateral_steps = config['DATASET']['lateral_steps'] # integer
axial_steps = config['DATASET']['axial_steps'] # integer
patch_z, patch_y, patch_x = config['DATASET']['z_patch'], config['DATASET']['y_patch'], config['DATASET']['x_patch']
#split_size = config['DATASET']['split_size'] # integer 
ex_autofluor = config['DATASET']['ex_autofluorescence'] # True/False
ex_melanocytes = config['DATASET']['ex_melanocytes'] # True/False
remove_artifacts = config['DATASET']['remove_artifacts']
artifacts = config['DATASET']['artifacts']
spatial_dim = config['MODEL']['spatial_dim']
patch_size = (patch_z, patch_y, patch_x)

# In[7]:

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


# ## Configure kFold from config file

# In[8]:


kfold_path = "/config/convnets/ResUNet/kfold_training/kfold_indices/"
kfold_file = args['kfold']
#kfold_file = "fold_1.yml"

with open(project_path + kfold_path + kfold_file, "r") as stream:
    try:
        fold_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc) 

fold_idx = fold_config['fold_idx']
train_idx = fold_config['train_idx']
val_idx = fold_config['val_idx']


# In[38]:

if training_type == "changing_patch_experiments":
    patch_name = f"z_{patch_z}_y_{patch_y}_x_{patch_x}"
    parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/{training_type}/{segmentation_exp}/{patch_name}/"
else:
    parent_folder = f"/home/jsfung/projects/def-haas/jsfung/Images/new_labels/{training_type}/{segmentation_exp}/"

fold_folder = os.path.join(parent_folder,f"fold_{fold_idx}/")
target_raw_folder = os.path.join(fold_folder,"Raw/")
target_mask_folder = os.path.join(fold_folder,"Mask/")

raw_folder = "/home/jsfung/projects/def-haas/jsfung/Images/new_labels/Raw/"
mask_folder = "/home/jsfung/projects/def-haas/jsfung/Images/new_labels/Mask/"

# In[41]:


raw_filename_list = np.array(os.listdir(raw_folder))
mask_filename_list = np.array(os.listdir(mask_folder))

if not os.path.isdir(target_raw_folder) and not os.path.isdir(target_mask_folder):
    os.makedirs(target_raw_folder)
    os.makedirs(target_mask_folder)

# patch_transform = transforms.Compose([
# #                                       new_shape(new_xy = (600,960)),
#                                     pf.MinMaxScalerVectorized(),
#                                     pf.patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = False)])

# # define transforms for labeled masks
# label_transforms = transforms.Compose([
# #                                        new_shape(new_xy = (600,960)),
#                                     pf.process_masks(exp = segmentation_exp,
#                                                     ex_autofluor=ex_autofluor,
#                                                     ex_melanocytes=ex_melanocytes,
#                                                     ),
#                                     pf.patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = True)])


def patch_and_save_images(filename, spatial_dim, img, xy_steps, z_steps, patch_size, target_directory, is_mask = False):
    # Function patch_images splits a 3D image into patches such that for a non equal divisible z-depth,
    # it will return two arrays with one that holds the integer divisible number of z-stack layers, and 
    # the second is lower half that supposedly got "cut-off" to account for the remainder z-stack layers.

    # INPUT:
    # img: np.ndarray of size (z,x,y)
    # steps: int
    # patch_size: np.ndarray of size (z_patch, x_patch, y_patch)
    # spatial_dim: int, 2 or 3

    # OUTPUT:
    # quotient_array, remainder_array: np.ndarray(remainder*steps, img.shape[1], img.shape[2]), np.ndarray(steps, img.shape[1], img.shape[2])
    
    # get image shape
    img_shape = img.shape
    
    # get depth of image
    if spatial_dim == 3:
        z_tot = img.shape[0]
        z_patch = patch_size[0]
        y_patch = patch_size[1]
        x_patch = patch_size[2]

        quotient, remainder = divmod(z_tot,z_steps) # find the non-overlapping region "remainder"

        if is_mask == True:
            image_type = "Mask"
        else:
            image_type = "Raw"

        if remainder != 0:
            # non_empty_arr is a temp array that holds the integer divisible number of z-stack layers
            quotient_arr = np.zeros((quotient, img_shape[1] // xy_steps, img_shape[2] // xy_steps) + (z_patch, y_patch, x_patch)).astype(np.uint16)
            # empty_arr is the temp array that will hold the steps from end of the z-layer up to the step size i.e., 
            remainder_arr = np.zeros((1, img_shape[1] // xy_steps, img_shape[2] // xy_steps) + (z_patch, y_patch, x_patch)).astype(np.uint16)

            # custom method
            z_vox_lim, y_vox_lim, x_vox_lim = quotient_arr.shape[0:3]
            for k in range(z_vox_lim-1):
                for i in range(y_vox_lim-1):
                    for j in range(x_vox_lim-1):
                        quotient_img = img[z_steps*k:(z_patch+z_steps*(k)), xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]
                        tifffile.imwrite(os.path.join(target_directory,filename+f"_{image_type}_upper_{k}_{i}_{j}.tif"),quotient_img)

            # patch the "remainder section"
            for i in range(y_vox_lim-1):
                for j in range(x_vox_lim-1):
                    remainder_img = img[-z_patch:, xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]
                    tifffile.imwrite(os.path.join(target_directory,filename+f"_{image_type}_lower_0_{i}_{j}.tif"),remainder_img)

        # if there is no remainder, then return the quotient array and None for remainder
        else:
            quotient_arr = np.zeros((quotient, img_shape[1] // xy_steps, img_shape[2] // xy_steps) + (z_patch, y_patch, x_patch)).astype(np.uint16)
            z_vox_lim, x_vox_lim, y_vox_lim = quotient_arr.shape[0:3]
            for k in range(z_vox_lim-1):
                for i in range(y_vox_lim-1):
                    for j in range(x_vox_lim-1):
                        quotient_img = img[z_steps*k:(z_patch+z_steps*(k)), xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]
                        tifffile.imwrite(os.path.join(target_directory,filename+f"_{image_type}_upper_{k}_{i}_{j}.tif"),quotient_img)
    
    elif spatial_dim == 2:
        for z in range(img_shape[0]-1):
            sliced_img = img[z,...]
            tifffile.imwrite(os.path.join(target_directory,filename+f"_slice_{z}.tif"), sliced_img)

raw_training_list, mask_training_list = raw_filename_list[train_idx], mask_filename_list[train_idx]
# raw_testing_list, mask_testing_list = raw_filename_list[val_idx], mask_filename_list[val_idx]
for (raw_name, mask_name) in zip(raw_training_list, mask_training_list):
    # read raw and mask imgs
    print("Processing: ",raw_name)
    raw_img = tifffile.imread(os.path.join(raw_folder,raw_name)).astype(np.float32)
    mask_img = tifffile.imread(os.path.join(mask_folder,mask_name)).astype(np.uint16)

    normalizer = pf.MinMaxScalerVectorized()
    mask_processor = pf.process_masks(exp = segmentation_exp,
                                      ex_autofluor = ex_autofluor,
                                      ex_melanocytes = ex_melanocytes)

    processed_raw_img = normalizer(raw_img)
    processed_mask_img = mask_processor(mask_img)
    
    basefile_name = os.path.splitext(raw_name)

    patch_and_save_images(filename = basefile_name[0],
                           spatial_dim = spatial_dim,
                           img = processed_raw_img, 
                           xy_steps = lateral_steps, 
                           z_steps = axial_steps, 
                           patch_size = patch_size, 
                           target_directory = target_raw_folder, 
                           is_mask = False)

    patch_and_save_images(filename = basefile_name[0],
                           spatial_dim = spatial_dim, 
                           img = processed_mask_img, 
                           xy_steps = lateral_steps, 
                           z_steps = axial_steps, 
                           patch_size = patch_size, 
                           target_directory = target_mask_folder, 
                           is_mask = True)
    


# In[ ]:




