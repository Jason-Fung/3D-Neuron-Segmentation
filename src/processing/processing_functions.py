import tensorflow as tf
import numpy as np
import torch
import tifffile
from tqdm import trange
from torch.utils.data import Dataset
from patchify import patchify, unpatchify
import torchio
from sklearn import preprocessing
import os
import onnxruntime

# list of classes for preprocessing and transforming images
class MyImageDataset(Dataset):
    def __init__(self, 
                 raw_list = None, 
                 mask_list = None, 
                 remove_artifacts = False,
                 artifacts = [],
                 transform = None,
                 label_transform = None, 
                 device = None, 
                 img_order = (0,1,2,3,4),
                 mask_order = (0,1,2,3,4),
                 num_classes = 2,
                 train = True):
        # raw_list: list of full paths to the raw images
        # mask_list: list of full paths to the masked images
        # transform: list of composed transforms for the raw images
        # label_transform: list of composed transforms for the masked images
        # train = True for training, False for test
        
        self.mask_list = mask_list
        self.raw_list = raw_list
        self.remove_artifacts = remove_artifacts
        self.artifacts = artifacts
        self.transform = transform
        self.label_transform = label_transform
        self.img_order = img_order
        self.mask_order = mask_order
        self.device = device
        self.num_classes = num_classes
#         self.target_transform = target_transform
        
    def __len__(self):
        return len(self.raw_list)
    
    def __getitem__(self,idx):
        if self.transform:
            x = tifffile.imread(self.raw_list[idx]).astype(np.float64) # raw image
            if self.remove_artifacts == True:
                y = tifffile.imread(self.mask_list[idx]).astype(np.uint16) # labeled image
                for classes in self.artifacts:
                    x[y == classes] = 0 # erase the artifacts in the raw image 

            upper_img, upper_shape, lower_img, lower_shape = self.transform(x)
            if lower_img is None:
                upper, lower = torch.FloatTensor(upper_img).to(self.device), None 
                upper = torch.unsqueeze(upper, -1)
                upper = torch.permute(upper, self.img_order)
                mask, upper_mask, lower_mask = [], dict(), dict()
            else:
                upper, lower = torch.FloatTensor(upper_img).to(self.device), torch.FloatTensor(lower_img).to(self.device)
                # upper, lower = torch.FloatTensor(upper_img), torch.FloatTensor(lower_img)
                upper, lower = torch.unsqueeze(upper, -1), torch.unsqueeze(lower, -1)
                upper, lower = torch.permute(upper, self.img_order), torch.permute(lower, self.img_order) 
                mask, upper_mask, lower_mask = [], dict(), dict()
        
        if self.label_transform is not None:
            y = tifffile.imread(self.mask_list[idx]) # labeled image
            mask, upper_mask, lower_mask = self.label_transform(y)
            if lower_mask is None:
                mask, upper_mask, lower_mask = to_categorical(mask, self.num_classes), to_categorical(upper_mask, self.num_classes), None
                mask, upper_mask = torch.FloatTensor(mask), torch.FloatTensor(upper_mask).to(self.device)
                # mask, upper_mask, lower_mask = torch.FloatTensor(mask) , torch.FloatTensor(upper_mask), torch.FloatTensor(lower_mask)
                upper_mask = torch.permute(upper_mask, self.mask_order)
            else:
                mask, upper_mask, lower_mask = to_categorical(mask, self.num_classes), to_categorical(upper_mask, self.num_classes), to_categorical(lower_mask, self.num_classes)
                mask, upper_mask, lower_mask = torch.FloatTensor(mask), torch.FloatTensor(upper_mask).to(self.device), torch.FloatTensor(lower_mask).to(self.device)
                # mask, upper_mask, lower_mask = torch.FloatTensor(mask), torch.FloatTensor(upper_mask), torch.FloatTensor(lower_mask)
                upper_mask, lower_mask = torch.permute(upper_mask, self.mask_order), torch.permute(lower_mask, self.mask_order)
        
        if lower_img is None:
            return dict(zip(list(range(len(upper))), upper)), upper_shape, [], [], mask, dict(zip(list(range(len(upper_mask))),upper_mask)), []
        else:
            return dict(zip(list(range(len(upper))), upper)), upper_shape, dict(zip(list(range(len(lower))),lower)), lower_shape, mask, dict(zip(list(range(len(upper_mask))),upper_mask)), dict(zip(list(range(len(lower_mask))),lower_mask))
        

class SubVolumeDataset(Dataset):
    # this Dataset module assumes that you have already preprocessed your images and stored them 
    # into a specified folder.

    def __init__(self, raw_directory, mask_directory, num_classes, device):
        self.img_list = raw_directory
        self.mask_list = mask_directory
        self.num_classes = num_classes
        # self.img_order = img_order
        # self.mask_order = mask_order
        self.device = device

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        # read and process the raw image
        image = tifffile.imread(self.img_list[idx]).astype(np.float32) # shape = (Batch Size, Z_size, Y_size, X_size)
        image = torch.FloatTensor(image).to(self.device) # type = np.array
        # image = torch.permute(image, self.img_order) # (Batch Size, Channel, Z_size, Y_size, X_size)
        # print(self.img_list[idx])

        # read and process the image mask
        mask = tifffile.imread(self.mask_list[idx]).astype(np.uint16)
        unique_classes = np.unique(mask)
        # print(f"Mask Patched Name: {self.mask_list[idx]} with unique categories: {unique_classes}")
        mask = torch.FloatTensor(to_categorical(mask, self.num_classes)).to(self.device)

        # print(self.mask_list[idx])
        return image, mask

class WholeVolumeDataset(Dataset):
    # this Dataset module accepts a list of raw image filenames and reads in the image one at a time

    def __init__(self, raw_directory = [],
                 raw_img = None, 
                 mask_directory = [], 
                 num_classes = None, 
                 raw_transform = None, 
                 label_transform = None, 
                 mask_order = (0,4,1,2,3), 
                 device = "cpu"):

        self.raw_img_list = raw_directory
        self.raw_img = raw_img
        self.mask_list = mask_directory
        self.num_classes = num_classes
        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.mask_order = mask_order
        self.device = device

    def __len__(self):
        if self.raw_img_list != None:
            return len(self.raw_img_list)
        else:
            return 0

    def __getitem__(self, idx):

        # read and process the raw image
        if self.raw_img is None: # check if there is already a raw image
            print("reading from list")
            image = tifffile.imread(self.raw_img_list[idx]).astype(np.float32) # raw image shape = (Batch Size, Z_size, Y_size, X_size)
        else: # pick from a string instead
            print("reading in existing image")
            idx = 0
            image = self.raw_img.astype(np.float16) # shape = (Batch Size, Z_size, Y_size, X_size)

        if self.raw_transform:
            image = self.raw_transform(image)
            image = torch.FloatTensor(image).to(self.device) # type = np.array
            image = torch.unsqueeze(image, dim =0)
        # image = torch.permute(image, self.img_order) # (Batch Size, Channel, Z_size, Y_size, X_size)
        # print(self.img_list[idx])

        if self.mask_list != []:
            print("Reading Mask from list")
            # read and process the image mask
            mask = tifffile.imread(self.mask_list[idx]).astype(np.uint16)
            if self.label_transform:
                mask = self.label_transform(mask)
        
            mask = torch.FloatTensor(to_categorical(mask, self.num_classes)).to(self.device)
            mask = torch.unsqueeze(mask,0)
            mask = torch.permute(mask, self.mask_order)
        else:
            mask = None
        # print(self.mask_list[idx])
        return image, mask

class MinMaxScalerVectorized(object):
    # def __call__(self,tensor):
    #     dist = (tensor.max() - tensor.min())
    #     dist[dist==0.0] = 1.0
    #     scale = 1.0 / dist
    #     tensor.mul_(scale).sub_(tensor.min())
    #     return tensor
    
    def __call__(self, array):
        return normalize(array)
    
class patch_imgs(object):
    def __init__(self, xy_step = 32, z_step = 16, patch_size = (16,32,32), remove_artifacts = True, artifacts = None, is_mask = False):
        self.patch_size = patch_size
        self.xy_step = xy_step
        self.z_step = z_step
        self.remove_artifacts = remove_artifacts
        self.artifacts = None
        self.is_mask = is_mask
        
    def __call__(self, array):
        return patch_images(array, self.xy_step, self.z_step, self.patch_size, self.is_mask)

class process_masks(object):
    def __init__(self, exp = '+s_+d_+f', ex_autofluor = True, ex_melanocytes = True, ignore_starting_from = 7):
        # exp or experiment = 's+f+d' or 'f+d'() or 'f+d+af', where s = soma; f = filopodia; d = dendrite
        self.ignore_starting_from = ignore_starting_from
        self.ex_autofluor = ex_autofluor
        self.ex_melano = ex_melanocytes
        self.exp = exp
    def __call__(self, array):
        # assuming the axon is labeled, make it a dendrite (2)
        axon_idx_z,axon_idx_x,axon_idx_y = np.where(array == 4)
        array[axon_idx_z,axon_idx_x,axon_idx_y] = 2

        if self.exp == '+s_+d_+f': # train model to segment soma, dendrite, and filopodias (semantic segmentation)

            # process noise artifacts
            if self.ex_autofluor: # exclude autofluorescence
                array[array == 6] = 0
                if self.ex_melano:
                    array[array == 7] = 0
                else:
                    array[array == 7] = 4
            else:
                array[array == 6] = 4
                if self.ex_melano:
                    array[array == 7] = 4 # make melanocyte as an autofluorescence
                else:
                    array[array == 7] = 5 # otherwise keep melanocyte as its own class (7->5 because models require contingent 0...n classes
            
            idx_z,idx_x,idx_y = np.where(array > self.ignore_starting_from)
            array[idx_z,idx_x,idx_y] = 0 # make everything else 0
            return array

        if self.exp == '-s_+d_-f': # train model to detect background vs foreground (binary segmentation)

            soma_idx_z,soma_idx_x,soma_idx_y = np.where(array == 2)
            array[soma_idx_z,soma_idx_x,soma_idx_y] = 1 # change dendrites to soma (vice versa)

            filo_idx_z,filo_idx_x,filo_idx_y = np.where(array == 3) 
            array[filo_idx_z,filo_idx_x,filo_idx_y] = 1 # change filopodias to soma

            # process noise artifacts
            if self.ex_autofluor: # exclude autofluorescence
                array[array == 6] = 0
                if self.ex_melano:
                    array[array == 7] = 0
                else:
                    array[array == 7] = 2
            else:
                array[array == 6] = 2
                if self.ex_melano:
                    array[array == 7] = 2
                else:
                    array[array == 7] = 3

            idx_z,idx_x,idx_y = np.where(array > self.ignore_starting_from)
            array[idx_z,idx_x,idx_y] = 0 # make everything else 0

            return array
        
        if self.exp == '+s_+d_-f': # Yes soma, Yes dendrite, No filopodia (merge filopodia to dendrite)
            filo_idx_z,filo_idx_x,filo_idx_y = np.where(array == 3)
            array[filo_idx_z,filo_idx_x,filo_idx_y] = 2

            # process noise artifacts
            if self.ex_autofluor: # exclude autofluorescence
                array[array == 6] = 0
                if self.ex_melano:
                    array[array == 7] = 0
                else:
                    array[array == 7] = 3
            else:
                array[array == 6] = 3
                if self.ex_melano:
                    array[array == 7] = 3
                else:
                    array[array == 7] = 4

            idx_z,idx_x,idx_y = np.where(array > self.ignore_starting_from)
            array[idx_z,idx_x,idx_y] = 0 # make everything else 0
            return array

class one_hot_encode(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def __call__(self, array):
        return to_categorical_torch(array, self.num_classes)
        
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def to_torch(np_arr):
    return torch.FloatTensor(np.array(np_arr))

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    #y = np.clip(y.astype(np.uint16), 0, num_classes - 1)
    return np.eye(num_classes, dtype = int)[y.astype(int)]


def to_categorical_torch(y, num_classes):
    return torch.eye(num_classes, dtype = bool)[y]

def normalize(input_image):
    input_image = input_image/input_image.max()
    return input_image

class new_shape(object):
    def __init__(self, new_xy = (600,960)):
        # new_xy: tuple(new y dimension, new x dimension). Example (600, 960)
        self.new_xy = new_xy
        
    def __call__(self, array):
        return place_into_center(array, self.new_xy)

def place_into_center(input_img, new_xy):
    # input_img = ndarray of size (z,y,x)
    
    # set up limits
    input_z_shape = input_img.shape[0]
    input_y_shape = input_img.shape[1]
    input_x_shape = input_img.shape[2]
    output_img = np.empty((input_z_shape, new_xy[0], new_xy[1]))
    
    # find the center of x direction
    new_x_half = np.uint16(new_xy[0]//2)
    new_y_half = np.uint16(new_xy[1]//2)
    
    old_x_half = np.uint16(input_x_shape//2)
    old_y_half = np.uint16(input_y_shape//2)
    
    output_img[:,new_x_half-old_x_half:new_x_half+old_x_half,new_y_half-old_y_half:new_y_half+old_y_half] = input_img
    return output_img

def patch_images(img, xy_steps, z_steps, patch_size, is_mask = False):
    # Function patch_images splits a 3D image into patches such that for a non equal divisible z-depth,
    # it will return two arrays with one that holds the integer divisible number of z-stack layers, and 
    # the second is lower half that supposedly got "cut-off" to account for the remainder z-stack layers.

    # INPUT:
    # img: np.ndarray of size (z,x,y)
    # steps: int
    # patch_size: np.ndarray of size (z_patch, x_patch, y_patch)

    # OUTPUT:
    # quotient_array, remainder_array: np.ndarray(remainder*steps, img.shape[1], img.shape[2]), np.ndarray(steps, img.shape[1], img.shape[2])
    
    # get image shape
    img_shape = img.shape
    
    # get depth of image
    z_tot = img.shape[0]
    z_patch = patch_size[0]
    y_patch = patch_size[1]
    x_patch = patch_size[2]

    quotient, remainder = divmod(z_tot,z_steps) # find the non-overlapping region "remainder"
    if is_mask == False:
        data_type = np.float32
    else:
        data_type = np.int16

    if remainder != 0:
        # non_empty_arr is a temp array that holds the integer divisible number of z-stack layers
        quotient_arr = np.zeros((quotient, img_shape[1] // xy_steps, img_shape[2] // xy_steps) + (z_patch, y_patch, x_patch)).astype(data_type)
        # empty_arr is the temp array that will hold the steps from end of the z-layer up to the step size i.e., 
        remainder_arr = np.zeros((1, img_shape[1] // xy_steps, img_shape[2] // xy_steps) + (z_patch, y_patch, x_patch)).astype(data_type)

        # custom method
        z_vox_lim, y_vox_lim, x_vox_lim = quotient_arr.shape[0:3]
        for k in range(z_vox_lim-1):
            for i in range(y_vox_lim-1):
                for j in range(x_vox_lim-1):
                    quotient_arr[k, i, j, :,:,:] = img[z_steps*k:(z_patch+z_steps*(k)), xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]

        # patch the "remainder section"
        for i in range(y_vox_lim-1):
            for j in range(x_vox_lim-1):
                remainder_arr[0,i,j,:,:,:] = img[-z_patch:, xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]

        
        # reshape to (quotient * img_shape[1] // x_patch * img_shape[2] // y_patch, z, x, y)
        patched_quotient_arr = quotient_arr.reshape(-1, *quotient_arr.shape[-3:]) 
        patched_remainder_arr = remainder_arr.reshape(-1, *remainder_arr.shape[-3:])

        # patched_data_set = np.vstack((patched_quotient_arr, patched_remainder_arr))
        # print(patched_quotient_arr.shape)
        if is_mask == False:
            return patched_quotient_arr, quotient_arr.shape, patched_remainder_arr, remainder_arr.shape
        else:
            return img, patched_quotient_arr, patched_remainder_arr
    
    # if there is no remainder, then return the quotient array and None for remainder
    else:
        quotient_arr = np.zeros((quotient, img_shape[1] // y_patch, img_shape[2] // x_patch) + (z_patch, y_patch, x_patch)).astype(data_type)
        remainder_arr = None
        z_vox_lim, x_vox_lim, y_vox_lim = quotient_arr.shape[0:3]
        for k in range(z_vox_lim):
            for i in range(y_vox_lim):
                for j in range(x_vox_lim):
                    quotient_arr[k, i, j, :,:,:] = img[z_steps*k:(z_patch+z_steps*(k)), xy_steps*i:(y_patch + xy_steps*(i)), xy_steps*j:(x_patch + xy_steps*(j))]
        
        patched_quotient_arr = quotient_arr.reshape(-1, *quotient_arr.shape[-3:]) 
        # patched_remainder_arr = remainder_arr.reshape(-1, *quotient_arr.shape[-3:])
        
        if is_mask == False:
            # print(patched_quotient_arr.shape)
            return patched_quotient_arr, quotient_arr.shape, remainder_arr, ()
        else:
            return img, patched_quotient_arr, remainder_arr
    

def reconstruct_training_masks(upper, lower, upper_size, lower_size, patch_size, orig_shape):
    
    # Reconstruct the image from batches (Batches, Channel, D, W, H)
    # upper: Tensor representing an image that is evenly split in the z-direction Tensor(Batches, D, W, H)
    # lower: Tensor representing an image that is leftover from the split in the z-direction Tensor(Batches, D, W, H)    
    # upper_size: Tensor representing the original shape of the split image in the form (z_loc, y_loc, x_loc, z_patch, y_patch, x_patch)
    # lower_size: Tensor representing the original shape of the split image in the form (z_loc, y_loc, x_loc, z_patch, y_patch, x_patch)
    # patch_size: Tensor representing the size of the subvolumes in the form Tensor(z_subsize, y_subsize, x_subsize)
    # orig_shape: the original shape of the input image (depth, height, width)
    
    # Batched data should not have the one-hot encoding     
    
    if lower == None:
        patch_size = list(patch_size)
        upper_size = list(upper_size)
        upper = torch.reshape(upper, upper_size)

        # get original size
        to_upper_shape = [a*b for a,b in zip(patch_size, upper_size)]
        upper = upper.cpu().detach().numpy()
        reconstructed_upper = unpatchify(upper, to_upper_shape)

        return reconstructed_upper

    else:
        patch_size = list(patch_size)
        upper_size = list(upper_size)
        lower_size = list(lower_size)
        
        # reshape batches (B, D, H, W) to voxelized (z_loc, y_loc, x_loc, z_patch, y_patch, x_patch)
#         upper = torch.cat(upper)
#         lower = torch.cat(lower)
        
        upper = torch.reshape(upper, upper_size)
        lower = torch.reshape(lower, lower_size)
        
        # get the original size
        to_upper_shape = [a*b for a,b in zip(patch_size, upper_size)]
        to_lower_shape = [a*b for a,b in zip(patch_size, lower_size)]
        
        # release the image from the device

        upper = upper.cpu().detach().numpy()
        lower = lower.cpu().detach().numpy()
        
        reconstructed_upper = unpatchify(upper, to_upper_shape)
        reconstructed_lower = unpatchify(lower, to_lower_shape)
        
        # Merge inferenced quotient and remainder arrays
        z_patch = patch_size[0]
        quot, remain = divmod(orig_shape[0],z_patch)
        shift_up = reconstructed_lower.shape[0]-remain # get the number of layers to shift up
        merged = np.zeros(orig_shape) # Make an array for predicted values
        merged[0:reconstructed_upper.shape[0],...] = reconstructed_upper # copy upper
        merged[(reconstructed_upper.shape[0]-shift_up):,...] = reconstructed_lower # shift the lower up and merge with upper
        
        return merged
    

def inference(dataloader, model, batch_size, patch_size, orig_shape, shuffle = False):
    # inference() performs a simple inference over the image using batches of tiles. This is similarly written to
    # training and validation. 
    # dataloader: generator for producing subvolume patches of the original image
    # model: pytorch model
    # batch_size: type int, should be 2^n where n is an integer
    
    model.eval()
    for upper_img, upper_shape, lower_img, lower_shape, _, _, _ in dataloader:
        # Empty list to place subvolumes in
        tmp_upper_dict = {}
        tmp_lower_dict = {}    

        upper_key_list = list(range(len(upper_img)))
        lower_key_list = list(range(len(lower_img)))


        # Only train on evenly split images
        if lower_img == None:
            print("Inferencing Upper Half of Image")
            num_subvolumes = len(upper_img)
            for bindex in trange(0, num_subvolumes, batch_size):
                if bindex + batch_size > num_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = upper_key_list[bindex:num_subvolumes]
                else:
                    batch_keys = upper_key_list[bindex:bindex+batch_size]
                
                sub_imgs = torch.stack([upper_img.get(key) for key in batch_keys], dim=0)

                output = model(sub_imgs) # predict the batches
                probabilities = torch.softmax(output, 1) 
                prediction = torch.argmax(probabilities,1)

                # update the upper img dictionary
                tmp_upper_dict.update(dict(zip(batch_keys,prediction)))

            # lower list does not exist
            tmp_lower_list = None

        # train on both 
        else:
            print("Inferencing Upper Half of Image")
            num_subvolumes = len(upper_img)
            for bindex in trange(0, num_subvolumes, batch_size):
                if bindex + batch_size > num_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = upper_key_list[bindex:num_subvolumes]
                else:
                    batch_keys = upper_key_list[bindex:bindex+batch_size]
                
                sub_imgs = torch.stack([upper_img.get(key) for key in batch_keys], dim=0)

                output = model(sub_imgs) # predict the batches
                probabilities = torch.softmax(output, 1) 
                prediction = torch.argmax(probabilities,1)

                # update the upper img dictionary
                tmp_upper_dict.update(dict(zip(batch_keys,prediction)))


            print("Inferencing Lower Half of Image")
            num_subvolumes = len(lower_img)
            for bindex in trange(0, num_subvolumes, batch_size):
                if bindex + batch_size > num_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = lower_key_list[bindex:num_subvolumes]
                else:
                    batch_keys = lower_key_list[bindex:bindex+batch_size]

                sub_imgs = torch.stack([lower_img.get(key) for key in batch_keys], dim=0)

                output = model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                prediction = torch.argmax(probabilities,1)

                # update the lower dictionary
                tmp_lower_dict.update(dict(zip(batch_keys,prediction)))

            # return tmp_upper_list, tmp_lower_list, running_loss / count

        # neuron reconstruction to calculate the dice metric.
        reconstructed_mask_order = (3,0,1,2)

        upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
        lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])


        reconstructed = reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                   lower_shape, patch_size, orig_shape) # returns (z,y,x)
        
        
    return reconstructed 


def inference_onnx(dataloader, model, batch_size, patch_size, orig_shape):
    # inference() performs a simple inference over the image using batches of tiles. This is similarly written to
    # training and validation. 
    # dataloader: generator for producing subvolume patches of the original image
    # model: ONNX file, i.e., "ResUNet.onnx"  
    # batch_size: type int, should be 2^n where n is an integer
    
    ort_session = onnxruntime.InferenceSession(model) # 
    upper_img, upper_shape, lower_img, lower_shape, _, _, _ = next(iter(dataloader))
        
    # Empty list to place subvolumes in
    tmp_upper_dict = {}
    tmp_lower_dict = {}    

    upper_key_list = list(range(len(upper_img)))
    lower_key_list = list(range(len(lower_img)))


    # Only train on evenly split images
    if lower_img == None:
        # num_subvolumes = len(upper_img)
        # for bindex in trange(0, num_subvolumes, batch_size):
        #     if bindex + batch_size > num_subvolumes:
        #         # if the bindex surpasses the number of number of sub volumes
        #         batch_keys = upper_key_list[bindex:num_sub_volumes]
        #     else:
        #         batch_keys = upper_key_list[bindex:bindex+batch_size]

        #     sub_imgs = torch.stack([upper_img.get(key) for key in batch_keys], dim=0)

        #     output = model(sub_imgs)
        #     probabilities = torch.softmax(output, 1)

        #     # discretize probability values 
        #     prediction = torch.argmax(probabilities, 1)
        #     tmp_upper_dict.update(dict(zip(batch_keys,sub_prediction)))

        #     # calculate the loss for the current batch, save the loss per epoch to calculate the average running loss
        #     current_loss = loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
        #     running_loss += current_loss.item()
        #     dice_count += 1

        # lower list does not exist
        tmp_lower_list = None

    # train on both 
    else:
        print("Inferencing Upper Half of Image")
        num_subvolumes = len(upper_img)
        for bindex in trange(0, num_subvolumes, batch_size):
            if bindex + batch_size > num_subvolumes:
                # if the bindex surpasses the number of number of sub volumes
                batch_keys = upper_key_list[bindex:num_subvolumes]
            else:
                batch_keys = upper_key_list[bindex:bindex+batch_size]
            
            sub_imgs = torch.stack([upper_img.get(key) for key in batch_keys], dim=0)
            inputs = {ort_session.get_inputs()[0].name: to_numpy(sub_imgs)}
            output = to_torch(ort_session.run(None,inputs)) # predict the batches
            probabilities = torch.softmax(output[0], 1) 
            prediction = torch.argmax(probabilities,1)

            # update the upper img dictionary
            tmp_upper_dict.update(dict(zip(batch_keys,prediction)))


        print("Inferencing Lower Half of Image")
        num_subvolumes = len(lower_img)
        for bindex in trange(0, num_subvolumes, batch_size):
            if bindex + batch_size > num_subvolumes:
                # if the bindex surpasses the number of number of sub volumes
                batch_keys = lower_key_list[bindex:num_subvolumes]
            else:
                batch_keys = lower_key_list[bindex:bindex+batch_size]

            sub_imgs = torch.stack([lower_img.get(key) for key in batch_keys], dim=0)
            inputs = {ort_session.get_inputs()[0].name: to_numpy(sub_imgs)}
            output = to_torch(ort_session.run(None,inputs)) # predict the batches
            probabilities = torch.softmax(output[0], 1)
            prediction = torch.argmax(probabilities,1)

            # update the lower dictionary
            tmp_lower_dict.update(dict(zip(batch_keys,prediction)))

        # return tmp_upper_list, tmp_lower_list, running_loss / count

    # neuron reconstruction to calculate the dice metric.
    reconstructed_mask_order = (3,0,1,2)

    upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
    lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])


    reconstructed = reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                lower_shape, patch_size, orig_shape) # returns (z,y,x)
    
    return reconstructed 


def get_index_nonempty_cubes(seg_dict):
    # find the index of non empty cubes from dictionary of segmentation masks
    # input: tensor(B,C,D,H,W)
    # output: list[non empty cube indices]
    get_keys = [key for (key,value) in seg_dict.items() if torch.count_nonzero(value[0,1:,...])!=0]
    return get_keys

def augmentation(transform, training_type, spatial_dim, img_dict, seg_dict, keys):
    aug_img = []
    aug_mask = []
    
    for key in keys:
        # changed convert torch.FloatTensor to torchio.ScalarImage and LabelMap for augmentation
        if spatial_dim == 3:

            img_of_interest = img_dict[key]
            mask_of_interest = seg_dict[key]

            if training_type == "sequential_image_processing":
                img_to_ScalarImage = torchio.ScalarImage(tensor = torch.squeeze(img_of_interest,dim=0).cpu())
                mask_to_LabelMap = torchio.LabelMap(tensor = torch.squeeze(mask_of_interest,dim=0).cpu())
            if training_type == "preprocessed_subpatch_image":
                img_to_ScalarImage = torchio.ScalarImage(tensor = img_of_interest.cpu())
                mask_to_LabelMap = torchio.LabelMap(tensor = mask_of_interest.cpu())

            # package ScalarImage and LabelMap into a Subject object for synchronized augmentation
            pseudo_subject = torchio.Subject(image = img_to_ScalarImage, seg = mask_to_LabelMap)
            pseudo_subject = transform(pseudo_subject)

            aug_img.append(pseudo_subject['image'].data)
            aug_mask.append(pseudo_subject['seg'].data)

        if spatial_dim == 2:
            
            # unsqueeze the dimensions from (1,512,512) -> (1,1,512,512) and for the mask (4,512,512) -> (1,4,512,512)
            img_of_interest = torch.unsqueeze(img_dict[key],dim=0)
            mask_of_interest = torch.unsqueeze(seg_dict[key],dim=0)

            # permute the dimensions from (a,b,c,d) -> (b,a,c,d) to match as z_direction
            img_of_interest = torch.permute(img_of_interest, dims=(1,0,2,3))
            mask_of_interest = torch.permute(mask_of_interest, dims=(1,0,2,3))

            img_to_ScalarImage = torchio.ScalarImage(tensor = img_of_interest.cpu())
            mask_to_LabelMap = torchio.LabelMap(tensor = mask_of_interest.cpu())

            # package ScalarImage and LabelMap into a Subject object for synchronized augmentation
            pseudo_subject = torchio.Subject(image = img_to_ScalarImage, seg = mask_to_LabelMap)
            pseudo_subject = transform(pseudo_subject)

            transformed_raw_img = torch.permute(pseudo_subject['image'].data, dims=(1,0,2,3))
            transformed_mask_img = torch.permute(pseudo_subject['seg'].data, dims=(1,0,2,3))

            transformed_raw_img = torch.squeeze(transformed_raw_img, dim = 0)
            transformed_mask_img = torch.squeeze(transformed_mask_img, dim = 0)

            # store the newly transformed data into a list for torch processing
            aug_img.append(transformed_raw_img)
            aug_mask.append(transformed_mask_img)

    return torch.stack(aug_img, dim = 0), torch.stack(aug_mask, dim = 0)

def get_class_weights(mask_list, classes = 3):
    # calculate balanced class weights
    # mask_list: list()
    # classes: integer num_class - 1

    process_mask = process_masks(classes)
    tot_mask_len = 0
    mask_bin_count = []
    le = preprocessing.LabelEncoder()
    for mask_name in mask_list:
        mask = tifffile.imread(mask_name).astype(np.float16)
        mask = process_mask(mask)
        mask_ind = le.fit_transform(mask.flatten())
        mask_bin_count.append(np.bincount(mask_ind).astype(np.float64))
        tot_mask_len += len(mask.flatten())

    mask_bin_count = np.sum(np.array(mask_bin_count),axis = 0)
    weights = tot_mask_len / (len(le.classes_) * mask_bin_count.astype(np.float64))
    return weights



# # Augmentation related functions
# import torchvision.transforms.functional as TF
# import random


# class augmentation_func:
#     def __init__(self,
#                  degree = 0,
#                  rand90 = False,
#                  vflip = False,
#                  hflip = False):
#         self.degree = 0
#         self.rand90 = rand90
#         self.vflip = vflip
#         self.hflip = hflip
    
#     def __call__(self, image, segmentation):
#         # random rotation around "degrees"
#         if self.degree > 0:
#             if random.random() > 0.5:
#                 degree_range = random.randint(-self.degree, self.degree) # get random angle
#                 image = TF.rotate(image, InterpolationMode = "lanczos", angle = degree_range)
#                 segmentation = TF.rotate(segmentation, angle = degree_range)

#         # random 90 degree rotation 
#         if self.rand90 == True:
#             if random.random() > 0.5:
#                 rand_int = random.randint(0,3)
#                 rotate_angle = rand_int*90 # rotate clockwise
#                 image = TF.rotate(image, angle = rotate_angle)
#                 segmentation = TF.rotate(segmentation, angle = rotate_angle)

#         # flip vertically?
#         if self.vflip == True:
#             if random.random() > 0.5:
#                 image = TF.vflip(image)
#                 segmentation = TF.vflip(segmentation)

#         # flip horizontally?
#         if self.hflip == True:
#             if random.random() > 0.5:
#                 image = TF.hflip(image)
#                 segmentation = TF.hflip(segmentation)

#         return torch.unsqueeze(image, dim = 0), torch.unsqueeze(segmentation, dim = 0)