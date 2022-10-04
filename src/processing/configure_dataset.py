from . import processing_functions as pf
import glob
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import ignite.distributed as idist

def split_data(config = None):
    raw_path = config['raw_path']
    mask_path = config['mask_path']

    split_size = config['DATASET']['split_size']

    raw_filename_list = glob.glob(raw_path) 
    mask_filename_list = glob.glob(mask_path)

    print(raw_filename_list)
    print(mask_filename_list)

    # Pre Shuffle
    raw_filename_list.sort()
    mask_filename_list.sort()

    # Shuffle the filename list
    
    raw_filename_list, mask_filename_list = shuffle(raw_filename_list, mask_filename_list, random_state = 42)
    raw_training_list, mask_training_list = raw_filename_list[:int(split_size*len(raw_filename_list))], mask_filename_list[:int(split_size*len(mask_filename_list))]
    raw_testing_list, mask_testing_list = raw_filename_list[int(split_size*len(raw_filename_list)):], mask_filename_list[int(split_size*len(mask_filename_list)):]

    return raw_training_list, mask_training_list, raw_testing_list, mask_testing_list

def load_dataset(device, config):
    #raw_path = config['raw_path']
    #mask_path = config['mask_path']

    lateral_steps = config['DATASET']['lateral_steps']
    axial_steps = config['DATASET']['axial_steps']
    patch_size = (axial_steps, lateral_steps, lateral_steps)
    dim_order = (0,4,1,2,3) # define the image and mask dimension order

    raw_training_list, mask_training_list, raw_testing_list, mask_testing_list = split_data(config=config)

    patch_transform = transforms.Compose([
    #                                       new_shape(new_xy = (600,960)),
                                        pf.MinMaxScalerVectorized(),
                                        pf.patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = False)])

    # define transforms for labeled masks
    label_transforms = transforms.Compose([
    #                                        new_shape(new_xy = (600,960)),
                                        pf.process_masks(int_class = 3),
                                        pf.patch_imgs(xy_step = lateral_steps, z_step = axial_steps, patch_size = patch_size, is_mask = True)])

    training_data = pf.MyImageDataset(raw_training_list,
                                mask_training_list,
                                transform = patch_transform,
                                label_transform = label_transforms,
                                device = device,
                                img_order = dim_order,
                                mask_order = dim_order,
                                num_classes = 4,
                                train=True)

    testing_data = pf.MyImageDataset(raw_testing_list,
                                mask_testing_list,
                                transform = patch_transform,
                                label_transform = label_transforms,
                                device = device,
                                img_order = dim_order,
                                mask_order = dim_order,
                                num_classes = 4,
                                train=False)

    training_dataloader = idist.auto_dataloader(training_data, batch_size = 1, num_workers=8, shuffle = False)
    testing_dataloader = idist.auto_dataloader(testing_data, batch_size = 1, num_workers=8, shuffle = False)

    return training_dataloader, testing_dataloader
