import torch
from processing import processing_functions as pf
from processing import configure_dataset as cd
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.handlers.ignite_metric import IgniteMetric

def metric_output_transform(output):
    y_pred, y = output["y_pred"], output["y"]
    return y_pred, y

# progress output transform
def loss_output_transform(output):
    loss = output["batch_loss"]
    return loss

def get_metric():
    dice_metric = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans = True)
    metric = IgniteMetric(dice_metric, output_transform= metric_output_transform)
    return metric

def get_loss(output_dir, config):
    config_loss = {"dice": DiceLoss, "dice_ce": DiceCELoss, "focal": FocalLoss, "dice_focal": DiceFocalLoss, "generalized_dice_focal": GeneralizedDiceFocalLoss, "generalized_dice": GeneralizedDiceLoss}
    weight_config = config['LOSS']['weights']
    loss_type = config['LOSS']['algo']
    if weight_config == "balanced":
        print("Calculating Focal Weights")
        _, mask_training_list, _, _ = cd.split_data()
        class_weights = torch.from_numpy(pf.get_class_weights(mask_training_list))
    elif weight_config == None:
        class_weights = None
    else:
        class_weights = weight_config

    with open(output_dir+'results.txt', 'w') as results_file:
        results_file.write("Using {}".format(loss_type))

    if loss_type == "dice_focal":
        #calculate weights based on relative frequency of pixels for each class
        loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_dice=1.0, lambda_focal=0.2)
        
    elif loss_type == "generalized_dice_focal":
        #calculate weights based on relative frequency of pixels for each class
        loss_function = config_loss[loss_type](focal_weight = class_weights, lambda_gdl=1.0, lambda_focal=0.2)
    elif loss_type == "focal":
        loss_function = config_loss[loss_type](weight = class_weights)
    else:
        loss_function = config_loss[loss_type]()
    
    return loss_function
