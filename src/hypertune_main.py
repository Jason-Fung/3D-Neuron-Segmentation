#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import utility libraries
import os
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz
import random
import yaml
import sys
import copy
import argparse
from functools import partial
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
sys.path.insert(1, '../')

# custom made libraries
from processing import processing_functions as pf
from processing import configure_dataset as cd
from train_engine import config_engines, config_augmentation, config_loss_metric

# import deep learning libraries
from torchvision import transforms, utils
import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from ignite.engine import Engine, Events
from ignite.metrics import Metric, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
import ignite.distributed as idist

# hyperparameter tuning
from ray import tune
from ray.tune.stopper import TrialPlateauStopper
from ray.tune import Tuner
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB

# from monai.losses import DiceLoss

from monai.networks.nets import BasicUNet, UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import skimage

# In[]
def train_model(config, data_cfg, results_dir, device,max_epochs):
    training_dataloader, testing_dataloader = cd.load_dataset(device=device, config=data_cfg)

    # # Define Model and Parameters

    # decay = 1e-5
    input_chnl = 1
    output_chnl = 4

    # model = config_model.get_model(config)

    model = UNet(spatial_dims=3, 
                in_channels = input_chnl,
                out_channels = output_chnl,
                channels = (16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm = 'batch',
                dropout = config['dropout'])

    model = idist.auto_model(model)

    # ## Loss, Metric, Schedulers

    loss_function = config_loss_metric.get_loss(results_dir, data_cfg)

    # ## Define Training and Validation Functions

    aug_transform = config_augmentation.augmentation(cfg = data_cfg)

    engine_config = config_engines.configure_engines(model=model,
                                                     loss=loss_function,
                                                     aug_config = aug_transform.transforms(),
                                                     device=device,
                                                     hyper_cfg = config,
                                                     data_cfg = data_cfg)

    # create the trainer and evaluator engines
    trainer = Engine(engine_config.train)
    evaluator = Engine(engine_config.validate)

    metric = config_loss_metric.get_metric()
    RunningAverage(output_transform=config_loss_metric.loss_output_transform).attach(evaluator, "batch_loss")
    # pbar = ProgressBar(persist=True)

    # Attach both metric to trainer and evaluator engine
    metric.attach(trainer,"Dice")
    metric.attach(evaluator,"Dice")

    @trainer.on(Events.EPOCH_COMPLETED)
    def report_metrics(trainer):
        print("Epoch Completed")
        evaluator.run(testing_dataloader)
        testing_metrics = copy.deepcopy(evaluator.state.metrics)
        batch_loss = testing_metrics["batch_loss"]
        val_dice = testing_metrics["Dice"]

        engine_config.scheduler.step()

        checkpoint_data = {"model":engine_config.model.state_dict(), "optimizer":engine_config.optimizer.state_dict()}

        session.report(metrics={"loss": batch_loss, 
                                "Soma Dice": val_dice[0], 
                                "Dendrite Dice": val_dice[1], 
                                "Filopodia Dice": val_dice[2], 
                                "mean_dice":  torch.mean(val_dice)}, 
                       checkpoint=Checkpoint.from_dict(checkpoint_data))

    # ## Setup Model and Log Saving Directories
    print("running training logic")
    trainer.run(training_dataloader, max_epochs = max_epochs)
    print("finished training logic")

# ## Early Stopping

# In[]:

def main(exp_config=None, checkpoint_dir=None, results_dir=None, device = "cpu"):
    # set up hyper parameters
    max_epochs = exp_config['TRAINING']['max_epochs']

    hyper_config = {
        'lr': tune.loguniform(5e-5,1e-3), # optimizer
        'l2': tune.loguniform(5e-5,1e-2), # optimizer
        'dropout': tune.uniform(0.1,0.4) # network
        # 'OPTIMIZER': tune.choice(['SGD','AdamW']), # optimizer
        #'SCHEDULER': tune.choice(['LR_expo', 'LR_Plateau', 'LR_Cosine']),
    }
    with open(results_dir+'hyper_params.yml', 'w') as outfile:
        yaml.dump(hyper_config, outfile, default_flow_style=False)
    
    # configure tuning algorithm
    if exp_config['TUNING_ALGO'] == 'HyperOpt':
        tune_algo = HyperOptSearch(metric="loss", mode="min",random_state_seed = 42)
    if exp_config['TUNING_ALGO'] == 'BayesOpt':
        tune_algo = BayesOptSearch(metric="loss", mode="min", random_state = 42)
    if exp_config['TUNING_ALGO'] == 'BOHBOpt':
        tune_algo = TuneBOHB(metric="loss", mode="min", seed = 42)

    # configure trial scheduler
    if exp_config['TRIAL_SCHEDULER'] == 'ASHA':
        with open(results_dir+"results.txt", "w") as results_file:
            results_file.write("Using ASHA for Scheduling \n")
        scheduler = ASHAScheduler(
            max_t=max_epochs,
            grace_period=5,
            reduction_factor=2)
    if exp_config['TRIAL_SCHEDULER'] == "BOHB": # must have BOHBOpt with it
        with open(results_dir+"results.txt", "w") as results_file:
            results_file.write("Using HyperBandForBOHB for Scheduling \n")
        scheduler = HyperBandForBOHB(
            max_t=max_epochs,
            reduction_factor = 2)

    reporter = CLIReporter(
        metric_columns=["loss","Soma Dice", "Dendrite Dice", "Filopodia Dice", "mean_dice","training_iteration"],
        parameter_columns=["lr","l2","dropout"],
        max_progress_rows=20,
        )

    stopper = TrialPlateauStopper(metric = "loss",
                                  std = 0.00001,
                                  grace_period = 3)

    tuner = Tuner(
        tune.with_resources(trainable = partial(train_model, data_cfg=exp_config, results_dir = results_dir, device = device, max_epochs = max_epochs), 
                            resources = {"cpu": 8, "gpu" : 1}), # use 1 gpu and 8 cpus for 1 trial
        tune_config = tune.TuneConfig(search_alg=tune_algo,
                                      metric = "Dendrite Dice",
                                      mode = "max",
                                      scheduler=scheduler,
                                      num_samples = exp_config['TRAINING']['num_trials']),
        param_space = hyper_config,
        run_config = RunConfig(name= exp_config["MODEL"]["model_name"] +"_"+exp_config['TUNING_ALGO'],
                               progress_reporter=reporter,
                               verbose = 1,
                               stop = stopper)
    )
    results = tuner.fit()

    with open(results_dir + "results.txt", "w") as results_file:
        results_file.write(str(results.get_best_result(metric="loss", mode="min").config)+"\n")

    #best_trial = results.get_best_result("loss", "min", "last")
    #print("Best trial config: {}".format(best_trial.config))
    #print("Best trial final validation loss: {}".format(
    #    best_trial["loss"]))
    #print("Best trial final mean validation Dice: {}".format(
    #    best_trial["val_mean_dice"]))



# In[ ]:
# Running Training Engine

if __name__ == "__main__":
    # yaml experiment file location
    project_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + "/../")
    exp_path = "/config/convnets/ResUNet/"

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str)
    args = vars(parser.parse_args())
    #exp_file = "exp_template.yml"
    exp_file = args['config']

    with open(project_path + exp_path + exp_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model_name = config['MODEL']['model_name']
    net_type = config['MODEL']['net_type']
    max_epochs = config['TRAINING']['max_epochs']

    # set up logs and results directory 
    model_directory =  project_path + f"/results/{model_name}/"
    date = datetime.now(tz=pytz.utc).strftime('%Y%m%d')
    time = datetime.now(tz=pytz.utc).strftime('%H%M%S')
    date_directory = f"{date}/"
    time_directory = f"{date}_{time}/"

    results_dir = model_directory + date_directory + time_directory

    log_directory = results_dir + "log"
    os.makedirs(log_directory)

    output_txt = open(results_dir +'results.txt', 'w')

    exp_save_path = open(results_dir +'experiment.yml', 'w')

    with open(results_dir + 'experiment.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # # create writer to log results into tensorboard
    # log_writer = SummaryWriter(log_directory)

    # set up device environment
    use_cuda = torch.cuda.is_available()
    device = idist.device()
    if use_cuda == True:
        print("Using Cuda")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")
    
    main(exp_config=config, 
         results_dir=results_dir, 
         device = device)

# @trainer.on(Events.STARTED)
# def print_start(trainer):
#     print("Training Started")

# @trainer.on(Events.EPOCH_STARTED)
# def print_epoch(trainer):
#     print("Epoch : {}".format(trainer.state.epoch))
    
# @trainer.on(Events.EPOCH_COMPLETED)
# def save_model(trainer):
#     global best_dice
#     global best_epoch
#     global best_epoch_file
#     global best_loss
    
#     epoch = trainer.state.epoch
#     def get_saved_model_path(epoch):
#         return model_directory + date_directory + time_directory + f"{model_name}_{epoch}.pth"

#     # initialize global values
#     best_dice = -torch.inf if epoch == 1 else best_dice
#     best_loss = torch.inf if epoch == 1 else best_loss
#     best_epoch = 1 if epoch == 1 else best_epoch
#     best_epoch_file = '' if epoch == 1 else best_epoch_file
    
#     def log_training_results(trainer):
#         evaluator.run(training_dataloader)
#         # Get engine metrics and losses
#         training_metrics = copy.deepcopy(evaluator.state.metrics)
#         # pbar.log_message(
#         #     "Training Results - Epoch: {} \nMetrics\n{}"
#         #     .format(trainer.state.epoch, pprint.pformat(training_metrics)))
#         with output_txt as results_file:
#             results_file.write("Training Results - Epoch: {} \nMetrics\n{}"
#                 .format(trainer.state.epoch, pprint.pformat(training_metrics)))
#         return training_metrics
    
#     def log_testing_results(trainer):
#         evaluator.run(testing_dataloader)
#         testing_metrics = copy.deepcopy(evaluator.state.metrics)
#         # scheduler.step(testing_metrics["batch_loss"])
#         scheduler.step()
#         # pbar.log_message(
#         #     "Validation Results - Epoch: {} \nMetrics\n{}"
#         #     .format(trainer.state.epoch, pprint.pformat(testing_metrics)))
#         with output_txt as results_file:
#             results_file.write("Validation Results - Epoch: {} \nMetrics\n{}"
#                 .format(trainer.state.epoch, pprint.pformat(testing_metrics)))
#         return testing_metrics
    
#     training_metrics= log_training_results(trainer)
#     testing_metrics= log_testing_results(trainer)
    
#     train_dice = training_metrics['Dice']
#     val_dice = testing_metrics['Dice']

#     train_mean_dice = torch.mean(train_dice)
#     val_mean_dice = torch.mean(val_dice)
#     train_loss = training_metrics['batch_loss']
#     val_loss = testing_metrics['batch_loss']
    

#     # log results
#     log_writer.add_scalars('Training vs. Validation Loss',
#                     {'Training' : train_loss, 'Validation' : val_loss}, epoch)
#     log_writer.add_scalars('Training vs. Validation Mean Dice ',
#                     {'Training Mean Dice' : train_mean_dice, 'Validation Mean Dice' : val_mean_dice}, epoch)
#     log_writer.add_scalars('Training vs. Validation Soma Dice ',
#                     {'Training Soma Dice' : train_dice[0], 'Validation Soma Dice' : val_dice[0]}, epoch)
#     log_writer.add_scalars('Training vs. Validation Dendrite Dice ',
#                     {'Training Dendrite Dice' : train_dice[1], 'Validation Dendrite Dice' : val_dice[1]}, epoch)
#     log_writer.add_scalars('Training vs. Validation Filopodias Dice ',
#                     {'Training Filopodias Dice' : train_dice[2], 'Validation Filopodias Dice' : val_dice[2]}, epoch)
#     log_writer.flush()

#     if (testing_metrics['batch_loss'] < best_loss):
        
#         # if there was a previous model saved, delete that one
#         prev_best_epoch_file = get_saved_model_path(best_epoch)
#         if os.path.exists(prev_best_epoch_file):
#             os.remove(prev_best_epoch_file)

#         # update the best mean dice and loss and save the new model state
# #         best_dice = val_mean_dice
#         best_loss = testing_metrics['batch_loss']
#         best_epoch = epoch
#         best_epoch_file = get_saved_model_path(best_epoch)
# #         print(f'\nEpoch: {best_epoch} - New best Dice and Loss! Mean Dice: {best_dice} Loss: {best_loss}\n\n\n')
#         # print(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
#         with output_txt as results_file:
#             results_file.write(f'\nEpoch: {best_epoch} - New best Loss! Loss: {best_loss}\n\n\n')
#         torch.save(model.state_dict(), best_epoch_file)

# %%
