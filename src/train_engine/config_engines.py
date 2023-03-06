import torch
import random
from tqdm import trange
from processing import processing_functions as pf
import ignite.distributed as idist

class configure_engines:
    def __init__(self, 
                 model = None,
                 loss = None, 
                 aug_config = None, 
                 device = "cpu", 
                 hyper_cfg = None,
                 data_cfg = None) -> None:

        # hyperparameters
        self.hyper_cfg = hyper_cfg

        # training parameters
        self.batch_size = data_cfg['DATASET']['batch_size']

        # devices
        self.device = device

        # reconstruction parameters
        self.dim_order = (0,4,1,2,3)
        self.lateral_steps = data_cfg['DATASET']['lateral_steps']
        self.axial_steps = data_cfg['DATASET']['axial_steps']
        self.patch_size = (self.axial_steps, self.lateral_steps, self.lateral_steps)

        # self.model and self.optimizer
        self.model = model
        self.optimizer = get_optimizer(model.parameters(), config=hyper_cfg) # set the class
        self.scheduler = get_lr_scheduler(optimizer = self.optimizer, config=data_cfg)
        self.curr_epoch = 0
        self.prev_epoch = 0
        self.loss_function = loss

        # augmentation
        self.augment = data_cfg['DATASET']['AUGMENTATION']['augment']
        if self.augment == True:
            self.aug_transform = aug_config
        self.shuffle = data_cfg['TRAINING']['shuffle']


    def train_sequential_image_based(self, engine, batch):
        # def step_scheduler():
        #     self.scheduler.step()
        # set up augmentation
        if self.curr_epoch > self.prev_epoch:
            self.scheduler.step() # reduce the learning rate
            self.prev_epoch = self.curr_epoch # replace prev epoch with new epoch
            self.curr_epoch = engine.state.epoch

        self.model.train()
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
        
        if self.shuffle == True:
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
            num_subvolumes = len(upper_img)
            for bindex in trange(0, num_subvolumes, self.batch_size):
                if bindex + self.batch_size > num_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = upper_key_list[bindex:num_subvolumes]
                else:
                    batch_keys = upper_key_list[bindex:bindex+self.batch_size]
                
                sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0)
                sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                
                self.optimizer.zero_grad()
                output = self.model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                
                # discretize probability values 
                prediction = torch.argmax(probabilities, 1)
                tmp_upper_dict.update(dict(zip(batch_keys,prediction)))
                
                # calculate the loss for the current batch, save the loss per epoch to calculate the average running loss
                current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                current_loss.backward()
                self.optimizer.step()
                running_loss += current_loss.item()
            
            # lower list does not exist
            tmp_lower_list = None
                
        # train on both 
        else:
            num_upper_subvolumes = len(upper_img)
            if self.augment:
                # Extract index of non-zero subvolumes
                upper_indexes = pf.get_index_nonempty_cubes(upper_mask)
                
                # Augment on non-zero subvolumes based on their location in the volume (by index)
                for bindex in range(0, len(upper_indexes), self.batch_size):
                    # for augmentation
                    if bindex + self.batch_size > len(upper_indexes):
                        upper_batch = upper_indexes[bindex:len(upper_indexes)]
                    else:
                        upper_batch = upper_indexes[bindex:bindex+self.batch_size]
                        
                    sub_imgs, sub_masks = pf.augmentation(self.aug_transform, upper_img, upper_mask, upper_batch)
                    sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(sub_imgs)
                    probabilities = torch.softmax(output, 1)
                    prediction = torch.argmax(probabilities, 1)
                    #print("Is probabilities using cuda?", probabilities.is_cuda)
                    #print("Is masks using cuda?", sub_masks.is_cuda)
                    
                    current_loss = self.loss_function(probabilities, sub_masks)
                    current_loss.backward()
                    self.optimizer.step()
                    running_loss += current_loss.item()
                    count_loss += 1
            
            for bindex in range(0, num_upper_subvolumes, self.batch_size):
                if bindex + self.batch_size > num_upper_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = upper_key_list[bindex:num_upper_subvolumes]
                else:
                    batch_keys = upper_key_list[bindex:bindex+self.batch_size]
                
                sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(sub_imgs) # predict the batches
                probabilities = torch.softmax(output, 1) 
                prediction = torch.argmax(probabilities,1)
                
                # update the upper img dictionary
                tmp_upper_dict.update(dict(zip(batch_keys,prediction)))
                
                current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                
                current_loss.backward()
                self.optimizer.step()
                running_loss += current_loss.item()
                count_loss += 1
            
            num_lower_subvolumes = len(lower_img)
            if self.augment:
                lower_indexes = pf.get_index_nonempty_cubes(lower_mask)
                
                for bindex in range(0, len(lower_indexes), self.batch_size):
                    # for augmentation
                    if bindex + self.batch_size > len(lower_indexes):
                        lower_batch = lower_indexes[bindex:len(lower_indexes)]
                    else:
                        lower_batch = lower_indexes[bindex:bindex+self.batch_size]
                        
                    sub_imgs, sub_masks = pf.augmentation(self.aug_transform, lower_img, lower_mask, lower_batch)
                    sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(sub_imgs)
                    probabilities = torch.softmax(output, 1)
                    prediction = torch.argmax(probabilities, 1)

                    current_loss = self.loss_function(probabilities, sub_masks)
                    current_loss.backward()
                    self.optimizer.step()
                    running_loss += current_loss.item()
                    count_loss += 1
                    
            for bindex in range(0, num_lower_subvolumes, self.batch_size):
                if bindex + self.batch_size > num_lower_subvolumes:
                    # if the bindex surpasses the number of number of sub volumes
                    batch_keys = lower_key_list[bindex:num_lower_subvolumes]
                else:
                    batch_keys = lower_key_list[bindex:bindex+self.batch_size]
                
                sub_imgs = torch.squeeze(torch.stack([lower_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                sub_masks = torch.squeeze(torch.stack([lower_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(sub_imgs)
                probabilities = torch.softmax(output, 1)
                prediction = torch.argmax(probabilities,1)
                
                # update the lower dictionary
                tmp_lower_dict.update(dict(zip(batch_keys,prediction)))

                current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                current_loss.backward()
                self.optimizer.step()
                running_loss += current_loss.item()
                count_loss += 1

            orig_shape = full_mask.shape[1:-1]
            reconstructed_mask_order = (3,0,1,2)


            upper_values = torch.stack([tmp_upper_dict[key] for key in list(range(len(tmp_upper_dict)))])
            lower_values = torch.stack([tmp_lower_dict[key] for key in list(range(len(tmp_lower_dict)))])


            reconstructed = pf.reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                    lower_shape, self.patch_size, orig_shape) # returns (z,y,x)
            reconstructed = pf.to_categorical_torch(reconstructed, num_classes = 4) # returns (z,y,x,c)
            reconstructed = reconstructed.type(torch.int16)
            reconstructed = torch.permute(reconstructed, reconstructed_mask_order)
            reconstructed = torch.unsqueeze(reconstructed, 0) # make reconstructed image into (Batch,c,z,y,x)
    #         full_mask = full_mask.type(torch.int16)
            gt_mask = torch.permute(full_mask, self.dim_order) # roll axis of grount truth mask into (batch,c,z,y,x)
        return {"batch_loss":running_loss/count_loss, "y_pred":reconstructed, "y":gt_mask}

        
    def validate_sequential_image_based(self, engine, batch):
        self.model.eval()
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
                for bindex in range(0, num_subvolumes, self.batch_size):
                    if bindex + self.batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = upper_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = upper_key_list[bindex:bindex+self.batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0)
                    sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                    
                    self.optimizer.zero_grad()
                    output = self.model(sub_imgs)
                    probabilities = torch.softmax(output, 1)

                    # discretize probability values 
                    prediction = torch.argmax(probabilities, 1)
                    tmp_upper_dict.update(dict(zip(batch_keys,prediction)))

                    # calculate the loss for the current batch, save the loss per epoch to calculate the average running loss
                    current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                    running_loss += current_loss.item()
                    count_loss += 1

                # lower list does not exist
                tmp_lower_list = None

            # train on both 
            else:
                num_subvolumes = len(upper_img)
                for bindex in range(0, num_subvolumes, self.batch_size):
                    if bindex + self.batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = upper_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = upper_key_list[bindex:bindex+self.batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([upper_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                    sub_masks = torch.squeeze(torch.stack([upper_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                    sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)


                    self.optimizer.zero_grad()
                    output = self.model(sub_imgs) # predict the batches
                    probabilities = torch.softmax(output, 1) 
                    prediction = torch.argmax(probabilities,1)

                    current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
                    running_loss += current_loss.item()
                    count_loss += 1

                    # update the upper img dictionary
                    tmp_upper_dict.update(dict(zip(batch_keys,prediction)))

                num_subvolumes = len(lower_img)
                for bindex in range(0, num_subvolumes, self.batch_size):
                    if bindex + self.batch_size > num_subvolumes:
                        # if the bindex surpasses the number of number of sub volumes
                        batch_keys = lower_key_list[bindex:num_subvolumes]
                    else:
                        batch_keys = lower_key_list[bindex:bindex+self.batch_size]
                    
                    sub_imgs = torch.squeeze(torch.stack([lower_img.get(key) for key in batch_keys], dim=1), dim = 0) 
                    sub_masks = torch.squeeze(torch.stack([lower_mask.get(key) for key in batch_keys], dim=1), dim = 0)
                    sub_imgs, sub_masks = sub_imgs.to(self.device), sub_masks.to(self.device)

                    output = self.model(sub_imgs)
                    probabilities = torch.softmax(output, 1)
                    prediction = torch.argmax(probabilities,1)
                    current_loss = self.loss_function(probabilities, sub_masks) # + dice_loss(predictions, patch_gt)
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


            reconstructed = pf.reconstruct_training_masks(upper_values, lower_values, upper_shape, 
                                                        lower_shape, self.patch_size, orig_shape) # returns (z,y,x)
            reconstructed = pf.to_categorical_torch(reconstructed, num_classes = 4) # returns (z,y,x,c)
            reconstructed = torch.permute(reconstructed, reconstructed_mask_order)
            reconstructed = torch.unsqueeze(reconstructed, 0) # make reconstructed image into (Batch,c,z,y,x)

    #         full_mask = full_mask.type(torch.int16).cpu()
            gt_mask = torch.permute(full_mask, self.dim_order) # roll axis of grount truth mask into (batch,c,z,y,x)
            
        return {"batch_loss":running_loss/count_loss, "y_pred":reconstructed, "y":gt_mask}


def get_optimizer(model_params, config):
    if 'OPTIMIZER' in config:
        choice = config['OPTIMIZER']
    else:
        choice = "AdamW"

    algo = {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}

    learning_rate = config['lr']
    l_2_weight = config['l2']
    if choice == "SGD":
        optimizer = idist.auto_optim(algo[choice](model_params, lr=learning_rate, weight_decay = l_2_weight, momentum = 0.9))
    else:
        optimizer = idist.auto_optim(algo[choice](model_params, lr=learning_rate, weight_decay = l_2_weight))

    return optimizer

def get_lr_scheduler(optimizer, config=None):
    # configures the type of learning rate scheduler. Currently
    # supporting only Exponential Decay ("LR_Expo"), Reduce on plateau ("LR_Plateau"), and Cosine Annealing ("LR_Cosine")
    if config == None:
        choice = "LR_Cosine"
    else:
        choice = config['SCHEDULER']['algo']
    algo = {"LR_Expo": torch.optim.lr_scheduler.ExponentialLR, 
            "LR_Plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "LR_Cosine": torch.optim.lr_scheduler.CosineAnnealingLR}

    print(f"using LR scheduler: {choice}")
    if choice == "LR_Cosine":
        max_epochs = config['TRAINING']['max_epochs']
        scheduler = algo[choice](optimizer, max_epochs/25, verbose = True)
    elif choice == "LR_Plateau":
        scheduler = algo[choice](optimizer, 
                                 'min',
                                 factor=0.5, 
                                 patience = 10, 
                                 threshold=1e-5, 
                                 threshold_mode= 'abs', 
                                 verbose=True)
    else: # other wise use a exponential 
        scheduler = algo[choice](optimizer, gamma=0.9)
    return scheduler
    
