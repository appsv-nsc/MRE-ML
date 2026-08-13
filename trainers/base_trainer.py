import sys
import os
import random
import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import pandas as pd
from monai.utils import set_determinism

from datasets import get_dataloder, dataloaders_dict
from networks import get_networks, networks_dict, freeze_by_keywords, unfreeze_by_keywords, rewarm_lr_groups, reset_all_lrs
from utils.losses import BCE_Dice_Loss, SoftDiceLoss, DiceLoss, MultiDiceLoss, BCEDiceLoss, TverskyLoss,TargetMaskWeightedMSELoss
from utils.dice_loss import DC_and_CE_loss
from utils.recorder import Recorder
# from nnunet.training.loss_functions.dice_loss import *

from utils.metrics import SegMetric_Numpy, SegMetric_Tensor
from utils.tools import AverageMeter, select_target_type

"""
BaseTrainer is for all the pretext tasks and downstream tasks.

"""


class BaseTrainer(object):
    def __init__(
            self,
            config,pipeline_task_monitor=None):
        """
       Steps:
           1、Init logger.
           2、Init device.
           3、Init seed.
           4、Init data_loader.
           6、Init model. For some complex pipelines, build the model or learner from network,
           the output of which is the desired variables in loss function.
               For the simple methods, network = model.
           7、Mount the model onto the device (gpus) and set nn.parallel if necessary.
           8、Check resume.
           9、Load the pre-trained model if the training_phase is 'fine-tuning'.
           10、Init loss criterion./ or re-define in the specific trainer.
           11、Init optimizer and scheduler.
           12、Load optimizer and scheduler from resume if necessary.

       After this call,
           All will be prepared for training.
       """

        self.config = config
        self.recorder = Recorder(config)
        if self.config.run_mode == 'full_pipeline':
            self.part_of_pipeline = True
            self.pipeline_monitor = pipeline_task_monitor
            self.pipeline_monitor.set_log_dir(self.recorder.save_dir)
        else:
            self.part_of_pipeline = False
        self.get_device()
        self.init_random_and_cudnn()
        self.init_dataloader()

        if self.config.model == 'Simple':
            self.init_model()
            self.model_to_gpu()  # must before initializing the optimizer
            self.get_training_phase()
            self.check_resume_and_pretrained_weights()
            self.set_trainable_params()
            self.init_loss_criterion()
            self.init_optimizer_and_scheduler()  # better after loading the pretrained weights since sometimes we might freeze some params.
            if self.config.resume is not None:
                self.load_optimizer_state_dict()  # continue training
        else:
            print('Model and training phase initialization is required in the specific Trainer for complex models')

        self.training = True
        sys.stdout.flush()

    def get_lr(self) -> int:
        return self.optimizer.param_groups[0]['lr']

    def get_device(self):
        self.use_cuda = torch.cuda.is_available() and len(self.config.gpu_ids) > 0
        self.device = torch.device('cuda:%d' % self.config.gpu_ids[0]
                                   if torch.cuda.is_available() and len(self.config.gpu_ids) > 0 else 'cpu')

    def init_random_and_cudnn(self):
        # Set seed
        if self.config.manualseed is None:
            self.config.manualseed = random.randint(1, 10000)
        np.random.seed(self.config.manualseed)
        random.seed(self.config.manualseed)
        torch.manual_seed(self.config.manualseed)
        set_determinism(self.config.manualseed)

        if self.use_cuda:
            torch.cuda.manual_seed(self.config.manualseed)
            torch.cuda.manual_seed_all(self.config.manualseed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = self.config.benchmark

    def init_dataloader(self):
        self.train_dataset, self.train_dataloader = get_dataloder(self.config, flag="train", drop_last=True)
        self.eval_dataset, self.eval_dataloader = get_dataloder(self.config, flag="valid", drop_last=False)

    def init_model(self):
        self.network = get_networks(self.config)
        self.model = self.network
        self.model.cuda()

    def init_loss_criterion(self):
        if self.config.loss == 'bce':
            self.criterion = nn.BCELoss(reduction='mean')
        elif self.config.loss == 'softdice':
            self.criterion = SoftDiceLoss()
        elif self.config.loss == 'bcelog':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.config.loss == "dice":
            self.criterion = DiceLoss()
        elif self.config.loss == 'bcedice':
            self.criterion = BCEDiceLoss(dice_weight=self.config.dice_weight)
        elif self.config.loss == 'multidice':
            self.criterion = MultiDiceLoss(weights=self.config.dice_weight)

        elif self.config.loss == 'nnunet_ce_dice':
            self.criterion = DC_and_CE_loss({}, {})

        elif self.config.loss == 'nnunet_ce_dice2':
            self.criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

        elif self.config.loss == 'bce_dice':
            self.criterion = BCE_Dice_Loss(bce_weight=self.config.bce_weight, dice_weight=self.config.dice_weight)

        elif self.config.loss == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        
        elif self.config.loss == 'huber':
            self.criterion = nn.SmoothL1Loss()

        elif self.config.loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        elif self.config.loss == 'target_mask_weighted_mse':
            self.criterion = TargetMaskWeightedMSELoss(mul_factor=self.config.loss_mul_factor)

        else:
            raise NotImplementedError("The loss function has not been defined.")

        self.criterion.to(self.device)

    def init_optimizer_and_scheduler(self):

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        # init optimizer
        if self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.config.lr, momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay, nesterov=False)

        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.config.lr, weight_decay=1e-6)

        else:
            raise NotImplementedError("The optimizer has not been defined.")

        # init scheduler
        if self.config.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config.patience * 0.8),
                                                             gamma=0.5)

        elif self.config.scheduler == 'StepLR_multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.learning_rate_decay)

        elif self.config.scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.epochs)

        elif self.config.scheduler == 'ReduceLROnPlateau':
            if any(task in self.config.task for task in ['classification','segmentation', 'jigsaw', 'rkb']):
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1,
                                                                            patience=20,
                                                                            verbose=True)
            else:
                if self.fine_tuning_scheme =='warmup_by_layer_by_patience':
                    module_dict = self.model.module.get_module_dicts()  # your function
                    all_keywords = module_dict["encoder"] + module_dict["decoder"] + module_dict["out"]

                    #param_groups = self.build_param_groups_by_keywords(all_keywords, weight_decay=1e-6)
                    if self.config.optimizer == 'adam':
                        param_groups = self.build_param_groups_by_keywords(all_keywords, weight_decay=1e-6)
                        self.optimizer = torch.optim.Adam(param_groups, weight_decay=1e-6)
                    elif self.config.optimizer == 'sgd':
                        param_groups = self.build_param_groups_by_keywords(all_keywords, weight_decay=self.config.weight_decay)
                        self.optimizer = torch.optim.SGD(
                            param_groups,
                            momentum=self.config.momentum, weight_decay=self.config.weight_decay,
                            nesterov=False
                        )
                    else:
                        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                                                            patience=10,
                                                                            verbose=True)

        else:
            self.scheduler = None

    def get_training_phase(self):

        if self.config.pretrained_model is not None:
            self.training_phase = 'fine_tuning'
            # module_dict: the layer name list of encoder, decoder and out transition.
            module_dict = self.model.module.get_module_dicts()
            if 'encoder' in self.config.transferred_part:
                self.encoder_layers_reversed = module_dict['encoder']
                self.encoder_layers_reversed.reverse()
                if self.config.transferred_part=='encoder':
                    self.transferred_dict = module_dict['encoder']
                else:
                    self.transferred_dict = module_dict['encoder'] + module_dict['decoder']
            elif self.config.transferred_part == 'decoder':
                self.transferred_dict = module_dict['decoder']
            #else:
                #self.transferred_dict = module_dict['encoder'] + module_dict['decoder']
        else:
            self.training_phase = 'from_scratch'

        self.recorder.logger.info('Training phase : {}'.format(self.training_phase))

    def model_to_gpu(self):
        self.recorder.logger.info('use: %d gpus', torch.cuda.device_count())
        self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)

    def check_resume_and_pretrained_weights(self):
        if self.config.resume is not None:
            # continue training -- load start_epoch and model
            self.load_model_state_dict(self.config.resume)
        else:
            if self.training_phase == 'fine_tuning':
                # load pretrained weights
                self.load_pretrained_weights()
            self.start_epoch = 0

    def set_trainable_params(self):
        # default fine-tuning scheme == "full"
        self.fine_tuning_scheme = 'full'
        if self.training_phase == 'fine_tuning' and hasattr(self.config, "fine_tuning_scheme"):
            if 'warmup' in self.config.fine_tuning_scheme:
                # freeze pretrained encoder for several epochs
                freeze_by_keywords(self.model, keywords=['enc'])
                freeze_by_keywords(self.model, keywords=['bottleneck'])
                self.full_freeze_epochs = self.config.full_freeze_epochs
                self.fine_tuning_scheme = self.config.fine_tuning_scheme
                self.warmup_epochs_steps = self.config.warmup_epochs_steps
                self.warmup_patience = self.config.warmup_patience

            elif self.config.fine_tuning_scheme == 'fixed':
                # freeze pretrained encoder for all the epochs
                freeze_by_keywords(self.model, keywords=['enc'])
                freeze_by_keywords(self.model, keywords=['bottleneck'])
                self.full_freeze_epochs = self.config.epochs
                self.fine_tuning_scheme = 'fixed'

            elif self.config.fine_tuning_scheme == 'full':
                pass

            else:
                raise NotImplementedError("the fine-tuning scheme is not settled!")

    def set_input(self, sample):
        input, target,_,_  = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        #self.image_index = image_index

    def forward(self):
        self.pred = self.model(self.input)

    def backward(self):
        if self.config.loss == 'target_mask_weighted_mse':
            self.loss = self.criterion(self.pred, self.target, self.seg_mask)
        else:
           self.loss = self.criterion(self.pred, self.target) 
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_inference(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        pred = self.network(input)
        loss = self.criterion(pred, target)
        return pred, loss

    
    def _get_params_by_names(self, names):
        """Collect parameters from attribute names on the model."""
        params = []
        for name in names:
            module = getattr(self.model, name, None)
            if module is None:
                raise AttributeError(f"Model has no submodule named '{name}'")
            params += list(module.parameters())
        return params

    def build_param_groups_by_keywords(self,all_keywords, weight_decay=1e-6):
        """
        Groups parameters by the *first* matching keyword in their name.
        Adds 'tag' and 'base_lr' so we can rewarm a group later.
        """
        #TODO: make the base lrs not hardcoded
        if self.config.optimizer == "adam":
            base_lrs = self.model.module.get_adam_base_LRs()
        elif self.config.optimizer == "sgd":
            base_lrs = self.model.module.get_SGD_base_LRs()
        else:
            raise ValueError("Optimizer not supported in build param group")

        # map keyword -> list of params
        buckets = {kw: [] for kw in all_keywords}
        other = []

        for name, p in self.model.named_parameters():
            matched = False
            for kw in all_keywords:
                if kw in name:
                    buckets[kw].append(p)
                    matched = True
                    break
            if not matched:
                other.append(p)

        groups = []
        for kw, params in buckets.items():
            if not params:
                continue
            lr = float(base_lrs.get(kw))
            groups.append({
                "params": params,
                "lr": lr,
                "base_lr": lr,
                "weight_decay": weight_decay,
                "tag": kw,
            })

        if other:  # fallback group for anything unmatched
            if self.config.optimizer == "adam":
                otherbaseLR = 1e-4
            elif self.config.optimizer == "sgd":
                otherbaseLR = 0.01
            
            groups.append({
                "params": other,
                "lr": otherbaseLR,
                "base_lr": otherbaseLR,
                "weight_decay": weight_decay,
                "tag": "other",
            })
            #raise ValueError("Weird Logic in Param Groups")
        return groups

    def print_lrs(self,optimizer, header="Current learning rates:"):
        print("\n" + header)
        for i, g in enumerate(optimizer.param_groups):
            tag = g.get("tag", f"group_{i}")
            print(f"  {i:02d} | {tag:15s} → lr = {g['lr']:.2e}")

    def check_freezing_patience_gradual_step(self):
        """
        function is used if fine_tuning_scheme == 'warmup_by_layer_by_patience'
        """
        if len(self.encoder_layers_reversed) != 0:
            self.recorder.logger.info('*****Unfreeze the encoder gradually *****')
            kw = self.encoder_layers_reversed[0]
            unfreeze_by_keywords(self.model, keywords=[kw])
            self.recorder.logger.info('*****Rewarming learning rate*****')
            if self.config.warmup_lr_mode == 'ResetAll':
                # print(f"reset all lrs to {self.config.lr}")
                reset_all_lrs(self.optimizer, new_lr=self.config.lr)
            elif self.config.warmup_lr_mode == 'RewarmByLayer':
                rewarm_lr_groups(self.optimizer, keywords=[kw])
            self.print_lrs(self.optimizer, header=f"After unfreezing {kw}")
            self.encoder_layers_reversed.remove(self.encoder_layers_reversed[0])
            #self.recorder.logger.info('*****Reinit the optimizer*****')
            #self.init_optimizer_and_scheduler()

    def load_model_state_dict(self, load_from: str):

        self.recorder.logger.info("Loading model from checkpoint '{}'".format(load_from))
        self.resume_checkpoint = torch.load(load_from, map_location=self.device)
        self.start_epoch = self.resume_checkpoint['epoch'] + 1
        self.model.load_state_dict(self.resume_checkpoint['state_dict'], strict=True)

    def load_optimizer_state_dict(self):
        self.recorder.logger.info("Loading optimizer from checkpoint.")
        self.optimizer.load_state_dict(self.resume_checkpoint['optimizer'])

    def load_pretrained_weights(self):

        self.recorder.logger.info("Loading pretrained_model from checkpoint '{}'".format(self.config.pretrained_model))
        checkpoint = torch.load(self.config.pretrained_model, map_location=self.device)
        pretrained_state_dict = checkpoint['state_dict']
        # pretrained_state_dict = checkpoint['online_state_dict']

        # select which layers to transfer
        transferred_layers = []
        for k, v in pretrained_state_dict.items():
            for keyword in self.transferred_dict:
                if k.find(keyword) != -1:
                    transferred_layers.append(k)
                    # print(k)

        # dismatched keys replacement (to ensure consistency between pretrained model layer names and downstream model layer names)
        if self.config.transferred_dismatched_keys is not None:
            transferred_state_dict = {k.replace(self.config.transferred_dismatched_keys[0],
                                                self.config.transferred_dismatched_keys[1]): v
                                      for k, v in pretrained_state_dict.items() if k in transferred_layers}
        else:
            transferred_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in transferred_layers}

        # print(transferred_layers)
        # model_dict = self.model.state_dict()
        # model_dict.update(transferred_state_dict)
        self.model.load_state_dict(transferred_state_dict, strict=False)

        # set all params trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def save_state_dict(self, epoch, full_path):

        if self.scheduler is not None:
            state = {
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'state_dict': self.model.state_dict()
            }
        else:
            state = {
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.model.state_dict()
            }
        torch.save(state, full_path)

    # From here on are the abstract functions that must be implemented by the specific trainer.
    def train(self) -> None:
        """
        Train stage.
        """
        pass

    def eval(self, epoch) -> float:
        """
         Evaluation stage.
        """
        metric = 0.
        return metric
