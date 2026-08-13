import os
import numpy as np
from tqdm import tqdm
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(grandparent_dir)

from trainers import *
import argparse


class mapping_task_config:
    object = ''

    phase = 'downstream_task'
    task = 'mapping'
    run_mode = 'individual_run_mode'
    model = 'Simple'
    init_weight_type = 'kaiming'

    input_size = [256, 256]

    train_dataloader = 'base_mre_map_2d'
    eval_dataloader = 'base_mre_map_2d'
    im_channel = 9
    out_ch = 2
    class_num = 2
    normalization = 'linear'

    # model
    optimizer = 'adam'  # 'sgd'
    lr = 1e-4
    momentum = 0.9
    weight_decay = 0.0
    scheduler = 'ReduceLROnPlateau'
    patience = 40  # 20
    verbose = 1
    train_batch = 16

    val_batch = 16

    val_freq = 1
    save_model_freq = 50
    num_workers = 0
    max_queue_size = num_workers * 1
    epochs = 500
    loss = 'target_mask_weighted_mse'  # or 'mse'
    loss_mul_factor = 5  # used with: loss = 'target_weighted_mse' only

    # pretrained_model = None (if you need to train downstream from scratch)
    pretrained_model = 'checkpoints/full_pipeline/20260411/pretext_task_mre_ssl/rot_2d/rot_unet_dense256x256/SSM_ROT.pth'
    transferred_part = 'encoder'
    # 'transferred_dismatched_keys' is used to replace the
    # start of the name of the layers from the original pretrained model(e.g., module.encoder.conv1) by the name that
    # is generalizable for the downstream task (e.g., module.conv1)
    # pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    # could be None if no changes between both exists
    transferred_dismatched_keys = ['module.encoder.', 'module.']

    # supported options: ['full', 'fixed', 'warmup_by_layer_by_patience']
    fine_tuning_scheme = 'fixed'

    warmup_patience = 20  # used only with fine_tuning_scheme = 'warmup_by_layer_by_patience' else should be None

    warmup_lr_mode = 'ResetAll' # or 'RewarmByLayer'

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = mapping_task_config()
    Trainer = Mapping2DTrainer(config)
    Trainer.train()
