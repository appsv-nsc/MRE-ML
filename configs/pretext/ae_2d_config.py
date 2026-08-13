import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # may need to be changed to "0,1" if needed or "1"

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(grandparent_dir)

from trainers import *


class Auto_Encoder_config:
    gpu_ids = [0]  # index of gpu to be used from 'CUDA_VISIBLE_DEVICES' defined above.

    # The benchmark flag allows you to enable the inbuilt cudnn auto-tuner to find the best configurations to use for
    # your hardware. Setting it to true can lead to faster training but it is not recommended in cases where the
    # input sizes vary during training or the network architecture contains hidden layers
    # Ref: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    benchmark = False
    # manualseed: any random integer number that shouldn't be changed between one run and another to ensure consistency
    manualseed = 666

    model = 'Simple'  # used to differentiate in the base trainer between simple models initialization and complex
    # models initialization in an extended version..it should always be Simple in autenocder, rotation and downstream tasks

    network = 'unet_2d'
    # supported options: ['normal','xavier', 'kaiming', 'orthogonal']. Implemented in networks/__init__.py
    init_weight_type = 'kaiming'

    # phase and task are both used to name the checkpoint folder path where the logs and weights will be saved.
    # 'phase': should be either 'pretext_task' or 'downstream_task'. 'task': should be related to the type of task
    # being performed options: ['auto_encoder_2d','rot_2d', mapping]
    # if there is an extra information that needs to be put into the folder name. Take note that the dataset name is
    # already added to the folder path through the 'dataset_name' config. Folder path will be:
    # checkpoints/phase/task/dataset_name/network_note/timestamp/ (set in utils/recorder.py )
    phase = 'pretext_task'
    task = 'auto_encoder_2d'
    note = '256x256'
    run_mode = 'individual_run_mode'

    # 'train_dataloader' and 'eval_dataloader' values are used to initialize the dataloaders and
    # 'dataset_name' is used to get the dataset path
    dataset_name = 'deformable_aug_2d_ae'  # key used in datasets/paths.py
    train_dataloader = 'base_ae_2d_pretask'  # key used in datasets/__init__.py
    eval_dataloader = 'base_ae_2d_pretask'  # key used in datasets/__init__.py

    input_size = [256, 256]

    im_channel = 5
    class_num = 5
    normalization = 'linear'

    # data_augmentation applied randomly during pretext to the entire volume. If flip_rate = 0, no flipping
    # will be applied
    flip_rate = 0

    # model pre-training
    verbose = 1
    train_batch = 16
    val_batch = 16

    # supported optimizer options: ['sgd','adam']
    optimizer = "sgd"
    # supported scheduler options: ['ReduceLROnPlateau', 'StepLR', 'StepLR_multi_step', 'Cosine']
    scheduler = 'ReduceLROnPlateau'
    momentum = 0.9
    weight_decay = 0.0
    # nesterov = False
    num_workers = 0
    max_queue_size = num_workers * 4
    epochs = 1000
    # save_model_freq = 500
    patience = 20
    lr = 0.01
    loss = 'mse'

    # 'resume': can be used if the current training stopped and we want to resume it. Should point to the path of the
    # checkpoint to load from
    resume = None
    resume_loss = None # should be None if resume = None otherwise should be set to the best loss achieved by the
    # loaded model for resume

    # 'pretrained_model': used to load other model weights into this one (transfer learning). Should be path of the
    # pretrained model
    pretrained_model = None

    # bec. save_tensorboard_graph sometimes takes up RAM, we can have 'save_tensorboard_graph = False' to avoid
    # memory error related to saving the graph
    save_tensorboard_graph = True

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = Auto_Encoder_config()
    Trainer = AETrainer(config)
    Trainer.train()
