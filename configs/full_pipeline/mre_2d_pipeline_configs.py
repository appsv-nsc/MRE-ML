import os
import sys

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(grandparent_dir)

from full_pipeline.full_2d_pipeline import full_pipeline_manager

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class pipeline_2d_configs:
    gpu_ids = [0]  # index of gpu to be used from 'CUDA_VISIBLE_DEVICES' defined above.

    # The benchmark flag allows you to enable the inbuilt cudnn auto-tuner to find the best configurations to use for
    # your hardware. Setting it to true can lead to faster training but it is not recommended in cases where the
    # input sizes vary during training or the network architecture contains hidden layers
    # Ref: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    benchmark = False

    # manualseed: any random integer number that shouldn't be changed between one run and another to ensure consistency
    manualseed = 123

    use_skip_in_pretext = False  # use skip connections in pretext tasks where applicable
    use_skip_in_downstream = True  # use skip connections in downstream tasks where applicable

    # network configs
    network_backbones = ['unet_2d']

    pretext_tasks_to_apply = []  # ['autoencoder', 'rot'] or if empty brackets if no pretraining is needed or a list containing one pretext task

    downstream_task = 'mapping'
    pretext_dataset_name = 'mre_ssl'  # name used for logging, doesn't have significance
    downstream_dataset_name = 'mre'  # name used for logging, doesn't have significance

    run_mode = 'full_pipeline'  # must always be 'full_pipeline' in this case

    # bec. save_tensorboard_graph sometimes takes up RAM, we can have 'save_tensorboard_graph = False' to avoid
    # memory error related to saving the graph. This is only related to saving computational graph not loss graphs and summaries.
    save_tensorboard_graph = False

    pretext_preprocessed_data_path = '/MRE_pretext_data/augmented_preprocessed_pretext_data'
    # already preprocessed data should be saved as a single .npy for each split with the names
    # 'bat_train_HeightxWidth.npy','bat_valid_HeightxWidth.npy'

    downstream_preprocessed_data_path = 'E:/MREData/augmented_preprocessed_data'

    downstream_inference_save_path = 'E:/mre-ml/results'  # downstream inference results path

    downstream_num_img_visualize_in_tensorboard = 2


class autoencoder_configs:
    model = 'Simple'  # used to differentiate in the base trainer between simple models initialization and complex
    # models initialization in an extended version..it should always be Simple in autenocder, rotation and downstream tasks

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
    note = '' #ptional if you want to add something to the checkpoint folder name

    # 'train_dataloader' and 'eval_dataloader' values are used to initialize the dataloaders
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

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


class rot_configs:
    model = 'Simple'  # used to differentiate in the base trainer between simple models initialization and complex
    # models initialization in an extended version..it should always be Simple in autenocder, rotation and downstream tasks

    # supported options: ['normal','xavier', 'kaiming', 'orthogonal']. Implemented in networks/__init__.py
    init_weight_type = 'kaiming'

    # phase and task are both used to name the checkpoint folder path where the logs and weights will be saved.
    # 'phase': should be either 'pretext_task' or 'downstream_task'. 'task': should be related to the type of task
    # being performed options: ['auto_encoder_2d','rot_2d', mapping]
    # if there is an extra information that needs to be put into the folder name. Take note that the dataset name is
    # already added to the folder path through the 'dataset_name' config. Folder path will be:
    # checkpoints/phase/task/dataset_name/network_note/timestamp/ (set in utils/recorder.py )
    phase = 'pretext_task'
    task = 'rot_2d'
    note = '256x256'

    # data
    input_size = [256, 256]

    # 'train_dataloader' and 'eval_dataloader' values are used to initialize the dataloaders
    train_dataloader = 'base_rot_2d_pretask'  # key used in datasets/__init__.py
    eval_dataloader = 'base_rot_2d_pretask'  # key used in datasets/__init__.py

    im_channel = 5
    # number of rotation classes (should be 4 in 2d)
    class_num = 4
    normalization = None  # for cross_entropy loss

    # model pre-training
    verbose = 1
    train_batch = 16
    val_batch = 16
    # val_epoch = 10
    # supported optimizer options: ['sgd','adam']
    optimizer = 'sgd'  # "adam"
    momentum = 0.9
    weight_decay = 0.0
    # supported scheduler options: ['ReduceLROnPlateau', 'StepLR', 'StepLR_multi_step', 'Cosine']
    scheduler = 'ReduceLROnPlateau'
    learning_rate_decay = [250]
    num_workers = 0
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 30
    patience = 20
    lr = 0.01  # 0.001
    loss = 'ce'

    # 'resume': can be used if the current training stopped and we want to resume it. Should point to the path of the
    # checkpoint to load from
    resume = None
    resume_acc = None # should be None if resume = None otherwise should be set to the best acc. achieved by the loaded model for resume
    # 'pretrained_model': used to load other model weights into this one (transfer learning). Should be path of the
    # pretrained model
    pretrained_model = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


class mapping_task_config:
    object = ''

    phase = 'downstream_task'
    task = 'mapping'

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

    patience = 40
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
    pipeline_config = pipeline_2d_configs()
    pretext_configs = {}
    downstream_configs = {}

    if 'autoencoder' in pipeline_config.pretext_tasks_to_apply:
        ae_cfgs = autoencoder_configs()
        pretext_configs['autoencoder'] = ae_cfgs

    if 'rot' in pipeline_config.pretext_tasks_to_apply:
        rot_cfgs = rot_configs()
        pretext_configs['rot'] = rot_cfgs

    if pipeline_config.downstream_task == 'mapping':
        downstream_configs['task_cfgs'] = mapping_task_config()

    full_pipeline_manager = full_pipeline_manager(pipeline_config, pretext_configs, downstream_configs)
    full_pipeline_manager.run_pipeline()
