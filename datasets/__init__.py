from datasets.paths import Path
from torch.utils.data import DataLoader
from datasets.pretext.AE.base_ae_2d_pretask import AEBase2d
from datasets.pretext.PTP.V_2D.base_rot_2d_pretask import RotPretaskSet2D
from datasets.downstream.data_2d_mre_map import MREDataset

dataloaders_dict = {
    'base_ae_2d_pretask': AEBase2d,
    'base_rot_2d_pretask': RotPretaskSet2D,

    'base_mre_map_2d': MREDataset
}


def get_dataloder(args, flag="train", drop_last=True):
    '''
    :return: the dataloader of special datasets
    '''

    if hasattr(args, 'dataset_path'):
        root = args.dataset_path
    else:
        root = Path.db_root_dir(args.dataset_name)

    if flag == "train":
        print('******Building training dataloder******')
        dataloader_name = args.train_dataloader
        assert dataloader_name in dataloaders_dict.keys(), "The dataloader use {} does not exist ".format(dataloader_name)
        dataset = dataloaders_dict[dataloader_name](config=args, base_dir=root, flag=flag)
        batch_size = args.train_batch
        shuffle = True
        num_workers = args.num_workers
        pin_memory = True
    else:
        print('******Building test dataloder******')
        dataloader_name = args.eval_dataloader
        assert dataloader_name in dataloaders_dict.keys(), "The dataloader use {} does not exist ".format(dataloader_name)
        dataset = dataloaders_dict[dataloader_name](config=args, base_dir=root, flag=flag)
        batch_size = args.val_batch
        shuffle = False
        # num_workers = args.num_workers
        num_workers = 0
        pin_memory = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=drop_last)

    return dataset, data_loader
