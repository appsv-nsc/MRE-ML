import random
import numpy as np
from torch.utils.data import Dataset
import torch
import copy
from tqdm import tqdm
import os


class AEBase2d(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.all_y_images = []
        self.flag = flag
        self.crop_size = config.input_size
        self.flip_rate = config.flip_rate

        # load data from pre-saved files ".npy"
        self.get_images_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        input = self.all_images[index]
        target = self.all_y_images[index]

        # Autoencoder
        gt = copy.deepcopy(target)

        # Flipping
        input, gt = self.data_augmentation(input, gt, self.flip_rate)

        return torch.from_numpy(input.copy()).float(), torch.from_numpy(gt.copy()).float()

    def get_images_list(self):
        self.all_images = []

        file_name = "bat_" + str(self.flag) + "_" \
                    + str(self.crop_size[0]) + "x" \
                    + str(self.crop_size[1]) + ".npy"
        
        y_file_name = "bat_" + str(self.flag) + "_y_" \
            + str(self.crop_size[0]) + "x" \
            + str(self.crop_size[1]) + ".npy"

        print('***file_name**:', file_name)
        s = np.load(os.path.join(self.base_dir, file_name))
        self.all_images.extend(s)
        self.all_images = np.array(self.all_images)

        print("x_{}: {} | {:.2f} ~ {:.2f}".format(self.flag, self.all_images.shape, np.min(self.all_images),
                                                  np.max(self.all_images)))

        print('***y_file_name**:', y_file_name)
        y = np.load(os.path.join(self.base_dir, y_file_name))
        self.all_y_images.extend(y)
        self.all_y_images = np.array(self.all_y_images)
        self.all_y_images = np.transpose(self.all_y_images, (0,3,1,2))
        print("x_{}: {} | {:.2f} ~ {:.2f}".format(self.flag, self.all_y_images.shape, np.min(self.all_y_images),
                                                  np.max(self.all_y_images)))
        return

    def data_augmentation(self, x, y, prob=0.5):
        # augmentation by flipping
        cnt = 3
        while random.random() < prob and cnt > 0:
            degree = random.choice([0, 1, 2])
            x = np.flip(x, axis=degree)
            y = np.flip(y, axis=degree)
            cnt = cnt - 1

        return x, y
