import copy
import random
import time

import numpy as np
from tqdm import tqdm
import os
import torch
from scipy.special import comb
import torchio as tio
import csv
from datasets.pretext.PTP.V_2D.base_ptp_2d_pretask import PTPBase2D
import argparse
from utils.tools import save_tensor2image
from torch.utils.data import DataLoader


# SSM: 2D Rotation prediction (2D-Rot)


class RotPretaskSet2D(PTPBase2D):
    def __init__(self, config, base_dir, flag):
        super(RotPretaskSet2D, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.input_size
        self.all_images = []

        # load data from pre-saved files ".npy"
        self.get_images_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image = self.all_images[index]

        #image = np.expand_dims(image, axis=3)
        #image_tensor = self.transform(image)
        #image_tensor = np.squeeze(image_tensor, axis=3)
        image_tensor = torch.tensor(image, dtype=torch.float32)

        rotated_input, label = self.rotate_tensor(image_tensor)

        return rotated_input, torch.from_numpy(np.array(label)).long()
