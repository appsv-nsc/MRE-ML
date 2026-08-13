import sys
import numpy as np
import random
import torch
from monai.utils import set_determinism
import matplotlib.pyplot as plt
import os


class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def seed_worker(worker_id, manualseed):
    worker_seed = manualseed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_random_and_cudnn(manualseed, use_gpu, use_benchmark=False):
    # Set seed
    if manualseed is None:
        manualseed = random.randint(1, 10000)
    np.random.seed(manualseed)
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    set_determinism(manualseed)

    if use_gpu:
        torch.cuda.manual_seed(manualseed)
        torch.cuda.manual_seed_all(manualseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = use_benchmark


def draw_plots(train_loss_history, val_loss_history, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss', marker='o')
    plt.plot(val_loss_history, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))  # Saves the plot as an image
    plt.close()  # Closes the figure to avoid memory issues
