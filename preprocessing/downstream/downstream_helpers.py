import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import nibabel as nib
import numpy as np
import os
import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_nifti_data(file_path):
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data


def find_single_file_with_two_keywords(path, keyword1, keyword2):
    for root, _, files in os.walk(path):
        for file_name in files:
            if keyword1.lower() in file_name.lower() and keyword2.lower() in file_name.lower():
                return os.path.join(root, file_name)  # return full path immediately
    return None  # if not found


def visualize_mask_centerOfMass(mask, cx, cy,save_visualizations_path,sliceIdx):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.scatter([cx], [cy], color='red', s=80, marker='x', label='Center of Mass')
    #plt.title("Mask with Center of Mass")
    #plt.legend()
    plt.axis('off')
    #plt.show()
    plt.savefig(save_visualizations_path + "/" + str(sliceIdx) + "_CenterOfMass.png", bbox_inches='tight',
                   pad_inches=0)
    plt.close()


def visualize_process(original, clipped, masked, mask):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("original")

    axs[1].imshow(clipped, cmap='gray')
    axs[1].set_title("phase clipped")

    axs[2].imshow(masked, cmap='gray')
    axs[2].set_title("phase masked")

    axs[3].imshow(mask, cmap='gray')
    #axs[3].set_title("segmentation mask")

    # Remove axes for clarity
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_mask_center(mask,save_visualizations_path,sliceIdx, area_threshold=10):
    if mask.max() != 0:
        cy, cx = center_of_mass(mask)
        cy, cx = int(round(cy)), int(round(cx))
        status = "success"
    else:
        cy, cx = 0, 0
        status = "fail"
    # print(cy, cx)

    visualize_mask_centerOfMass(mask, cx, cy,save_visualizations_path,sliceIdx)
    return cy, cx, status


def compute_clip_boundaries(cy, cx, crop_size=256, image_shape=320):
    """assumes square image"""

    half_crop = crop_size // 2

    # Compute crop boundaries
    top = max(cy - half_crop, 0)
    bottom = min(cy + half_crop, image_shape)
    left = max(cx - half_crop, 0)
    right = min(cx + half_crop, image_shape)

    # Adjust to maintain crop size
    if bottom - top < crop_size:
        if top == 0:
            bottom = crop_size
        elif bottom == image_shape:
            top = image_shape - crop_size
        print("clipping adjusted to avoid image boundaries")

    if right - left < crop_size:
        if left == 0:
            right = crop_size
        elif right == image_shape:
            left = image_shape - crop_size
        print("clipping adjusted to avoid image boundaries")

    return top, bottom, left, right