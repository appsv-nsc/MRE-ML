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


def visualize_mask_centerOfMass(mask, cx, cy):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.scatter([cx], [cy], color='red', s=80, marker='x', label='Center of Mass')
    plt.title("Mask with Center of Mass")
    plt.legend()
    plt.axis('off')
    plt.show()


def visualize_process(original, clipped, masked, mask):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("original")

    axs[1].imshow(clipped, cmap='gray')
    axs[1].set_title("phase clipped")

    axs[2].imshow(masked, cmap='gray')
    axs[2].set_title("phase masked")

    axs[3].imshow(mask, cmap='gray')
    axs[3].set_title("segmentation mask")

    # Remove axes for clarity
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_mask_center(mask, area_threshold=10):
    # Step 1: Label connected components
    """print(np.unique(mask))
    labeled_mask, num_labels = label(mask)

    # Step 2: Filter blobs by size
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels + 1):
        blob = (labeled_mask == i)
        if blob.sum() >= area_threshold:
            filtered_mask[blob] = 1  # keep this blob

    # Step 3: Compute global center of mass
    if filtered_mask.sum() == 0:
        raise ValueError("No blobs passed the area threshold!")"""

    if mask.max()!=0:
        cy, cx = center_of_mass(mask)
        cy, cx = int(round(cy)), int(round(cx))
        status = "success"
    else:
        cy, cx = 0, 0
        status = "fail"
    # print(cy, cx)

    #visualize_mask_centerOfMass(mask, cx, cy)
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


def get_timeframe_segmentation_data(patient_folder_path, slice_index, timeframe_index):
    segmentations_folder_path = os.path.join(patient_folder_path, "segmentations/liver")
    timeframe_segmentation_file_path = os.path.join(segmentations_folder_path,
                                                    "slice_" + str(slice_index + 1) + "_timeframe_" + str(
                                                        timeframe_index + 1) + "_M.npy")
    timeframe_segmentation = np.load(timeframe_segmentation_file_path)

    cy, cx, status = get_mask_center(timeframe_segmentation)

    print("centroid for " + patient_folder_path + " slice number " + str(slice_index + 1) + " timeframe " + str(
        timeframe_index + 1) + " is: (" + str(cy) + "," + str(cx) + ")")

    top, bottom, left, right = compute_clip_boundaries(cy, cx, 256, 320)

    clipped_mask = timeframe_segmentation[top:bottom, left:right]
    # boundaries = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return top, bottom, left, right, clipped_mask


def save_processed_data(processed_data_list, clipped_data_list, masked_data_list, output_dir_path):
    """
    concatenates the separate patient arrays into one collective array for each type (normal, clipped, clipped and masked)
    and saves them as .npy
    Args:
        processed_data_list (List): List of processed patient arrays
        clipped_data_list (List): list of processed and clipped patient arrays
        masked_data_list (List): list of processed, clipped and masked patient arrays
        output_dir_path (str): output directory path
    Returns:
        None, but saves the output numpy arrays in output_dir_path
    """
    try:
        all_patients_data = np.concatenate(processed_data_list, axis=0)
        all_patients_data_clipped = np.concatenate(clipped_data_list, axis=0)
        all_patients_data_clipped_masked = np.concatenate(masked_data_list, axis=0)

        output_save_path = os.path.join(output_dir_path, "mre_processed_mode_" + ".npy")
        np.save(output_save_path, all_patients_data)
        print("processed mre of shape " + str(all_patients_data.shape) + " saved to: " + output_save_path)

        output_save_path = os.path.join(output_dir_path, "mre_processed_clipped_mode_" + ".npy")
        np.save(output_save_path, all_patients_data_clipped)
        print("processed mre of shape " + str(all_patients_data_clipped.shape) + " saved to: " + output_save_path)

        output_save_path = os.path.join(output_dir_path, "mre_processed_clipped_masked_mode_"+ ".npy")
        np.save(output_save_path, all_patients_data_clipped_masked)
        print("processed mre of shape " + str(all_patients_data_clipped_masked.shape) + " saved to: " + output_save_path)
    except Exception as e:
        print(f"An error occurred: {e}")
