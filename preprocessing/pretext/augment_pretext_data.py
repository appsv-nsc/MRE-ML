import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import cv2
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(grandparent_dir)
print(grandparent_dir)

from pretext_augmenter import *
from pretext_preprocess_helpers import Logger

print(torch.__version__)


def main():
    train_array_path = "/MRE_pretext_data/preprocessed_pretext_data/bat_train_256x256.npy"
    train_mask_path = "/MRE_pretext_data/preprocessed_pretext_data/bat_train_mask_256x256.npy"
    valid_array_path = "/MRE_pretext_data/preprocessed_pretext_data/bat_valid_256x256.npy"
    valid_mask_path = "/MRE_pretext_data/preprocessed_pretext_data/bat_valid_mask_256x256.npy"
    save_dir = "/MRE_pretext_data/augmented_preprocessed_pretext_data"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'SSL_augmentations_log.txt')
    sys.stdout = Logger(log_path)

    augment_pretraining(train_array_path, train_mask_path, valid_array_path, valid_mask_path,  save_dir)

def augment_pretraining(train_array_path, train_mask_path,valid_array_path, valid_mask_path,save_dir):
    train_array = np.load(train_array_path)
    train_masks = np.load(train_mask_path)
    train_array = np.transpose(train_array, (0,2,3,1))
    print("original train array shape", train_array.shape)
    print("original mask array shape", train_masks.shape)
    valid_array = np.load(valid_array_path)
    valid_masks = np.load(valid_mask_path)
    valid_array = np.transpose(valid_array, (0,2,3,1))
    print("original valid array shape", valid_array.shape)
    print("original valid mask shape", valid_masks.shape)

    visualizations_dir = os.path.join(save_dir, "augmentation_visualizations")
    nOriginalTrain = train_array.shape[0]
    nOriginalValid = valid_array.shape[0]

    train_rot_deg = 30.0
    train_translate_fr = ((-0.1, 0.1), (-0.1, 0.1))
    train_apply_noise_prb = 0.9
    train_noise_pct = (0.3, 0.6)
    train_rot_prob = 0.6  # 80% chance to rotate
    train_trans_prob = 0.8  # 50% chance to translate
    n_augmentations_normal_train = 6

    print(
        f"augmentation parameters: rotation degrees {train_rot_deg}, translate frac {train_translate_fr}, apply_noise_prob{train_apply_noise_prb}, noise_pct {train_noise_pct}, n_augmentations for train slice: {n_augmentations_normal_train}, n_augmentations for valid slice: {n_augmentations_normal_valid}")
    print(f"train_rot_prob {train_rot_prob}, train_trans_prob {train_trans_prob}")
    
    train_aug_obj = TVPairedAffineNoiseAugmentor(
    rot_degrees=train_rot_deg,
    translate_frac=train_translate_fr,
    rot_prob=train_rot_prob,
    trans_prob=train_trans_prob,
    apply_noise_prob=train_apply_noise_prb,
    noise_pct=train_noise_pct,
    seed=123
    )

    curr_visualizations_dir = os.path.join(visualizations_dir, "train")
    os.makedirs(curr_visualizations_dir, exist_ok=True)
    train_reconstr_target_array = train_array.copy()
    for sliceIdx in range(nOriginalTrain):
        print(f"augmenting train {sliceIdx}/{nOriginalTrain}")
        train_array, train_masks, train_reconstr_target_array = augment_slice(train_aug_obj,
                                            slice_index=sliceIdx,
                                            input_array=train_array,
                                            mask_array=train_masks,
                                            reconstr_target_array=train_reconstr_target_array,
                                            n_augmentations=n_augmentations_normal_train,
                                            visualizations_path=curr_visualizations_dir)

    valid_reconstr_target_array = valid_array.copy()

    train_array = np.transpose(train_array, (0,3,1,2))
    valid_array = np.transpose(valid_array, (0,3,1,2))
    print("train shape after augmentations: ",train_array.shape)
    print("valid shape: ",valid_array.shape)
    print("train y shape after augmentations: ",train_reconstr_target_array.shape)
    print("valid y shape: ",valid_reconstr_target_array.shape)
    train_save_path = os.path.join(save_dir,"bat_train_256x256.npy")
    np.save(train_save_path, train_array)

    train_save_path = os.path.join(save_dir,"bat_train_y_256x256.npy")
    np.save(train_save_path, train_reconstr_target_array)

    valid_save_path = os.path.join(save_dir,"bat_valid_256x256.npy")
    np.save(valid_save_path, valid_array)

    valid_save_path = os.path.join(save_dir,"bat_valid_y_256x256.npy")
    np.save(valid_save_path, valid_reconstr_target_array)

def augment_slice(augObj, slice_index, input_array, mask_array,reconstr_target_array, n_augmentations=1):
    # augment extreme train slices
    slice_to_augment_x = input_array[slice_index].copy()
    slice_to_augment_seg = mask_array[slice_index].copy()
    
    x_aug, y_aug, m_aug, _ = augObj(
        slice_to_augment_x,     # x_in
        None,                   # y_in (None => y = affine(x) by design)
        slice_to_augment_seg,   # mask_in
        None,                   # banana_mask_in
        channels_last=True, n_augmentations=n_augmentations
    )

    input_array = np.concatenate([input_array, x_aug], axis=0)
    m_aug = m_aug.squeeze(1)
    mask_array = np.concatenate([mask_array, m_aug], axis=0)
    reconstr_target_array = np.concatenate([reconstr_target_array, y_aug], axis=0)
    return_input_array = input_array
    return_mask_array = mask_array
    return_reconstr_array = reconstr_target_array

    return return_input_array, return_mask_array, return_reconstr_array

main()