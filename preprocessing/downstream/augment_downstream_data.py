import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from helpers import *
import sys
from downstream_augmenter import *
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use("Agg")  # ensure no GUI backend


def augment_slice(augObj, slice_index, input_array, target_array, mask_array
                  , banana_array, n_augmentations=1
                  , flag="train", mode="normal", visualizations_path=None):
    # augment extreme train slices
    slice_to_augment_x = input_array[slice_index].copy()
    slice_to_augment_y = target_array[slice_index].copy()
    slice_to_augment_seg = mask_array[slice_index].copy()
    slice_to_augment_banana = banana_array[slice_index].copy()

    for nAugments in range(n_augmentations):
        x_aug, y_aug, m_aug, b_aug = augObj(slice_to_augment_x, slice_to_augment_y,
                                            slice_to_augment_seg, slice_to_augment_banana,
                                            channels_last=True)
        x_aug = np.expand_dims(x_aug, axis=0)
        y_aug = np.expand_dims(y_aug, axis=0)
        m_aug = np.expand_dims(m_aug, axis=0)
        b_aug = np.expand_dims(b_aug, axis=0)

        # plt.figure()
        plt.imshow(slice_to_augment_seg.astype("uint8"), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        # plt.show()
        plt.savefig(visualizations_path + f"/SliceIndex{slice_index}_nAugment{nAugments}_original_mask.png", dpi=100,
                    pad_inches=0)
        plt.clf()
        plt.close()

        # plt.figure()
        plt.imshow(slice_to_augment_banana.astype("uint8"), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        plt.savefig(visualizations_path + f"/SliceIndex{slice_index}_nAugment{nAugments}_original_banana.png", dpi=100,
                    pad_inches=0)
        plt.clf()
        plt.close()

        # plt.figure()
        plt.imshow(m_aug[0].astype("uint8"), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        plt.savefig(visualizations_path + f"/SliceIndex{slice_index}_nAugment{nAugments}_augmented_mask.png", dpi=100,
                    pad_inches=0)
        plt.clf()
        plt.close()

        plt.figure()
        plt.imshow(b_aug[0].astype("uint8"), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        plt.savefig(visualizations_path + f"/SliceIndex{slice_index}_nAugment{nAugments}_augmented_banana.png", dpi=100,
                    pad_inches=0)
        plt.clf()
        plt.close()

        input_array = np.concatenate([input_array, x_aug], axis=0)
        target_array = np.concatenate([target_array, y_aug], axis=0)
        mask_array = np.concatenate([mask_array, m_aug], axis=0)
        banana_array = np.concatenate([banana_array, b_aug], axis=0)
        return_input_array = input_array
        return_target_array = target_array
        return_mask_array = mask_array
        return_banana_array = banana_array

    return return_input_array, return_target_array, return_mask_array, return_banana_array


def augment_mre_data(root_path, save_dir, drop_magnitude):
    np.random.seed(42)
    train_target_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "train", "output"))
    val_target_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "valid", "output"))
    test_target_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "test", "output"))

    train_input_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "train", "input"))
    val_input_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "valid", "input"))
    test_input_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "test", "input"))

    train_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "train", "seg"))
    val_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "valid", "seg"))
    test_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "test", "seg"))

    train_croi_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "train", "croi"))
    val_croi_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "valid", "croi"))
    test_croi_mask_path = os.path.join(root_path, find_single_file_with_two_keywords(root_path, "test", "croi"))

    train_target_array = np.load(train_target_path)
    val_target_array = np.load(val_target_path)
    test_target_array = np.load(test_target_path)

    train_input_array = np.load(train_input_path)
    val_input_array = np.load(val_input_path)
    test_input_array = np.load(test_input_path)

    train_mask_array = np.load(train_mask_path)
    val_mask_array = np.load(val_mask_path)
    test_mask_array = np.load(test_mask_path)

    train_croi_array = np.load(train_croi_mask_path)
    val_croi_array = np.load(val_croi_mask_path)
    test_croi_array = np.load(test_croi_mask_path)

    visualizations_dir = os.path.join(save_dir, "augmentation_visualizations")
    nOriginalTrain = train_input_array.shape[0]

    train_rot_deg = 30.0
    train_translate_fr = ((-0.1, 0.1), (-0.1, 0.1))
    train_apply_noise_prb = 0.9
    train_noise_pct = (0.3, 0.6)
    train_rot_prob = 0.6  # 80% chance to rotate
    train_trans_prob = 0.8  # 50% chance to translate
    n_augmentations_normal_train = 12
    n_augmentations_extreme_train = 36
    print(
        f"Train augmentation parameters: rotation degrees {train_rot_deg}, translate frac {train_translate_fr}, apply_noise_prob{train_apply_noise_prb}, noise_pct {train_noise_pct}, n_augmentations for normal slice: {n_augmentations_normal_train}, n_augmentations for extreme slice: {n_augmentations_extreme_train}")
    print(f"train_rot_prob {train_rot_prob}, train_trans_prob {train_trans_prob}")

    aug_obj = TVPairedAffineNoiseAugmentor(rot_degrees=train_rot_deg, translate_frac=train_translate_fr,
                                                 rot_prob=train_rot_prob, trans_prob=train_trans_prob,
                                                 apply_noise_prob=train_apply_noise_prb, noise_pct=train_noise_pct,
                                                 seed=123)

    curr_visualizations_dir = os.path.join(visualizations_dir, "train")
    os.makedirs(curr_visualizations_dir, exist_ok=True)
    for trainSliceIdx in range(nOriginalTrain):
        print(f"train {trainSliceIdx} of {nOriginalTrain}")
        train_input_array, train_target_array, train_mask_array, train_croi_array = augment_slice(aug_obj,
                                                                                                    slice_index=trainSliceIdx,
                                                                                                    input_array=train_input_array,
                                                                                                    target_array=train_target_array,
                                                                                                    mask_array=train_mask_array,
                                                                                                    banana_array=train_croi_array,
                                                                                                    n_augmentations=n_augmentations_normal_train,
                                                                                                    visualizations_path=curr_visualizations_dir)

    train_extreme_list = [32,33,2,3,37,55]
    for ele in train_extreme_list:
        train_input_array, train_target_array, train_mask_array, train_croi_array = augment_slice(aug_obj,
                                                                                                    slice_index=ele,
                                                                                                    input_array=train_input_array,
                                                                                                    target_array=train_target_array,
                                                                                                    mask_array=train_mask_array,
                                                                                                    banana_array=train_croi_array,
                                                                                                    n_augmentations=(n_augmentations_extreme_train - n_augmentations_normal_train),
                                                                                                    visualizations_path=curr_visualizations_dir)

    if drop_magnitude:
        train_input_array = train_input_array[..., :4]
        val_input_array = val_input_array[..., :4]
        test_input_array = test_input_array[..., :4]

    print("Saving split data")
    print("train shape: ", train_input_array.shape)
    print("valid shape: ", val_input_array.shape)
    print("train shape: ", test_input_array.shape)

    np.save(os.path.join(save_dir, "input_train.npy"), train_input_array)
    np.save(os.path.join(save_dir, "input_valid.npy"), val_input_array)
    np.save(os.path.join(save_dir, "input_test.npy"), test_input_array)

    np.save(os.path.join(save_dir, "lroi_seg_train.npy"), train_mask_array)
    np.save(os.path.join(save_dir, "lroi_seg_valid.npy"), val_mask_array)
    np.save(os.path.join(save_dir, "lroi_seg_test.npy"), test_mask_array)

    np.save(os.path.join(save_dir, "output_train.npy"), train_target_array)
    np.save(os.path.join(save_dir, "output_valid.npy"), val_target_array)
    np.save(os.path.join(save_dir, "output_test.npy"), test_target_array)

    np.save(os.path.join(save_dir, "croi_seg_train.npy"), train_croi_array)
    np.save(os.path.join(save_dir, "croi_seg_valid.npy"), val_croi_array)
    np.save(os.path.join(save_dir, "croi_seg_test.npy"), test_croi_array)


data_path = "E:/MREData/preprocessed_data" # path of preprocessed data generated from preprocess_downstream_data.py
save_dir = "E:/MREData/augmented_preprocessed_data"
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, f'augment_downstream_data_log.txt')
sys.stdout = Logger(log_path)
augment_mre_data(data_path, save_dir, drop_magnitude=False)
