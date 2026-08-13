import os
import numpy as np
import json

with open("/MRE_pretext_data/train_folders_list.json", "r") as f:
    train_folders = json.load(f)
with open("/MRE_pretext_data/val_folders_list.json", "r") as f:
    val_folders = json.load(f)

base_preprocessed_path = '/MRE_pretext_data/preprocessed_pretext_data/'
# Initialize a list to collect arrays
data_arrays = []
mask_arrays = []

for split in ["train","valid"]:

    src_directory_path = os.path.join(base_preprocessed_path,split)
    output_directory_path = os.path.join(base_preprocessed_path,"all_data")

    folder_list = train_folders if split == "train" else val_folders
    # Loop through each patient preprocessed .npy file and load it
    for patient in folder_list:
        patient_processed_array_path = os.path.join(src_directory_path, patient+"_clipped_masked.npy")
        data = np.load(patient_processed_array_path)  # shape: (n, 256, 256, 5)
        data_arrays.append(data)

        patient_processed_mask_path = os.path.join(src_directory_path, patient+"_roi_mask.npy")
        mask = np.load(patient_processed_mask_path)  # shape: (n, 256, 256, 5)
        mask_arrays.append(mask)

    # Concatenate all loaded arrays along axis 0
    concatenated_data_array = np.concatenate(data_arrays, axis=0)  # shape: (totalN, 256, 256, 5)
    concatenated_data_array = np.transpose(concatenated_data_array,(0, 3, 1, 2))
    print("Final shape:", concatenated_data_array.shape)

    output_save_path = os.path.join(output_directory_path, f"bat_{split}_256x256" + ".npy")
    np.save(output_save_path, concatenated_data_array)

    # Concatenate all loaded arrays along axis 0
    concatenated_mask_array = np.concatenate(mask_arrays, axis=0)  # shape: (totalN, 256, 256, 5)
    print("Final shape:", concatenated_mask_array.shape)
    output_save_path = os.path.join(output_directory_path, f"bat_{split}_mask_256x256" + ".npy")
    np.save(output_save_path, concatenated_mask_array)
