import sys
import random
import os
from tqdm import tqdm
import gc
import pandas as pd
import json
from multiprocessing import Process
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(grandparent_dir)
print(grandparent_dir)

from pretext_helpers import *


def process_datasets(patient_idx, patient_folder, data_base_path, output_path, normalize_magnitude=True):

    try:
        patient_folder_full_path = os.path.join(data_base_path, patient_folder)
        phase_folder_full_path = os.path.join(patient_folder_full_path, "phase")

        print("Current patient folder being processed: ", patient_folder)

        number_slices = len(os.listdir(phase_folder_full_path)) // 4
        #print(patient_folder_full_path,number_slices)
        cur_patient_data = np.zeros((number_slices, 320, 320, 5))
        cur_patient_data_clipped = np.zeros((number_slices, 256, 256, 5))
        cur_patient_data_clipped_masked = np.zeros((number_slices, 256, 256, 5))
        cur_patient_data_mask = np.zeros((number_slices, 256, 256))

        for slice_idx in range(number_slices):
            cur_slice = "slice_" + str(slice_idx + 1)

            phase_timeframes = [
                f for f in os.listdir(phase_folder_full_path)
                if cur_slice in f
            ]

            for timeframe_idx, timeframe_file in enumerate(phase_timeframes):
                timeframe_phase_file_path = os.path.join(phase_folder_full_path,
                                                            timeframe_file)
                phase_data = get_nifti_data(timeframe_phase_file_path)

                phase_data = np.rot90(phase_data, k=1)
                phase_data = np.fliplr(phase_data)

                cur_patient_data[slice_idx, :, :, timeframe_idx] = phase_data


                top, bottom, left, right, clipped_mask = \
                    get_timeframe_segmentation_data(patient_folder_path=patient_folder_full_path,
                                                    slice_index=slice_idx,
                                                    timeframe_index=timeframe_idx, patient_index=patient_idx
                                                    , seg_class="liver_without_vessels", apply_erosion=True)
 
                #print("Clipped mask shape:",clipped_mask.shape)
                cur_patient_data_mask[slice_idx,:,:] = clipped_mask
                cur_patient_data_clipped[slice_idx, :, :, timeframe_idx] = cur_patient_data[slice_idx,
                                                                            top:bottom,
                                                                            left:right,
                                                                            timeframe_idx] 

                cur_patient_data_clipped_masked[slice_idx, :, :, timeframe_idx] = np.where(clipped_mask == 1,
                                                                                            cur_patient_data_clipped[
                                                                                            slice_idx, :, :,
                                                                                            timeframe_idx], 0)


                if timeframe_idx == 0:
                    magnitude_folder_full_path = os.path.join(patient_folder_full_path, "magnitude")
                    magnitude_file_path = os.path.join(magnitude_folder_full_path, cur_slice + '_timeframe_1_M.nii')
                    magnitude_data = get_nifti_data(magnitude_file_path)
                    magnitude_data = np.rot90(magnitude_data, k=1)
                    magnitude_data = np.fliplr(magnitude_data)

                    cur_patient_data[slice_idx, :, :, -1] = magnitude_data

                    cur_patient_data_clipped[slice_idx, :, :, -1] = cur_patient_data[slice_idx,
                                                                    top:bottom,
                                                                    left:right,
                                                                    -1] 
                    cur_patient_data_clipped_masked[slice_idx, :, :, -1] = np.where(clipped_mask == 1,
                                                                                    cur_patient_data_clipped[
                                                                                    slice_idx, :, :, -1], 0)
                    if normalize_magnitude:
                        masked_magnitude_values = cur_patient_data_clipped_masked[slice_idx, :, :, -1][clipped_mask==1]
                        min_val = np.min(masked_magnitude_values)
                        max_val = np.max(cur_patient_data_clipped_masked[slice_idx, :, :, -1])
                        if max_val > min_val:
                            cur_patient_data_clipped_masked[slice_idx, :, :, -1] = (cur_patient_data_clipped_masked[slice_idx, :, :, -1] - min_val) / (max_val - min_val)
                            cur_patient_data_clipped_masked[slice_idx, :, :, -1][~clipped_mask] = 0.0
                        else:
                            print("Max = 0 or Max < Min")
                            cur_patient_data_clipped_masked[slice_idx, :, :, -1] = 0.0  # fallback if constant


        cur_patient_data_clipped_masked_arr = cur_patient_data_clipped_masked.astype(np.float32)
        print(cur_patient_data_clipped_masked_arr.shape)
        np.save(os.path.join(output_path, f"{patient_folder}_clipped_masked.npy"), cur_patient_data_clipped_masked_arr)
        np.save(os.path.join(output_path, f"{patient_folder}_roi_mask.npy"), cur_patient_data_mask)
        # Cleanup memory
        del phase_data, magnitude_data, clipped_mask
        del cur_patient_data, cur_patient_data_clipped, cur_patient_data_clipped_masked, cur_patient_data_mask
        
        gc.collect()
    except Exception as e:
        print(f"An error occurred: {e} for patient folder {patient_folder}")


def main():
    output_dir_path="/MRE_pretext_data/preprocessed_pretext_data"
    os.makedirs(output_dir_path, exist_ok=True)
    data_base_path="/MRE_pretext_data/nifti_2D_dataset"
    log_path = os.path.join(output_dir_path, 'preprocessing_log.txt')
    sys.stdout = Logger(log_path)
    print("logs being saved to: ", log_path)

    manual_seed = 42
    random.seed(manual_seed)
    patient_ids = list(range(1, 160))  # 1 to 159 unique patient ids
    random.shuffle(patient_ids)

    train_patients = patient_ids[:134]
    val_patients = patient_ids[134:]
    print(train_patients)
    print(val_patients)
    print(len(train_patients))
    print(len(val_patients))

    patient_scan_data = pd.read_csv('/MRE_pretext_data/MRELiverPatientDatasets.csv') # csv that has the following columns: no. (PatientID), Code(FolderName),ScanDate
    train_folders = patient_scan_data[patient_scan_data["no."].isin(train_patients)]["Code"].tolist()
    val_folders = patient_scan_data[patient_scan_data["no."].isin(val_patients)]["Code"].tolist()
    print(len(train_folders))
    print(len(val_folders))
    patient_folders = os.listdir(data_base_path)
    print(len(patient_folders))
    random.shuffle(train_folders)
    random.shuffle(val_folders)

    with open(os.path.join("/MRE_pretext_data/train_folders_list.json"), "w") as f:
        json.dump(train_folders, f)
    with open(os.path.join("/MRE_pretext_data/val_folders_list.json"), "w") as f:
        json.dump(val_folders, f)

    for split in ["train", "valid"]:
        print(f"Currently preprocessing {split} set")
        folder_list = train_folders if split == "train" else val_folders
        for patient_idx, patient_folder in tqdm(enumerate(folder_list)):
            print(patient_idx, len(folder_list))
            output_dir_path_split = os.path.join(output_dir_path, split)
            p = Process(target=process_datasets, args=(patient_idx, patient_folder,
                                                       data_base_path, output_dir_path_split))
            p.start()
            p.join()  # Wait for patient to finish before continuing

main()