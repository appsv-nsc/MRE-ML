"""
In order to run this code:
1- change the value of 'all_data_dict' dictionary with the paths to the checkpoints you want to evaluate,
2- choose 'mse' or 'mae' as a parameter to the 'calculate_normalized_range_error' function.
'mse' caluclates nmse and saves the results as csv files in checkpoint_path/evaluation_reports/cumulative_pixelwise/
'mae' caluclates nmae and saves the results as csv files in checkpoint_path/evaluation_reports/cumulative_pixelwise/
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def get_data_list(flag, ROI, channel,k):
    curr_data_files = [f for f in os.listdir(all_data_dict[k]) if (f"{flag}_{ROI}_PixelWiseROIData" in f)]
    all_target_data = []
    all_pred_data = []
    for slice_file in curr_data_files:
        full_slice_data_path = os.path.join(all_data_dict[k], slice_file)
        curr_slice_data = pd.read_csv(full_slice_data_path)
        curr_slice_GD_target_data = curr_slice_data[f"target_{channel}"]
        curr_slice_GD_pred_data = curr_slice_data[f"pred_{channel}"]
        if len(all_target_data) == 0:
            all_target_data = list(curr_slice_GD_target_data.copy())
            all_pred_data = list(curr_slice_GD_pred_data.copy())
            print(len(curr_slice_GD_target_data))
            print(len(all_target_data))
        else:
            all_target_data.extend(list(curr_slice_GD_target_data.copy()))
            all_pred_data.extend(list(curr_slice_GD_pred_data.copy()))
            print(len(curr_slice_GD_target_data))
            print(len(all_target_data))
    return all_target_data, all_pred_data

def calculate_normalized_range_error(mseORmaeFlag= "mse"):
    ROIs = ["Liver"]
    Channels = ["GD", "GL"]
    for roi_type in ROIs:
        for channel in Channels:
            for k, v in all_data_dict.items():
                print("Current Key", k)

                all_test_target_data, all_test_pred_data = get_data_list("Test", roi_type, channel,k)

                df = pd.DataFrame({
                    "TestPred": all_test_pred_data,
                    "TestTarget": all_test_target_data
                })

                if mseORmaeFlag == "mse":
                    df["error"] = (df["TestPred"] - df["TestTarget"]) ** 2
                elif mseORmaeFlag == "mae":
                    df["error"] = (df["TestPred"] - df["TestTarget"]).abs()

                # Define bins
                bins = list(np.arange(0, 13.1, 0.5)) + [np.inf]
                labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 2)] + [">13"]

                # Bin by target
                df["target_bin"] = pd.cut(
                    df["TestTarget"],
                    bins=bins,
                    labels=labels,
                    right=False
                )

                # Average error per bin
                bin_means = df.groupby("target_bin")["error"].mean()

                # Global average error
                global_from_bins = bin_means.mean()

                # Create output dataframe (ranges as columns, values as one row)
                output = pd.DataFrame([bin_means.tolist() + [global_from_bins]],
                                      columns=list(bin_means.index) + ["global"])

                # Save to CSV
                new_save_dir = all_data_dict[k].replace("roi_pixel_data", "cumulative_pixelwise")
                os.makedirs(new_save_dir, exist_ok=True)
                output.to_csv(os.path.join(new_save_dir,
                                           f"error_by_range_{mseORmaeFlag}_pixelwise_{roi_type}_{channel}.csv"),
                              index=False)


# dictionary containing checkpoint paths for models you want to evaluate
all_data_dict = {  "scratch-unet-mse":"checkpoint_full_path/evaluation_reports/roi_pixel_data",
                   "scratch-unet-TargetWeightedMse": "checkpoint_full_path/evaluation_reports/roi_pixel_data"}

calculate_normalized_range_error(mseORmaeFlag= "mse")

