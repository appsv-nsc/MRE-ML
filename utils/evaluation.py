import torch
import numpy as np
import os
import csv

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def extract_pixels(pred, target, mask):
    """Extract pixels from within ROI for all samples and return per channel."""
    pred_np = to_numpy(pred)
    target_np = to_numpy(target)
    mask_np = to_numpy(mask).astype(bool)

    B, C, H, W = target_np.shape
    
    target_pixels = {} 
    pred_pixels = {}
    for i in range(B):  # per sample
        target_pixels[i] = {}
        pred_pixels[i] = {}
        roi_mask = mask_np[i, 0]  # shape: (256, 256)
        
        for ch in range(C):  # per channel
            t_vals = target_np[i, ch][roi_mask]
            p_vals = pred_np[i, ch][roi_mask]

            sorted_indices = np.argsort(t_vals)
            sorted_t = t_vals[sorted_indices]
            sorted_p = p_vals[sorted_indices]

            target_pixels[i][ch] = sorted_t
            pred_pixels[i][ch] = sorted_p

    return target_pixels, pred_pixels   # target_pixels, pred_pixels dictionaries like this {0:{0:[20,12,14],1:[],2:[]},1:}

def export_pixel_csvs(target_pixel_data: dict, pred_pixel_data: dict, out_path: str = ""):
    for key, target_slice_dict in target_pixel_data.items():
        pred_slice_dict = pred_pixel_data[key]

        # Get the four lists
        slice_GD_target = target_slice_dict[0]
        slice_GL_target = target_slice_dict[1]
        slice_GD_pred = pred_slice_dict[0]
        slice_GL_pred = pred_slice_dict[1]

        # Make sure they are the same length (zip will stop at shortest anyway)
        rows = zip(slice_GD_target, slice_GL_target, slice_GD_pred, slice_GL_pred)

        # File name per outer key
        filename = f"{out_path}_slice_{key}.csv"

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            # header row
            writer.writerow(["target_GD", "target_GL", "pred_GD", "pred_GL"])
            # Write all rows
            writer.writerows(rows)

def save_roi_pixel_preds(test_targets, test_preds, test_masks,
                         mask_mode, save_dir, flag):
    # Extract Liver ROI Q2 pixels from both sets
    if mask_mode=="Liver":
        print(f"Extracting pixels from test-{flag} samples")
        test_q2_targets_avg, test_q2_preds_avg,test_q2_target_pixels, test_q2_pred_pixels = extract_pixels(test_preds, test_targets, test_masks)

    pixel_data_dir = os.path.join(save_dir, "roi_pixel_data")
    os.makedirs(pixel_data_dir, exist_ok=True)
    export_pixel_csvs(target_pixel_data=test_q2_target_pixels
                      ,pred_pixel_data=test_q2_pred_pixels
                      ,out_path=(os.path.join(pixel_data_dir,f"{flag}_{mask_mode}_PixelWiseROIData")))