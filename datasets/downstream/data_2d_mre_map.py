import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json


class MREDataset(Dataset):
    def __init__(self, config=None, base_dir=None, flag="train", normalize_target=False, normalize_magnitude=True,
                 multiply_target_factor=None):
        super().__init__()
        self.mode = flag
        self.normalize_target = normalize_target
        self.normalize_magnitude = normalize_magnitude
        self.multiply_target_factor = multiply_target_factor
        if self.mode != "inference":
            assert base_dir is not None
            self.load_data(base_dir, self.mode)
            assert self.inputs.shape[0] == self.targets.shape[0], "Input and target size mismatch!"

        # Load or compute target normalization stats
        if self.mode == "train":
            self.target_means, self.target_stds, self.target_mins, self.target_maxs = self.compute_global_target_stats()
        else:
            self.target_means, self.target_stds, self.target_mins, self.target_maxs = None, None, None, None  # To be loaded manually later

    def load_data(self, data_path, mode):
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if "input" in file and mode in file:
                self.inputs = np.load(file_path)  # (N, H, W, C)
                # print("Inputs:", self.inputs.shape)
            elif "output" in file and mode in file:
                self.targets = np.load(file_path)  # (N, H, W, C)
                # print("Targets:", self.targets.shape)
            elif "lroi" in file and mode in file:
                self.lroi_segmentations = np.load(file_path)
            elif "croi" in file and mode in file:
                self.croi_segmentations = np.load(file_path)

    def compute_global_target_stats(self):
        """
        Compute global mean and std per target channel across the training set,
        only considering pixels where seg == 1.
        """
        means = []
        stds = []
        mins = []
        maxs = []

        for c in range(self.targets.shape[-1]):
            masked_values = []

            # for i in range(self.targets.shape[0]):
            for i in range(60):  # TODO: make (60) dynamic or a parameter (It is put here to only get the stats from original data not augmented samples)
                seg_mask = self.lroi_segmentations[i] == 1
                target_channel = self.targets[i, ..., c]
                values = target_channel[seg_mask]

                if values.size > 0:
                    masked_values.append(values)

            if masked_values:
                all_vals = np.concatenate(masked_values)
                means.append(np.mean(all_vals))
                stds.append(np.std(all_vals))
                mins.append(np.min(all_vals))
                maxs.append(np.max(all_vals))
            else:
                means.append(0.0)
                stds.append(1.0)  # fallback to avoid divide by zero
                mins.append(0.0)
                maxs.append(0.0)

        return means, stds, mins, maxs

    def __len__(self):
        return self.inputs.shape[0]

    def transform(self, x, y, seg):
        """
        Apply per-sample min-max to last input channel,
        and global z-score normalization to targets.
        Pixels outside seg==1 will be set to 0.
        """
        x = x.copy()
        y = y.copy()
        seg_mask = (seg == 1)
        if self.normalize_magnitude:
            mag_channel = x[..., -1]
            magnitude_values = mag_channel[seg == 1]
            min_val = np.min(magnitude_values)
            max_val = np.max(magnitude_values)
            # min_val = np.min(mag_channel)
            # max_val = np.max(mag_channel)

            if max_val > min_val:
                x[..., -1] = (mag_channel - min_val) / (max_val - min_val)
            else:
                x[..., -1] = 0.0  # fallback if constant
            x[..., -1][~seg_mask] = 0.0

        if self.normalize_target:
            if y is not None and seg is not None:
                # seg_mask = (seg == 1)

                for c, (mu, std) in enumerate(zip(self.target_means, self.target_stds)):
                    if std > 0:
                        y[..., c] = (y[..., c] - mu) / std
                    else:
                        y[..., c] = y[..., c] - mu

                    # Set values outside the mask to zero
                    y[..., c][~seg_mask] = 0.0

        if not self.normalize_target and self.multiply_target_factor is not None:
            y = y * 1.2
        return x, y

    def __getitem__(self, idx):
        x = self.inputs[idx]  # (H, W, C)
        y = self.targets[idx]  # (H, W, C)
        seg_mask = self.lroi_segmentations[idx]
        croi_mask = self.croi_segmentations[idx]
        x_orig = np.copy(x)
        y_orig = np.copy(y)
        x, y = self.transform(np.copy(x), np.copy(y), np.copy(seg_mask))  # TODO: remove redundent copy

        # Convert to torch tensors (C, H, W)
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()

        x_orig = torch.from_numpy(x_orig).permute(2, 0, 1).float()
        y_orig = torch.from_numpy(y_orig).permute(2, 0, 1).float()

        return x, y, x_orig, y_orig, seg_mask, croi_mask

    def denormalize_target(self, y_norm, seg_mask=None):
        """
        Per-channel denormalization for a batch of normalized targets.

        Args:
            y_norm (torch.Tensor): shape (B, C, H, W)
            seg_mask (torch.Tensor or None): optional mask of shape (B, 1, H, W) or (B, H, W),
                                            used to zero out regions outside the ROI after denormalization.

        Returns:
            torch.Tensor: denormalized targets, shape (B, C, H, W)
        """
        assert y_norm.ndim == 4
        y_denorm = y_norm.clone()
        if self.normalize_target:
            B, C, H, W = y_denorm.shape

            for c in range(C):
                mu = self.target_means[c]
                std = self.target_stds[c]
                y_denorm[:, c, :, :] = y_norm[:, c, :, :] * std + mu

            if seg_mask is not None:
                if seg_mask.ndim == 3:
                    seg_mask = seg_mask.unsqueeze(1)  # make shape (B, 1, H, W)
                y_denorm = y_denorm * seg_mask  # broadcast-safe masking

        return y_denorm

    def save_normalization_stats(self, save_path):
        stats = {
            "target_stats": [
                {"mean": float(mu), "std": float(std), "min": float(mn), "max": float(mx)}
                for mu, std, mn, mx in zip(self.target_means, self.target_stds, self.target_mins, self.target_maxs)
            ]
        }

        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def load_normalization_stats(self, json_path):
        with open(json_path, 'r') as f:
            stats = json.load(f)

        self.target_means = [d["mean"] for d in stats["target_stats"]]
        self.target_stds = [d["std"] for d in stats["target_stats"]]
        self.target_mins = [d["min"] for d in stats["target_stats"]]
        self.target_maxs = [d["max"] for d in stats["target_stats"]]
