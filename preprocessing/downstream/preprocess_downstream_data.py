"""
This script is a standalone script that can be used to prepare the mre
downstream data for GD/GL prediction

This data should have 5 channels in the input representing:
0- phase data of timeframe 1
1- phase data of timeframe 2
2- phase data of timeframe 3
3- phase data of timeframe 4
4- magnitude timeframe 1


In this case, the data shapes would be:
input shape: nSlices, 256, 256, 5
target shape: nSlices, 256, 256, 2

The two channels in target are for GD and GL

Takes:
A-input data path: .npy path of shape (74,256,256,5) where there are redundent channels
B- segmentations path: folder path containing 74 .npy files, each contains the segmentation array for the corresponding slice
C- groundtruth path: path to single .npy file of shape (74,256,256,5) where the 5 are:
   0: phase_velocity
   1: attenuation
   2: G_elastic
   3: Gd
   4: Gl

D-output path: the output of running this code will be saved in this path. This will include:
1- 'preprocessing_visualizations' folder where segmetnations overlay and other visualizations are saved if needed
if train:
    2- 'input_train60_images_liver_clipped_masked.npy': train input array of shape 60,256,256,5
    3- 'input_val14_images_liver_clipped_masked.npy': val input array of shape 14,256,256,5
    4- 'output_train60_images_liver_clipped_masked.npy': train input array of shape 60,256,256,2
    5- 'output_val14_images_liver_clipped_masked.npy': val input array of shape 14,256,256,2
    6- 'seg_train60_images_liver_clipped_masked.npy': the segmentations after clipping to be used by ROI Norm and weighted mse in the downstream dataloader code
    7- 'seg_val14_images_liver_clipped_masked.npy': the segmentations after clipping to be used by ROI Norm and weighted mse in the downstream dataloader code
if test:
    2- 'input_test_images_liver_clipped_masked.npy': train input array of shape 26,256,256,5
    3- 'output_test_images_liver_clipped_masked.npy': train input array of shape 26,256,256,2
    4- 'seg_test_images_liver_clipped_masked.npy': the segmentations after clipping to be used by ROI Norm and weighted mse in the downstream dataloader code
"""
import os
import numpy as np
from downstream_helpers import *
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt


def process_clipped_masked(input_dataset_path, groundtruth, seg_class
                           , output_path, seg_path, croi_seg_path
                           , flag, apply_erosion):
    save_visualizations_path = os.path.join(output_path, f"preprocessing_visualizations/{flag}")
    os.makedirs(save_visualizations_path, exist_ok=True)

    input_data_array = np.load(input_dataset_path)
    groundtruth_array = np.load(groundtruth)
    croi_array = get_nifti_data(croi_seg_path)
    croi_array = np.transpose(croi_array, (2,0,1))
    print(input_data_array.shape)
    print(groundtruth_array.shape)
    print(croi_array.shape)
    segmentations = np.zeros((input_data_array.shape[0], 320, 320))
    liver_and_vessels_segmentations = np.zeros((input_data_array.shape[0], 320, 320))

    for seg_file in os.listdir(seg_path):
        indx = int(seg_file.split('_')[1].split('.')[0])
        seg_file_path = os.path.join(seg_path, seg_file)

        print(seg_file_path, indx)
        seg = np.load(seg_file_path)
        current_croi_mask = croi_array[indx,:,:].copy()
        unique_values = np.unique(seg)
        print("Unique values in mask:", unique_values)
        colors = ['black', 'green', 'red']
        cmap = ListedColormap(colors[:len(unique_values)])
        plt.imshow(seg, cmap=cmap)
        plt.axis('off')
        plt.savefig(save_visualizations_path + "/" + str(indx) + "_LiverVesselsMask.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()
        # plt.close()

        # whether it is liver class or liver vessel class turn it to 1 to get the entire liver ROI
        liver_roi_mask = (seg == 1).astype(np.uint8)
        plt.imshow(liver_roi_mask, cmap='gray')
        plt.axis('off')
        plt.savefig(save_visualizations_path + "/" + str(indx) + "_LiverWithoutVesselsMask.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        current_croi_mask = (current_croi_mask > 0).astype(np.uint8)
        #print(np.unique(banana_roi_array))
        liver_and_vessels_roi_mask = (seg > 0).astype(np.uint8)
        unique_values = np.unique(liver_roi_mask)
        print("Unique values in mask:", unique_values)

        cmapBinaryGreen = ListedColormap([
                (0, 0, 0, 0),      # fully transparent
                (0, 1, 0, 0.3)     # green with alpha
                ])
        cmapBinaryRed = ListedColormap([
                (0, 0, 0, 0),      # fully transparent
                (1, 0, 0, 0.7)     # red with alpha
                ])
        plt.imshow(input_data_array[indx, :, :,-1], cmap='gray')
        plt.axis('off')
        plt.savefig(save_visualizations_path + "/" + str(indx) + "_Magnitude.png", bbox_inches='tight', pad_inches=0)
        # liver_roi_mask = np.squeeze(liver_roi_mask,axis=-1)
        # liver_roi_mask = np.transpose(liver_roi_mask,(1,0))
        current_croi_mask = np.rot90(current_croi_mask, k=1)
        current_croi_mask = np.fliplr(current_croi_mask)
        current_croi_mask = liver_roi_mask * current_croi_mask # this is correct but does not account for erosion removing part of the liver so this was handled in the mre tester code for the banana ROI
        plt.imshow(current_croi_mask, cmap=cmapBinaryRed)
        plt.imshow(liver_roi_mask, cmap=cmapBinaryGreen)
        plt.savefig(save_visualizations_path + "/" + str(indx) + "_BananaLiverSegOverlay.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.show()

        plt.imshow(input_data_array[indx, :, :, -1], cmap='gray')
        plt.imshow(liver_roi_mask, cmap=cmapBinaryGreen)
        plt.axis('off')
        plt.savefig(save_visualizations_path + "/" + str(indx) + "_SegOverlay.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        # liver_roi_mask = np.squeeze(liver_roi_mask,axis=-1)
        segmentations[indx, :, :] = liver_roi_mask
        croi_array[indx,:,:] = current_croi_mask
        liver_and_vessels_segmentations[indx, :, :] = liver_and_vessels_roi_mask



    input_data_clipped = np.zeros((input_data_array.shape[0], 256, 256, input_data_array.shape[-1]))
    input_data_clipped_masked = np.zeros((input_data_array.shape[0], 256, 256, input_data_array.shape[-1]))

    groundtruth_clipped = np.zeros((groundtruth_array.shape[0], 256, 256, groundtruth_array.shape[-1]))
    groundtruth_masked = np.zeros((groundtruth_array.shape[0], 256, 256, groundtruth_array.shape[-1]))
    nMaskPixels = []
    clipped_lroi_segmentations = np.zeros((segmentations.shape[0], 256, 256))
    clipped_croi_segmentations = np.zeros((segmentations.shape[0], 256, 256))
    for sliceIdx in range(input_data_array.shape[0]):
        print("processing slice: " + str(sliceIdx))
        current_segmentation = segmentations[sliceIdx, :, :]
        #current_slice_liver_vessel_segmentation = liver_and_vessels_segmentations[sliceIdx, :, :]

        cy, cx, status = get_mask_center(current_segmentation, save_visualizations_path, sliceIdx)
        print("centroid for " + str(sliceIdx) + " slice number " + str(sliceIdx) + " is: (" + str(cy) + "," + str(
            cx) + ")")
        top, bottom, left, right = compute_clip_boundaries(cy, cx, 256, 320)

        clipped_mask = current_segmentation[top:bottom, left:right].copy()
        clipped_croi_segmentations[sliceIdx, :, :] = croi_array[sliceIdx,top:bottom, left:right].copy()
        clipped_lroi_segmentations[sliceIdx, :, :] = clipped_mask

        if apply_erosion:
            structure = np.ones((5, 5))

            eroded_mask = binary_erosion(clipped_mask, structure=structure)

            fig, axes = plt.subplots(1, 2, figsize=(18, 10))

            # Row 1
            # 1. Original Target
            im0 = axes[0].imshow(clipped_mask, cmap='gray')
            axes[0].set_title('Original Mask')
            axes[0].axis('off')

            # 2. Original Mask
            axes[1].imshow(eroded_mask, cmap='gray')
            axes[1].set_title('Eroded Mask')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_visualizations_path, f"slice_{sliceIdx}_erosion.png"))
            plt.close()

            clipped_mask = eroded_mask
            clipped_lroi_segmentations[sliceIdx, :, :] = clipped_mask
            plt.imshow(clipped_mask, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_visualizations_path, f"{sliceIdx}_erosion.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        # to calculate percentage of mask to overall image
        nMaskPixels.append(np.count_nonzero(clipped_mask) / clipped_mask.size)
        print(f"count mask (non-zero) pixels in slice idx {sliceIdx}: ", np.count_nonzero(clipped_mask))
        print(f"count total pixels in slice idx {sliceIdx}: ", clipped_mask.size)
        print(f"mask percentage in slice idx {sliceIdx}: ", np.count_nonzero(clipped_mask) / clipped_mask.size)

        for datatypeIdx in range(input_data_array.shape[-1]):

            plt.imshow(input_data_array[sliceIdx, :, :, datatypeIdx], cmap="gray")
            plt.axis('off')
            plt.savefig(save_visualizations_path + "/" + str(sliceIdx) + f"_Channel{datatypeIdx}_Original.png", bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            input_data_clipped[sliceIdx, :, :, datatypeIdx] = input_data_array[sliceIdx, top:bottom,
                                                              left:right, datatypeIdx]

            plt.imshow(input_data_clipped[sliceIdx, :, :, datatypeIdx], cmap="gray")
            plt.axis('off')
            plt.savefig(save_visualizations_path + "/" + str(sliceIdx) + f"_Channel{datatypeIdx}_ClippedAroundCenterOfMass.png", bbox_inches='tight',
                        pad_inches=0)
            plt.close()
            input_data_clipped_masked[sliceIdx, :, :, datatypeIdx] = np.where(clipped_mask == 1,
                                                                              input_data_clipped[sliceIdx, :, :,
                                                                              datatypeIdx], 0)

                #plt.imshow(input_data_clipped_masked[sliceIdx, :, :, datatypeIdx], cmap='twilight', vmin=-np.pi, vmax=np.pi)
            plt.imshow(input_data_clipped_masked[sliceIdx, :, :, datatypeIdx], cmap='gray')
            plt.axis('off')
            plt.savefig(save_visualizations_path + "/" + str(sliceIdx) + f"_channel{datatypeIdx}.png", bbox_inches='tight',
                        pad_inches=0)
            plt.close()

        for outputIdx in range(groundtruth_array.shape[-1]):
            current_slice = groundtruth_array[sliceIdx, :, :, outputIdx].copy()
            current_slice_before_transform = groundtruth_array[sliceIdx, :, :, outputIdx].copy()
            groundtruth_array[sliceIdx, :, :, outputIdx] = current_slice

            if outputIdx in [groundtruth_array.shape[-1] - 1, groundtruth_array.shape[-1] - 2]:
                if outputIdx == groundtruth_array.shape[-1] - 2:
                    channel = "GD"
                else:
                    channel = "GL"
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                # Original
                axs[0].imshow(current_slice_before_transform, cmap='viridis')
                axs[0].set_title('Original')
                axs[0].axis('off')

                # After Gaussian filter
                axs[1].imshow(groundtruth_array[sliceIdx, :, :, outputIdx], vmin=0, vmax=8, cmap='nipy_spectral')
                axs[1].set_title('After Filter(s)')
                axs[1].axis('off')

                # Save the figure to a file
                plt.tight_layout()
                tag = ""
                if apply_erosion:
                    tag = tag + "_eroded"
                plt.savefig(os.path.join(save_visualizations_path, f"{flag}_slice_{sliceIdx}_{channel}{tag}"), dpi=300)
                plt.close()
                # plt.show()
                print(np.array_equal(current_slice_before_transform, groundtruth_array[sliceIdx, :, :, outputIdx]))

            groundtruth_clipped[sliceIdx, :, :, outputIdx] = groundtruth_array[sliceIdx, top:bottom,
                                                             left:right, outputIdx]

            groundtruth_masked[sliceIdx, :, :, outputIdx] = np.where(clipped_mask == 1,
                                                                     groundtruth_clipped[sliceIdx, :, :, outputIdx], 0)

            plt.imshow(groundtruth_masked[sliceIdx, :, :, outputIdx], vmin=0, vmax=8, cmap='nipy_spectral')
            plt.axis('off')
            plt.savefig(save_visualizations_path + "/" + str(sliceIdx) + f"_OutputChannel{outputIdx}.png", bbox_inches='tight',
                        pad_inches=0)
            plt.close()
    print("mean n Pixels in mask", np.mean(nMaskPixels))
    print("currently saving processed data....")

    if flag == "train":
        np.save(os.path.join(output_path,
                             "input_train.npy"),
                input_data_clipped_masked[:60, :, :, :])
        np.save(
            os.path.join(output_path,
                         "input_valid.npy"),
            input_data_clipped_masked[60:, :, :, :])

        np.save(
            os.path.join(output_path, "seg_train.npy"),
            clipped_lroi_segmentations[:60, :, :])
        np.save(
            os.path.join(output_path, "seg_valid.npy"),
            clipped_lroi_segmentations[60:, :, :])

        np.save(
            os.path.join(output_path, "croi_train.npy"),
            clipped_croi_segmentations[:60, :, :])
        np.save(
            os.path.join(output_path, "croi_valid.npy"),
            clipped_croi_segmentations[60:, :, :])

        np.save(os.path.join(output_path,
                             "output_train.npy"),
                groundtruth_masked[:60, :, :, -2:])
        np.save(os.path.join(output_path,
                             "output_valid.npy"),
                groundtruth_masked[60:, :, :, -2:])


    else:
        np.save(
            os.path.join(output_path, "input_test.npy"),
            input_data_clipped_masked)

        np.save(
            os.path.join(output_path, "seg_test.npy"),
            clipped_lroi_segmentations)

        np.save(
            os.path.join(output_path, "croi_test.npy"),
            clipped_croi_segmentations)

        np.save(
            os.path.join(output_path, "output_test.npy"),
            groundtruth_masked[:, :, :, -2:])

    print("finished saving processed data....")


def main(flag, apply_erosion=False):
    output_dir = "E:/MREData/preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{flag}_log_all_frames.txt')
    sys.stdout = Logger(log_path)

    train_args = {
        "dataset_path": "E:/MREData/input_images_all_timeframes.npy",
        "segmentations_path": "E:/MREData/train_all_masks",
        "groundtruth_path": "E:/MREData/output_train_images_all_timeframes.npy",
        "Banana_ROI_path": "E:/MREData/train_all_mask_CRoI_masks.nii.gz"}

    test_args = {
        "dataset_path": "E:/MREData/input_images_all_timeframes.npy",
        "segmentations_path": "E:/MREData/test_all_masks",
        "groundtruth_path": "E:/MREData/output_test_images_all_timeframes.npy",
        "Banana_ROI_path": "E:/MREData/test_all_mask_CRoI_masks.nii.gz"}

    print("-------- processing all time frames format (:,256,256,5) -------------")

    if flag == "train":
        process_clipped_masked(input_dataset_path=train_args["dataset_path"],
                               groundtruth=train_args["groundtruth_path"], seg_class="liver",
                               output_path=output_dir, seg_path=train_args["segmentations_path"], croi_seg_path=train_args["Banana_ROI_path"], flag=flag,
                               apply_erosion=apply_erosion)
    else:
        process_clipped_masked(input_dataset_path=test_args["dataset_path"], groundtruth=test_args["groundtruth_path"],
                               seg_class="liver",
                               output_path=output_dir, seg_path=test_args["segmentations_path"], croi_seg_path=test_args["Banana_ROI_path"], flag=flag,
                               apply_erosion=apply_erosion)


main(flag="train", apply_erosion=True)
