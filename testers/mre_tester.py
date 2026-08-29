import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
import numpy as np

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(grandparent_dir)

from datasets.downstream.data_2d_mre_map import MREDataset
from networks import get_unet_model, init_weights
from utils.evaluation import save_roi_pixel_preds
from utils.generic_helpers import init_random_and_cudnn, Logger

def get_preds(device, model, curr_dataset, curr_dataloader):
    all_preds_norm = []
    all_targets_norm = []

    all_preds_denorm = []
    all_targets_denorm = []
    all_vals = []
    all_masks = []
    all_croi_masks = []
    with torch.no_grad():
        for i, (inputs_norm, targets_norm,x_orig,y_orig, seg_mask, croi_mask) in enumerate(curr_dataloader):
            inputs_norm = inputs_norm.to(device)
            pred_norm = model(inputs_norm)
            criterion = nn.MSELoss(reduction='mean')
            targets_norm = targets_norm.to(device)
            val_loss = criterion(pred_norm, targets_norm)
            all_vals.append(val_loss.item())
            pred_norm = pred_norm.cpu()
            targets_norm = targets_norm.cpu()

            # Denormalize predictions and targets
            pred_denorm = curr_dataset.denormalize_target(pred_norm, seg_mask)
            target_denorm = curr_dataset.denormalize_target(targets_norm, seg_mask)

            all_preds_norm.append(pred_norm)
            all_targets_norm.append(targets_norm)

            all_preds_denorm.append(pred_denorm)
            all_targets_denorm.append(target_denorm)
            seg_mask = seg_mask.unsqueeze(1)
            
            all_masks.append(seg_mask)

            croi_mask = croi_mask.unsqueeze(1)
            all_croi_masks.append(croi_mask)

    print("mean val mse", np.mean(all_vals))

    all_preds_norm = torch.cat(all_preds_norm)
    all_targets_norm = torch.cat(all_targets_norm)

    all_preds_denorm = torch.cat(all_preds_denorm)
    all_targets_denorm = torch.cat(all_targets_denorm)
    all_masks = torch.cat(all_masks)
    all_croi_masks = torch.cat(all_croi_masks)
    print(f"Targets Denorm shape: {all_targets_denorm.shape}")
    print(f"Masks shape: {all_masks.shape}")

    print(f"Predictions shape: {all_preds_norm.shape}")
    print(f"Ground truth shape: {all_targets_norm.shape}")

    return all_preds_norm, all_targets_norm, all_preds_denorm, all_targets_denorm, all_masks, all_croi_masks

def run_model_inference(model_variant, data_path, device, batch_size, model_path, save_dir):
    print(f"Currenty running inference for model variant {model_variant}")

    # === Load Model ===
    data_mode = (model_variant.split('data_')[-1]).split('_network')[0]
    network_name = "mapping_"+model_variant
    model = get_unet_model(network_name)
    init_weights(model,"kaiming")
    model = model.cuda()
    checkpoint = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("Successfully loaded model weights from ",model_path)
    model.to(device)
    model.eval()

    # === Load Data ===
    data_path = data_path
    stats_path = os.path.join(save_dir, "train_stats.json")
    
    test_dataset = MREDataset(base_dir=data_path, flag="test")
    test_dataset.load_normalization_stats(stats_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = MREDataset(base_dir=data_path, flag="valid")
    val_dataset.load_normalization_stats(stats_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = MREDataset(base_dir=data_path, flag="train")
    train_dataset.load_normalization_stats(stats_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    all_preds_norm_test, all_targets_norm_test, all_preds_denorm_test, all_targets_denorm_test, all_masks_test, all_croi_masks_test = get_preds(device, model, test_dataset, test_loader)
    all_preds_norm_val, all_targets_norm_val, all_preds_denorm_val, all_targets_denorm_val, all_masks_val, all_croi_masks_val =get_preds(device, model, val_dataset, val_loader)
    all_preds_norm_train, all_targets_norm_train, all_preds_denorm_train, all_targets_denorm_train, all_masks_train, all_croi_masks_train =get_preds(device, model, train_dataset, train_loader)
    
    res_save_path = Path(save_dir, 'inference_results')
    res_save_path.mkdir(parents=True, exist_ok=True)

    np.save(res_save_path / 'test_all_preds.npy', all_preds_denorm_test)
    np.save(res_save_path / 'valid_all_preds.npy', all_preds_denorm_val)
    np.save(res_save_path / 'train_all_preds.npy', all_preds_denorm_train)

    Path(save_dir, 'evaluation_reports').mkdir(parents=True, exist_ok=True)
    save_dir = Path(save_dir, 'evaluation_reports')
    ROI_List = ["Liver"]
    #all_targets_denorm_test_copy = all_targets_denorm_test.clone()
    for roi in ROI_List:
        save_roi_pixel_preds(test_targets=all_targets_denorm_test,
                            test_preds=all_preds_denorm_test,
                            test_masks=all_masks_test,mask_mode=roi,
                            save_dir=save_dir,flag="Test")


def main():
    # === Argument Parser ===
    parser = argparse.ArgumentParser(description="Run inference using trained model")
    parser.add_argument("--model_variant", type=str,
                        default="unet_2d",
                        help="Model name")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--data_path", type=str, default='E:/MREData/augmented_preprocessed_data',
                        help="models_path")
    parser.add_argument("--models_path", type=str, default='/data/projects/mre-ml/checkpoints/full_pipeline/20260328/downstream_task_mre/mapping',
                        help="models_path")
    parser.add_argument("--note", type=str, default='_autoencoder',
                        help="should be _autoencoder if AEpretrained, _rot if rot pretrained or _scrach if rand init.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = True if torch.cuda.is_available() else False
    init_random_and_cudnn(manualseed=123, use_gpu=use_gpu)
    log_path = os.path.join(args.models_path, "test_log.txt")
    sys.stdout = Logger(log_path)
    
    parent_dir = args.models_path
    #parent_dir = Path(__file__).resolve().parent
    full_cur_model_path = os.path.join(parent_dir, args.model_variant+args.note)
    model_path = os.path.join(full_cur_model_path,"best_model.pth")
    save_dir = full_cur_model_path
    run_model_inference(model_variant=args.model_variant,
                        data_path=args.data_path,
                        device=device, batch_size=args.batch_size,
                        model_path=model_path,
                        save_dir=save_dir)


main()