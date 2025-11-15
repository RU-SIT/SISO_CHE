#!/usr/bin/env python3
"""
MAML Fine-tuning for PCGrad-Trained Models

This script fine-tunes a PCGrad-trained MAML model on unseen channels.
During fine-tuning, we use STANDARD gradient updates (no PCGrad) because:
- We're adapting to ONE channel at a time (single task)
- No gradient conflicts exist in single-task setting
- PCGrad only helps when multiple tasks are trained simultaneously

Usage: Load PCGrad-trained checkpoint → Fine-tune on new channel → Evaluate
"""

import os
import torch
import numpy as np
import pandas as pd
from utils import Utils
import argparse
from meta import Meta
import pdb


# ---------------- helpers ----------------
def _scale_with_params(x, params, eps=1e-8):
    """
    Apply [-1,1] scaling using provided per-channel params from Utils.standard_scaling.
    x: np.ndarray (..., 2) channel-last
    params: dict with keys {min_real, max_real, min_imag, max_imag}
    """
    real = 2.0 * (x[..., 0] - params["min_real"]) / (params["max_real"] - params["min_real"] + eps) - 1.0
    imag = 2.0 * (x[..., 1] - params["min_imag"]) / (params["max_imag"] - params["min_imag"] + eps) - 1.0
    return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)


def _to_torch_ch_first(x_np, device):
    """
    x_np: [N, H, W, 2] -> torch [N, 2, H, W]
    """
    x_cf = np.transpose(x_np, (0, 3, 1, 2))
    return torch.from_numpy(x_cf).to(device)


def _load_minmax_params(scaler_dir):
    """
    Load global min/max scaling parameters saved by ChannelNet.
    """
    params_path = os.path.join(scaler_dir, "minmax_params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Cannot find minmax_params.npz in {scaler_dir}. "
            "Ensure ChannelNet training has saved the scaling parameters."
        )
    data = np.load(params_path)
    x_params = {
        "min_real": data["x_min_real"],
        "max_real": data["x_max_real"],
        "min_imag": data["x_min_imag"],
        "max_imag": data["x_max_imag"],
    }
    y_params = {
        "min_real": data["y_min_real"],
        "max_real": data["y_max_real"],
        "min_imag": data["y_min_imag"],
        "max_imag": data["y_max_imag"],
    }
    return x_params, y_params
# -----------------------------------------


def _find_best_checkpoint(ckpt_dir, k_qry, k_spt, meta_lr, update_lr, use_pcgrad=True):
    """
    Locate the best checkpoint saved during training.
    Supports both PCGrad and standard MAML checkpoint naming.
    
    Args:
        ckpt_dir: Directory containing checkpoints
        k_qry: Number of query samples
        k_spt: Number of support samples
        meta_lr: Meta learning rate
        update_lr: Inner loop learning rate
        use_pcgrad: If True, look for PCGrad checkpoints; else standard MAML
        
    Returns:
        (path, step) if found, else (None, None)
    """
    if not os.path.isdir(ckpt_dir):
        return None, None
    
    if use_pcgrad:
        # PCGrad checkpoint naming: MAML_PCGrad_5_shot_5_query_BEST_step_X_MetaLr5e-4_TaskLr1e-4.pth.tar
        prefix = f"MAML_PCGrad_{k_qry}_shot_{k_spt}_query_BEST_step_"
        suffix = f"_MetaLr{meta_lr}_TaskLr{update_lr}.pth.tar"
    else:
        # Standard MAML checkpoint naming: MAML_5_shot_5_query_BEST_checkpoint_step_X_MetaLr5e-4_TaskLr1e-4.pth.tar
        prefix = f"MAML_{k_qry}_shot_{k_spt}_query_BEST_checkpoint_step_"
        suffix = f"_MetaLr{meta_lr}_TaskLr{update_lr}.pth.tar"
    
    best_path = None
    best_step = -1
    
    for fname in os.listdir(ckpt_dir):
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue
        try:
            # Extract step number from filename
            step_part = fname[len(prefix):]
            step_str = step_part.split("_")[0]
            step_val = int(step_str)
        except (ValueError, IndexError):
            continue
        
        if step_val > best_step:
            best_step = step_val
            best_path = os.path.join(ckpt_dir, fname)
    
    return best_path, best_step


def fine_tune(args):
    """
    Fine-tune a PCGrad-trained MAML model on unseen channels.
    
    Key points:
    - Loads PCGrad-trained checkpoint (meta-trained on multiple tasks)
    - Fine-tunes on SINGLE unseen channel at a time
    - Uses STANDARD gradient descent (no PCGrad needed)
    - Evaluates on held-out test set
    """
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print("="*80)
    print("MAML FINE-TUNING (from PCGrad-trained model)")
    print("="*80)
    print(args)
    print()

    # Define the model configuration (must match training)
    config = [
        ('conv2d', [32, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [32]),
        ('conv2d', [128, 32, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [128]),
        ('conv2d', [256, 128, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [128, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [128]),
        ('conv2d', [32, 128, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [32]),
        ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [args.batchsz]),
        ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    ]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    maml_finetuning = Meta(args, config).to(device)

    # Load data
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    # Fine-tune on test channels (unseen during meta-training)
    all_channels = list(data_dict.keys())
    
    # Determine split based on dataset type
    if 'TDL' in args.root:
        # TDL: first 10 are training channels, rest are test
        num_train = 10
    elif 'UMi' in args.root:
        # UMi: first 4 are training channels, rest are test
        num_train = 4
    else:
        # Default: use last 20% as test channels
        num_train = int(0.8 * len(all_channels))
    
    fine_tune_file_names = all_channels[num_train:]
    
    print(f"Total channels: {len(all_channels)}")
    print(f"Training channels: {num_train}")
    print(f"Fine-tuning channels (unseen): {len(fine_tune_file_names)}")
    print(f"Channels to fine-tune: {fine_tune_file_names}")
    print()

    # Load scaling parameters
    x_params, y_params = _load_minmax_params(args.scaler_dir)

    # Load PCGrad-trained checkpoint
    ckpt_dir = os.path.join(args.save_init, f"meta_pcgrad_model_nway_{args.n_way}")
    
    print(f"Looking for checkpoint in: {ckpt_dir}")
    best_ckpt_path, best_step = _find_best_checkpoint(
        ckpt_dir, args.k_qry, args.k_spt, args.meta_lr, args.update_lr, use_pcgrad=True
    )
    
    if best_ckpt_path is not None:
        print(f"✓ Found PCGrad checkpoint: {os.path.basename(best_ckpt_path)} (step {best_step})")
    else:
        print("✗ No PCGrad checkpoint found. Trying fallback...")
        # Try standard MAML checkpoint as fallback
        ckpt_dir_fallback = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}")
        best_ckpt_path, best_step = _find_best_checkpoint(
            ckpt_dir_fallback, args.k_qry, args.k_spt, args.meta_lr, args.update_lr, use_pcgrad=False
        )
        if best_ckpt_path is not None:
            print(f"✓ Found standard MAML checkpoint: {os.path.basename(best_ckpt_path)} (step {best_step})")
            ckpt_dir = ckpt_dir_fallback
        else:
            raise FileNotFoundError(
                f"No checkpoint found in either:\n"
                f"  - {ckpt_dir}\n"
                f"  - {ckpt_dir_fallback}\n"
                f"Please train the model first using MAML_PCGrad_trainer.py or MAML_trainer_with_tracking.py"
            )
    
    print()

    # Track MSE for all channels
    all_mse_results = []
    
    # Fine-tune on each unseen channel
    for idx, outer_channel_name in enumerate(fine_tune_file_names):
        print("="*80)
        print(f"Fine-tuning [{idx+1}/{len(fine_tune_file_names)}]: {outer_channel_name}")
        print("="*80)

        # ---- Load channel data ----
        x_all = data_dict[outer_channel_name].astype(np.float32)   # [N, H, W, 2]
        y_all = labels_dict[outer_channel_name].astype(np.float32) # [N, H, W, 2]
        
        print(f"  Data shape: {x_all.shape}")

        # ---- Apply scaling using shared parameters ----
        x_all_s = _scale_with_params(x_all, x_params)
        y_all_s = _scale_with_params(y_all, y_params)

        # ---- Split into k-shot pool and eval (no leakage) ----
        x_pool_s, y_pool_s = x_all_s[:30], y_all_s[:30]
        x_eval_s, y_eval_s = x_all_s[30:], y_all_s[30:]
        
        print(f"  k-shot pool size: {x_pool_s.shape[0]}")
        print(f"  Evaluation set size: {x_eval_s.shape[0]}")

        # ---- k-shot subset from the (scaled) pool ----
        x_shot_s, y_shot_s = x_pool_s[:args.k_qry], y_pool_s[:args.k_qry]
        
        print(f"  Using {args.k_qry}-shot for fine-tuning")

        # ---- Reload meta-trained weights (fresh start for each channel) ----
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        maml_finetuning.load_state_dict(checkpoint['state_dict'])
        print(f"  ✓ Loaded meta-trained weights")

        # ---- To torch (channel-first) ----
        x_shot_t = _to_torch_ch_first(x_shot_s, device)
        y_shot_t = _to_torch_ch_first(y_shot_s, device)
        x_eval_t = _to_torch_ch_first(x_eval_s, device)
        y_eval_t = _to_torch_ch_first(y_eval_s, device)
        
        # ---- Fine-tune on scaled k-shot (STANDARD gradient descent, no PCGrad) ----
        print(f"  Starting fine-tuning (epochs={args.epoch}, lr={args.update_lr})...")
        optimizer = torch.optim.AdamW(maml_finetuning.parameters(), lr=args.update_lr, weight_decay=0.01)
        loss_path = os.path.join(ckpt_dir, f"MAML_PCGrad_{args.k_qry}_shot_{outer_channel_name}_loss.npy")
        
        maml_finetuning.finetuning(optimizer, x_shot_t, y_shot_t, args.epoch, args.batchsz, device, loss_path)
        print(f"  ✓ Fine-tuning complete")

        # Save the fine-tuned model
        os.makedirs(ckpt_dir, exist_ok=True)
        finetuned_model_path = os.path.join(
            ckpt_dir,
            f"MAML_PCGrad_{args.k_qry}_shot_fine_tuned_{outer_channel_name}_lr{args.update_lr}.pth"
        )
        torch.save({'state_dict': maml_finetuning.state_dict()}, finetuned_model_path)
        print(f"  ✓ Saved fine-tuned model: {os.path.basename(finetuned_model_path)}")

        # ---- Evaluate on scaled eval split ----
        eval_save_path_scaled = os.path.join(
            ckpt_dir,
            f"MAML_PCGrad_{args.k_qry}_shot_{outer_channel_name}_predictions_scaled.npy"
        )
        eval_loss, preds_scaled_cf = maml_finetuning.evaluate(
            x_eval_t, y_eval_t, args.batchsz, device, save_path=eval_save_path_scaled
        )
        print(f"  ✓ Evaluation loss (scaled): {eval_loss:.6f}")

        # ---- Unscale predictions to original range for saving/metrics ----
        # preds_scaled_cf is [N, 2, H, W] -> channel-last for unscale
        preds_scaled_cl = np.transpose(preds_scaled_cf, (0, 2, 3, 1))  # [N, H, W, 2]
        preds_unscaled = Utils.unscale_standard(preds_scaled_cl, y_params)  # back to original scale

        eval_save_path_unscaled = os.path.join(
            ckpt_dir,
            f"MAML_PCGrad_{args.k_qry}_shot_{outer_channel_name}_predictions.npy"
        )
        np.save(eval_save_path_unscaled, preds_unscaled)
        print(f"  ✓ Predictions saved: {os.path.basename(eval_save_path_unscaled)}")
        
        # ---- Calculate MSE in original (unscaled) space ----
        # Unscale ground truth labels for comparison
        y_eval_cl = np.transpose(y_eval_t.cpu().numpy(), (0, 2, 3, 1))  # [N, H, W, 2]
        y_eval_unscaled = Utils.unscale_standard(y_eval_cl, y_params)
        
        # Compute MSE in original space
        mse_unscaled = np.mean((preds_unscaled - y_eval_unscaled) ** 2)
        
        # Compute MSE per channel (real and imaginary)
        mse_real = np.mean((preds_unscaled[..., 0] - y_eval_unscaled[..., 0]) ** 2)
        mse_imag = np.mean((preds_unscaled[..., 1] - y_eval_unscaled[..., 1]) ** 2)
        
        print(f"  ✓ MSE (unscaled space):")
        print(f"      Total: {mse_unscaled:.6e}")
        print(f"      Real:  {mse_real:.6e}")
        print(f"      Imag:  {mse_imag:.6e}")
        print()
        
        # Store results for summary
        all_mse_results.append({
            'channel': outer_channel_name,
            'mse_total': mse_unscaled,
            'mse_real': mse_real,
            'mse_imag': mse_imag,
            'eval_loss_scaled': eval_loss
        })

    print("="*80)
    print(f"Fine-tuning complete for all {len(fine_tune_file_names)} channels!")
    print(f"Results saved in: {ckpt_dir}")
    print("="*80)
    
    # Print summary of all MSE results
    if all_mse_results:
        print("\n" + "="*80)
        print("MSE SUMMARY (Unscaled Space)")
        print("="*80)
        print(f"{'Channel':<40} {'Total MSE':>15} {'Real MSE':>15} {'Imag MSE':>15}")
        print("-"*80)
        
        for result in all_mse_results:
            print(f"{result['channel']:<40} {result['mse_total']:>15.6e} "
                  f"{result['mse_real']:>15.6e} {result['mse_imag']:>15.6e}")
        
        # Calculate and display average MSE
        avg_mse_total = np.mean([r['mse_total'] for r in all_mse_results])
        avg_mse_real = np.mean([r['mse_real'] for r in all_mse_results])
        avg_mse_imag = np.mean([r['mse_imag'] for r in all_mse_results])
        
        print("-"*80)
        print(f"{'AVERAGE':<40} {avg_mse_total:>15.6e} "
              f"{avg_mse_real:>15.6e} {avg_mse_imag:>15.6e}")
        print("="*80)
        
        # Save summary to file
        df_summary = pd.DataFrame(all_mse_results)
        summary_csv_path = os.path.join(ckpt_dir, f"MAML_PCGrad_{args.k_qry}_shot_mse_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nMSE summary saved to: {summary_csv_path}")
        print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Fine-tune PCGrad-trained MAML model on unseen channels'
    )
    
    # Data arguments
    argparser.add_argument('--root', type=str,
                          default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-TDL/interpolated_noleak",
                          help='Path to dataset directory')
    argparser.add_argument('--device', type=str, default='cuda:0',
                          help='Device to use (cuda:0, cuda:1, or cpu)')
    argparser.add_argument('--save_init', type=str,
                          default="/home/rghasemi/Wireless_communication/SISO_TDL_init/pcgrad_std_scaler_interpolated_noleak",
                          help='Directory containing PCGrad-trained checkpoints')
    argparser.add_argument('--scaler_dir', type=str,
                          default="/home/rghasemi/Wireless_communication/TDL_updated_model",
                          help='Directory containing ChannelNet minmax_params.npz for scaling')
    
    # Training arguments
    argparser.add_argument('--step', type=int, default=5000,
                          help='Training step (for checkpoint naming)')
    argparser.add_argument('--epoch', type=int, default=5000,
                          help='Number of fine-tuning epochs per channel')
    argparser.add_argument('--batchsz', type=int, default=8,
                          help='Batch size for fine-tuning')
    
    # Few-shot arguments
    argparser.add_argument('--k_qry', type=int, default=5,
                          help='Number of query samples (k-shot)')
    argparser.add_argument('--k_spt', type=int, default=5,
                          help='Number of support samples')
    
    # Learning rates (must match training for checkpoint loading)
    argparser.add_argument('--update_lr', type=float, default=1e-4,
                          help='Fine-tuning learning rate')
    argparser.add_argument('--meta_lr', type=float, default=5e-4,
                          help='Meta learning rate (for checkpoint naming)')
    
    # MAML configuration (must match training)
    argparser.add_argument('--n_way', type=int, default=5,
                          help='n-way setting (for checkpoint naming)')
    argparser.add_argument('--update_step', type=int, default=3,
                          help='Number of inner loop updates')

    args = argparser.parse_args()
    fine_tune(args)

