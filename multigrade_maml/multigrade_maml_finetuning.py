import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
import pdb

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add current directory to path for multigrade_maml_stair
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import Utils
from Data_Nshot import ChannelEstimationNShot
from multigrade_maml_stair import MultigradeMAMLStair

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
# -----------------------------------------


def _load_minmax_params(save_dir):
    """
    Load global min/max scaling parameters saved by ChannelNet.
    """
    params_path = os.path.join(save_dir, "minmax_params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Cannot find minmax_params.npz in {save_dir}. "
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


def fine_tune(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print("=" * 60)
    print("MULTIGRADE MAML FINE-TUNING")
    print("=" * 60)
    print(args)
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data to determine dataset type
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()
    
    # Determine dataset type and get test file names (for fine-tuning)
    # Use Data_Nshot to get train/test split logic
    temp_data_loader = ChannelEstimationNShot(
        root=args.root,
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    
    # Get test file names (these are the unseen channels for fine-tuning)
    fine_tune_file_names = temp_data_loader.test_file_names
    dataset_name = temp_data_loader.data_type
    
    print(f"Dataset type: {dataset_name}")
    print(f"Training channels: {len(temp_data_loader.train_file_names)}")
    print(f"Fine-tuning channels (unseen): {len(fine_tune_file_names)}")
    print(f"Fine-tuning files: {fine_tune_file_names}")
    print("=" * 60)

    # Load shared scaling parameters (same as ChannelNet fine-tuning)
    x_params, y_params = _load_minmax_params(args.scaler_dir)

    # Determine checkpoint directory based on dataset
    root_parts = args.root.split('/')
    dataset_name_for_path = dataset_name
    for part in root_parts:
        if 'UMi' in part or 'umi' in part.lower():
            dataset_name_for_path = 'UMi'
            for p in root_parts:
                if 'speed' in p.lower():
                    dataset_name_for_path = f'UMi_{p}'
                    break
            break
        elif 'TDL' in part or 'tdl' in part.lower():
            dataset_name_for_path = 'TDL'
            for p in root_parts:
                if 'speed' in p.lower() or any(char.isdigit() for char in p):
                    dataset_name_for_path = f'TDL_{p}'
                    break
            break

    # Find the checkpoint (use the last grade's checkpoint)
    checkpoint_base_dir = os.path.join(
        "multigrade_maml_results",
        dataset_name_for_path,
        f"checkpoints_nway_{args.n_way}_grades_{args.grades}"
    )
    
    print(f"Looking for checkpoints in: {checkpoint_base_dir}")
    
    # Find the checkpoint for the last grade at the specified step
    checkpoint_files = [f for f in os.listdir(checkpoint_base_dir) if f.endswith('.pth.tar')] if os.path.exists(checkpoint_base_dir) else []
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_base_dir}")
    
    # Find the checkpoint for the last grade at the specified step
    last_grade = args.grades
    step_str = str(args.step)
    
    # Try to find checkpoint matching the step
    matching_checkpoints = [f for f in checkpoint_files if f'Grade{last_grade}' in f and step_str in f]
    
    if not matching_checkpoints:
        # If exact step not found, use the last checkpoint for the last grade
        last_grade_checkpoints = [f for f in checkpoint_files if f'Grade{last_grade}' in f]
        if last_grade_checkpoints:
            # Sort by step number and take the last one
            def extract_step(fname):
                parts = fname.split('_')
                for p in parts:
                    if p.startswith('Step'):
                        return int(p.replace('Step', ''))
                return 0
            last_grade_checkpoints.sort(key=extract_step, reverse=True)
            checkpoint_file = last_grade_checkpoints[0]
            print(f"Warning: Step {args.step} not found. Using latest checkpoint: {checkpoint_file}")
        else:
            raise FileNotFoundError(f"No checkpoint found for Grade {last_grade} in {checkpoint_base_dir}")
    else:
        checkpoint_file = matching_checkpoints[0]
    
    ckpt_path = os.path.join(checkpoint_base_dir, checkpoint_file)
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Get training configuration from checkpoint
    # If batchsz is not in checkpoint metadata, detect it from state_dict
    if 'batchsz' in checkpoint:
        training_batchsz = checkpoint['batchsz']
    else:
        # Detect batchsz from checkpoint state_dict by looking at net.vars.20 shape
        # net.vars.20 should be [batchsz, 32, 3, 3] based on the config
        state_dict = checkpoint['state_dict']
        if 'net.vars.20' in state_dict:
            training_batchsz = state_dict['net.vars.20'].shape[0]
            print(f"Detected batchsz={training_batchsz} from checkpoint state_dict")
        else:
            training_batchsz = 2  # Default fallback
            print(f"Warning: Could not detect batchsz from checkpoint, using default={training_batchsz}")
    
    training_n_way = checkpoint.get('n_way', args.n_way)
    training_grades = checkpoint.get('grades', args.grades)
    training_meta_lr = checkpoint.get('meta_lr', args.meta_lr)
    training_update_lr = checkpoint.get('update_lr', args.update_lr)
    
    print(f"Checkpoint info:")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Grade: {checkpoint.get('grade', 'unknown')}")
    print(f"  Batch size: {training_batchsz} (from checkpoint)")
    print(f"  N-way: {training_n_way}")
    print(f"  Grades: {training_grades}")
    print(f"  Meta LR: {training_meta_lr}")
    print(f"  Update LR: {training_update_lr}")
    
    # Update args to match checkpoint
    args.batchsz = training_batchsz
    args.n_way = training_n_way
    args.grades = training_grades
    
    # Simple args class for MultigradeMAMLStair
    class TrainingArgs:
        def __init__(self):
            self.update_lr = training_update_lr
            self.meta_lr = training_meta_lr
            self.n_way = training_n_way
            self.k_spt = args.k_spt
            self.k_qry = args.k_qry
            self.batchsz = training_batchsz
            self.update_step = args.update_step
            self.num_grades = training_grades

    training_args = TrainingArgs()
    
    # Create config with correct batchsz from checkpoint
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
    
    # Create multigrade MAML model with correct config
    maml_finetuning = MultigradeMAMLStair(training_args, config, num_grades=training_grades).to(device)
    
    maml_finetuning.load_state_dict(checkpoint['state_dict'])
    print(f"✓ Loaded checkpoint successfully")

    # Track MSE for all channels
    all_mse_results = []

    # Fine-tuning loop
    for outer_channel_name in fine_tune_file_names:
        print(f"\n{'='*60}")
        print(f"Fine-tuning on channel: {outer_channel_name}")
        print(f"{'='*60}")

        # ---- Load channel data ----
        x_all = data_dict[outer_channel_name].astype(np.float32)   # [N, H, W, 2]
        y_all = labels_dict[outer_channel_name].astype(np.float32) # [N, H, W, 2]

        # ---- Apply global scaling params (shared with ChannelNet) ----
        x_all_s = _scale_with_params(x_all, x_params)
        y_all_s = _scale_with_params(y_all, y_params)

        # ---- Split into k-shot pool and eval (no leakage) ----
        x_pool_s, y_pool_s = x_all_s[:30], y_all_s[:30]
        x_eval_s, y_eval_s = x_all_s[30:], y_all_s[30:]

        # ---- k-shot subset from the (scaled) pool ----
        x_shot_s, y_shot_s = x_pool_s[:args.k_qry], y_pool_s[:args.k_qry]

        # ---- To torch (channel-first) ----
        x_shot_t  = _to_torch_ch_first(x_shot_s, device)
        y_shot_t  = _to_torch_ch_first(y_shot_s, device)
        x_eval_t  = _to_torch_ch_first(x_eval_s, device)
        y_eval_t  = _to_torch_ch_first(y_eval_s, device)
        
        # ---- Fine-tune on scaled k-shot ----
        optimizer = torch.optim.AdamW(maml_finetuning.parameters(), lr=args.update_lr, weight_decay=0.01)
        
        # Create save directory for this dataset
        save_base_dir = os.path.join("multigrade_maml_results", dataset_name_for_path, "finetuning")
        os.makedirs(save_base_dir, exist_ok=True)
        
        loss_path = os.path.join(save_base_dir,
                                 f"MultigradeMAML_{args.k_qry}shot_{outer_channel_name}_loss.npy")
        
        print(f"Fine-tuning on {len(x_shot_t)} samples...")
        maml_finetuning.finetuning(optimizer, x_shot_t, y_shot_t, args.epoch, args.batchsz, device, loss_path)

        # Save the fine-tuned model
        model_save_path = os.path.join(save_base_dir,
                                       f"MultigradeMAML_{args.k_qry}shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}.pth")
        torch.save({'state_dict': maml_finetuning.state_dict()}, model_save_path)
        print(f"✓ Fine-tuned model saved to: {model_save_path}")

        # ---- Evaluate on scaled eval split ----
        eval_save_path_scaled = os.path.join(save_base_dir,
                                             f"MultigradeMAML_{args.k_qry}shot_{outer_channel_name}_predictions_scaled.npy")
        eval_loss, preds_scaled_cf = maml_finetuning.evaluate(x_eval_t, y_eval_t, args.batchsz, device,
                                                              save_path=eval_save_path_scaled)
        print(f"Eval (scaled space) loss: {eval_loss:.6f}")

        # ---- Unscale predictions to original range for saving/metrics ----
        # preds_scaled_cf is [N, 2, H, W] -> channel-last for unscale
        preds_scaled_cl = np.transpose(preds_scaled_cf, (0, 2, 3, 1))  # [N, H, W, 2]
        preds_unscaled  = Utils.unscale_standard(preds_scaled_cl, y_params)  # back to original scale

        eval_save_path_unscaled = os.path.join(save_base_dir,
                                               f"MultigradeMAML_{args.k_qry}shot_{outer_channel_name}_predictions.npy")
        np.save(eval_save_path_unscaled, preds_unscaled)
        print(f"✓ Predictions (unscaled) saved to {eval_save_path_unscaled}")
        
        # ---- Calculate MSE in original (unscaled) space ----
        # Unscale ground truth labels for comparison
        y_eval_cl = np.transpose(y_eval_t.cpu().numpy(), (0, 2, 3, 1))  # [N, H, W, 2]
        y_eval_unscaled = Utils.unscale_standard(y_eval_cl, y_params)
        
        # Compute MSE in original space
        mse_unscaled = np.mean((preds_unscaled - y_eval_unscaled) ** 2)
        
        # Compute MSE per channel (real and imaginary)
        mse_real = np.mean((preds_unscaled[..., 0] - y_eval_unscaled[..., 0]) ** 2)
        mse_imag = np.mean((preds_unscaled[..., 1] - y_eval_unscaled[..., 1]) ** 2)
        
        print(f"✓ MSE (unscaled space):")
        print(f"    Total: {mse_unscaled:.6e}")
        print(f"    Real:  {mse_real:.6e}")
        print(f"    Imag:  {mse_imag:.6e}")
        
        # Store results for summary
        all_mse_results.append({
            'channel': outer_channel_name,
            'mse_total': mse_unscaled,
            'mse_real': mse_real,
            'mse_imag': mse_imag,
            'eval_loss_scaled': eval_loss
        })
        print(f"{'='*60}")

    print("\n" + "=" * 60)
    print(" FINE-TUNING COMPLETED!")
    print("=" * 60)
    print(f"Results saved in: {save_base_dir}")
    print("=" * 60)
    
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
        summary_csv_path = os.path.join(save_base_dir, f"MultigradeMAML_{args.k_qry}shot_mse_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nMSE summary saved to: {summary_csv_path}")
        print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Multigrade MAML Fine-tuning')
    argparser.add_argument('--root', type=str, 
                          default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak",
                          help='Path to data root directory')
    argparser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    argparser.add_argument('--save_init', type=str, default="multigrade_maml_results",
                          help='Base directory for saving results (will be extended with dataset name)')
    argparser.add_argument('--step', type=int, default=99, 
                          help='Training step to load checkpoint from (will use last grade checkpoint)')
    argparser.add_argument('--epoch', type=int, default=5000, help='Number of fine-tuning epochs')
    argparser.add_argument('--batchsz', type=int, default=8, help='Batch size for fine-tuning')
    argparser.add_argument('--k_qry', type=int, default=5, help='Number of query samples (k-shot for fine-tuning)')
    argparser.add_argument('--k_spt', type=int, default=5, help='Number of support samples')
    argparser.add_argument('--update_lr', type=float, default=1e-3, help='Fine-tuning learning rate')
    argparser.add_argument('--meta_lr', type=float, default=1e-4, help='Meta learning rate (used during training)')
    argparser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
    argparser.add_argument('--update_step', type=int, default=2, help='Number of inner loop update steps')
    argparser.add_argument('--grades', type=int, default=3, help='Number of grades in multigrade model')
    argparser.add_argument('--scaler_dir', type=str,
                          default="/home/rghasemi/Wireless_communication/TDL_updated_model",
                          help='Directory containing ChannelNet minmax_params.npz for scaling')

    args = argparser.parse_args()
    fine_tune(args)

