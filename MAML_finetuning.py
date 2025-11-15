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
# -----------------------------------------

def _find_best_checkpoint(ckpt_dir, k_qry, k_spt, meta_lr, update_lr):
    """
    Locate the best checkpoint saved during training based on filename pattern.
    Returns (path, step) if found, else (None, None).
    """
    if not os.path.isdir(ckpt_dir):
        return None, None
    
    prefix = f"MAML_{k_qry}_shot_{k_spt}_query_BEST_checkpoint_step_"
    suffix = f"_MetaLr{meta_lr}_TaskLr{update_lr}.pth.tar"
    
    best_path = None
    best_step = -1
    
    for fname in os.listdir(ckpt_dir):
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue
        try:
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
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # Define the model configuration
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
    maml_finetuning = Meta(args, config).to(device)

    data_dict   = np.load(os.path.join(args.root, 'channel_data_dict.npy'),  allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    fine_tune_file_names = list(data_dict.keys())[10:]  

    x_params, y_params = _load_minmax_params(args.scaler_dir)

    # Track MSE for all channels
    all_mse_results = []

    for outer_channel_name in fine_tune_file_names:
        print(f"\n=== Fine-tuning on channel: {outer_channel_name} ===")

        # ---- Load channel data ----
        x_all = data_dict[outer_channel_name].astype(np.float32)   # [N, H, W, 2]
        y_all = labels_dict[outer_channel_name].astype(np.float32) # [N, H, W, 2]
        # pdb.set_trace()
        # ---- Load global scaling parameters (shared with ChannelNet) ----
        # ---- Apply scaling using shared parameters ----
        x_all_s = _scale_with_params(x_all, x_params)
        y_all_s = _scale_with_params(y_all, y_params)

        # ---- Split into k-shot pool and eval (no leakage) ----
        x_pool_s, y_pool_s = x_all_s[:30], y_all_s[:30]
        x_eval_s, y_eval_s = x_all_s[30:], y_all_s[30:]

        # ---- k-shot subset from the (scaled) pool ----
        x_shot_s, y_shot_s = x_pool_s[:args.k_qry], y_pool_s[:args.k_qry]

        # ---- Load meta-trained weights (path matches MAML_Trainer naming) ----
        ckpt_dir = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}")
        best_ckpt_path, best_step = _find_best_checkpoint(
            ckpt_dir, args.k_qry, args.k_spt, args.meta_lr, args.update_lr
        )
        
        if best_ckpt_path is not None:
            print(f"Loading best checkpoint: {best_ckpt_path} (step {best_step})")
            checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        else:
            fallback_path = os.path.join(
                ckpt_dir,
                f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{args.step-1}"
                + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            print(f"Best checkpoint not found. Falling back to: {fallback_path}")
            checkpoint = torch.load(fallback_path, map_location=device, weights_only=True)
        maml_finetuning.load_state_dict(checkpoint['state_dict'])

        # ---- To torch (channel-first) ----
        x_shot_t  = _to_torch_ch_first(x_shot_s, device)
        y_shot_t  = _to_torch_ch_first(y_shot_s, device)
        x_eval_t  = _to_torch_ch_first(x_eval_s, device)
        y_eval_t  = _to_torch_ch_first(y_eval_s, device)
        # npdb.set_trace()
        
        # ---- Fine-tune on scaled k-shot ----
        optimizer = torch.optim.AdamW(maml_finetuning.parameters(), lr=args.update_lr, weight_decay=0.01)
        loss_path = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}",
                                 f"MAML_{args.k_qry}_shot_{outer_channel_name}_loss.npy")
        maml_finetuning.finetuning(optimizer, x_shot_t, y_shot_t, args.epoch, args.batchsz, device, loss_path)

        # Save the fine-tuned model
        os.makedirs(os.path.join(args.save_init, f"meta_model_nway_{args.n_way}"), exist_ok=True)
        torch.save({'state_dict': maml_finetuning.state_dict()},
                   os.path.join(args.save_init, f"meta_model_nway_{args.n_way}",
                                f"MAML_{args.k_qry}_shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}.pth"))

        # ---- Evaluate on scaled eval split ----
        eval_save_path_scaled = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}",
                                             f"MAML_{args.k_qry}_shot_{outer_channel_name}_predictions_scaled.npy")
        eval_loss, preds_scaled_cf = maml_finetuning.evaluate(x_eval_t, y_eval_t, args.batchsz, device,
                                                              save_path=eval_save_path_scaled)
        print(f"Eval (scaled space) loss: {eval_loss:.6f}")

        # ---- Unscale predictions to original range for saving/metrics ----
        # preds_scaled_cf is [N, 2, H, W] -> channel-last for unscale
        preds_scaled_cl = np.transpose(preds_scaled_cf, (0, 2, 3, 1))  # [N, H, W, 2]
        preds_unscaled  = Utils.unscale_standard(preds_scaled_cl, y_params)  # back to original scale

        eval_save_path_unscaled = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}",
                                               f"MAML_{args.k_qry}_shot_{outer_channel_name}_predictions.npy")
        np.save(eval_save_path_unscaled, preds_unscaled)
        print(f"Predictions (unscaled) saved to {eval_save_path_unscaled}")
        
        # ---- Calculate MSE in original (unscaled) space ----
        # Unscale ground truth labels for comparison
        y_eval_cl = np.transpose(y_eval_t.cpu().numpy(), (0, 2, 3, 1))  # [N, H, W, 2]
        y_eval_unscaled = Utils.unscale_standard(y_eval_cl, y_params)
        
        # Compute MSE in original space
        mse_unscaled = np.mean((preds_unscaled - y_eval_unscaled) ** 2)
        
        # Compute MSE per channel (real and imaginary)
        mse_real = np.mean((preds_unscaled[..., 0] - y_eval_unscaled[..., 0]) ** 2)
        mse_imag = np.mean((preds_unscaled[..., 1] - y_eval_unscaled[..., 1]) ** 2)
        
        print(f"MSE (unscaled space):")
        print(f"  Total: {mse_unscaled:.6e}")
        print(f"  Real:  {mse_real:.6e}")
        print(f"  Imag:  {mse_imag:.6e}")
        
        # Store results for summary
        all_mse_results.append({
            'channel': outer_channel_name,
            'mse_total': mse_unscaled,
            'mse_real': mse_real,
            'mse_imag': mse_imag,
            'eval_loss_scaled': eval_loss
        })

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
        ckpt_dir = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}")
        summary_csv_path = os.path.join(ckpt_dir, f"MAML_{args.k_qry}_shot_mse_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nMSE summary saved to: {summary_csv_path}")
        print()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    argparser.add_argument('--device', type=str, default='cuda:0')
    argparser.add_argument('--save_init', type=str, default="/home/rghasemi/Wireless_communication/SISO_UMi_init/std_scaler_interpolated_noleak")
    argparser.add_argument('--step', type=int, default=5000)
    argparser.add_argument('--epoch', type=int, default=5000)
    argparser.add_argument('--batchsz', type=int, default=8)
    argparser.add_argument('--k_qry', type=int, default=5)
    argparser.add_argument('--k_spt', type=int, default=5)
    argparser.add_argument('--update_lr', type=float, default=1e-4)
    argparser.add_argument('--meta_lr', type=float, default=5e-4)
    argparser.add_argument('--n_way', type=int, default=5)
    argparser.add_argument('--update_step', type=int, default=3)
    argparser.add_argument('--scaler_dir', type=str,
                           default="/home/rghasemi/Wireless_communication/TDL_updated_model",
                           help='Directory containing ChannelNet minmax_params.npz for scaling')

    args = argparser.parse_args()
    fine_tune(args)


# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# from utils import Utils
# import argparse
# from meta import Meta
# import pdb



# def fine_tune(args):
#     torch.manual_seed(222)
#     torch.cuda.manual_seed_all(222)
#     np.random.seed(222)

#     print(args)

#     # Define the model configuration
#     config = [
#         ('conv2d', [64, 2, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [64]),
#         ('conv2d', [256, 64, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [256]),
#         ('conv2d', [512, 256, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [512]),
#         ('conv2d', [256, 512, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [256]),
#         ('conv2d', [32, 256, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [32]),
#         ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [args.batchsz]),
#         ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
#     ]
   
   
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     maml_finetuning = Meta(args, config).to(device)

#     data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
#     labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

#     fine_tune_file_names = list(data_dict.keys())[4:]

#     for outer_channel_name in fine_tune_file_names:
        
#         data = torch.tensor(data_dict[outer_channel_name].transpose(0, 3, 1, 2), dtype=torch.float32)
#         labels = torch.tensor(labels_dict[outer_channel_name].transpose(0, 3, 1, 2), dtype=torch.float32)

#         train_data, eval_data = data[:30], data[30:]
#         train_labels, eval_labels = labels[:30], labels[30:]
#         data_shot, label_shot = train_data[:args.k_qry], train_labels[:args.k_qry]
#         # pdb.set_trace()
#         checkpoint_path = os.path.join(
#             args.save_init,
#             f"meta_model_nway_{args.n_way}",
#             f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{args.step-1}_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.pth.tar"
#         )
#         checkpoint = torch.load(checkpoint_path, weights_only=True)
#         maml_finetuning.load_state_dict(checkpoint['state_dict'])

#         optimizer = torch.optim.Adam(maml_finetuning.parameters(), lr=args.update_lr)

#         print(f"Fine-tuning on channel: {outer_channel_name}")
#         loss_path = os.path.join(args.save_init,f"meta_model_nway_{args.n_way}", f"MAML_{args.k_qry}_shot_{outer_channel_name}_loss.npy")
#         maml_finetuning.finetuning(optimizer, data_shot, label_shot, args.epoch, args.batchsz, device, loss_path)

#         # Save the fine-tuned model
#         torch.save({'state_dict': maml_finetuning.state_dict()},
#                    os.path.join(args.save_init,f"meta_model_nway_{args.n_way}", f"MAML_{args.k_qry}_shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}.pth"))

#         # Evaluate the model
#         eval_save_path = os.path.join(args.save_init,f"meta_model_nway_{args.n_way}", f"MAML_{args.k_qry}_shot_{outer_channel_name}_predictions.npy")
#         eval_loss, predictions = maml_finetuning.evaluate(eval_data, eval_labels, args.batchsz, device, save_path = eval_save_path)
#         print(f"MAML_{args.k_qry}_shot[{outer_channel_name}] Evaluation Loss: {eval_loss:.4f}")
#         print(f"Predictions saved to {eval_save_path}")


