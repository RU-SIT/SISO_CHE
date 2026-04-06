import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from Data_Nshot import ChannelEstimationNShot
from learner_model import Learner
from learner_model import make_fc_network, make_conv_network
from utils import DataLog
import pdb
from datautils import Utils
import pdb


def _scale_with_params(x, params, eps=1e-8):
    """
    Apply [-1,1] scaling using provided per-channel params from ChannelNet.
    x: np.ndarray (..., 2) channel-last
    params: dict with keys {min_real, max_real, min_imag, max_imag}
    """
    real = 2.0 * (x[..., 0] - params["min_real"]) / (params["max_real"] - params["min_real"] + eps) - 1.0
    imag = 2.0 * (x[..., 1] - params["min_imag"]) / (params["max_imag"] - params["min_imag"] + eps) - 1.0
    return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)


def _unscale_with_params(x_scaled, params, eps=1e-8):
    """
    Invert _scale_with_params to recover the original range.
    x_scaled: np.ndarray (..., 2) channel-last in [-1, 1]
    """
    real = 0.5 * (x_scaled[..., 0] + 1.0) * (params["max_real"] - params["min_real"] + eps) + params["min_real"]
    imag = 0.5 * (x_scaled[..., 1] + 1.0) * (params["max_imag"] - params["min_imag"] + eps) + params["min_imag"]
    return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)


def _to_torch_ch_first(x_np):
    """
    x_np: [N, H, W, 2] -> torch [N, 2, H, W]
    """
    x_cf = np.transpose(x_np, (0, 3, 1, 2))
    return torch.from_numpy(x_cf)


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

   
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Initialize the Learner-based model using a convolutional network
    # Here, we use in_channels=2 and out_dim=2 (as in your training code) and set the batch size via args.batchsz.
    learner_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=args.batchsz)
    meta_learner = Learner(model=learner_net,
                           loss_function=torch.nn.MSELoss(),
                           inner_lr=args.inner_lr,
                           outer_lr=args.outer_lr,
                           GPU=(args.device.startswith("cuda")))
    meta_learner.model.to(device)

    # Load data dictionaries (channel data and labels).
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    # Use files from index 10 onward for fine-tuning.
    fine_tune_file_names = list(data_dict.keys())[10:]
    
    # Track MSE for all channels
    all_mse_results = []
    
    # pdb.set_trace()
    for outer_channel_name in fine_tune_file_names:
        # Prepare the data:
        x_all = data_dict[outer_channel_name].astype(np.float32)
        y_all = labels_dict[outer_channel_name].astype(np.float32)

        # Compute per-channel scaling parameters from this channel's data
        x_all_s, x_params = Utils.standard_scaling(x_all)
        y_all_s, y_params = Utils.standard_scaling(y_all)
        # print(f"  Per-channel scaling computed: X[{float(x_params['min_real']):.3f}, {float(x_params['max_real']):.3f}], "
        #       f"Y[{float(y_params['min_real']):.3f}, {float(y_params['max_real']):.3f}]")

        # Convert to torch tensors (channel-first)
        data = _to_torch_ch_first(x_all_s)
        labels = _to_torch_ch_first(y_all_s)

        # Split the data into training and evaluation sets.
        train_data, eval_data = data[:30], data[30:]
        train_labels, eval_labels = labels[:30], labels[30:]
        # Use a subset for few-shot adaptation.
        data_shot, label_shot = train_data[:args.k_qry], train_labels[:args.k_qry]

        checkpoint_path = os.path.join(
            args.save_dir,
            f"meta_model_nway_{args.n_way}",
            f"wireless_IMAML_{args.K_shot}_shot_checkpoint_step_{args.step-1}_LAM{args.lam}.pth.tar"
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        meta_learner.model.load_state_dict(checkpoint['state_dict'])
    
        # Create an optimizer for fine tuning
        optimizer = torch.optim.Adam(meta_learner.model.parameters(), lr=args.update_lr)
    
        # Fine-tune the model on the current channel.
        loss_path = os.path.join(args.save_dir,
                                 f"meta_model_nway_{args.n_way}",
                                 f"wireless_IMAML_{args.K_shot}_shot_{outer_channel_name}_loss.npy")
        meta_learner.finetuning(optimizer, data_shot, label_shot, args.epoch, args.batchsz, device, loss_path)
    
        # Save the fine-tuned model checkpoint
        finetuned_model_path = os.path.join(
            args.save_dir,
            f"meta_model_nway_{args.n_way}",
            f"wireless_IMAML_{args.K_shot}_shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}_LAM{args.lam}.pth"
        )
        torch.save({'state_dict': meta_learner.model.state_dict()}, finetuned_model_path)
        print(f"Fine-tuned model saved to {finetuned_model_path}")
    
        # Evaluate the fine-tuned model
        eval_save_path = os.path.join(
            args.save_dir,
            f"meta_model_nway_{args.n_way}",
            f"wireless_IMAML_{args.K_shot}_shot_LAM{args.lam}_{outer_channel_name}_predictions.npy"
        )
        eval_save_path_scaled = os.path.join(
            args.save_dir,
            f"meta_model_nway_{args.n_way}",
            f"wireless_IMAML_{args.K_shot}_shot_LAM{args.lam}_{outer_channel_name}_predictions_scaled.npy"
        )
        eval_loss, preds_scaled_cf = meta_learner.evaluate(eval_data, eval_labels, args.batchsz, device,
                                                           save_path=eval_save_path_scaled)
        print(f"Evaluation Loss for channel {outer_channel_name}: {eval_loss:.4f}")

        # Convert predictions back to channel-last, unscale, and save
        preds_scaled_cl = np.transpose(preds_scaled_cf, (0, 2, 3, 1))
        preds_unscaled = _unscale_with_params(preds_scaled_cl, y_params)
        np.save(eval_save_path, preds_unscaled)
        print(f"Evaluation Loss for channel {outer_channel_name}: {eval_loss:.4f}")
        print(f"Scaled predictions saved to {eval_save_path_scaled}")
        print(f"Predictions (unscaled) saved to {eval_save_path}")
        
        # ---- Calculate MSE in original (unscaled) space ----
        # Unscale ground truth labels for comparison
        eval_labels_cl = np.transpose(eval_labels.cpu().numpy(), (0, 2, 3, 1))  # [N, H, W, 2]
        eval_labels_unscaled = _unscale_with_params(eval_labels_cl, y_params)
        
        # Compute MSE in original space
        mse_unscaled = np.mean((preds_unscaled - eval_labels_unscaled) ** 2)
        
        # Compute MSE per channel (real and imaginary)
        mse_real = np.mean((preds_unscaled[..., 0] - eval_labels_unscaled[..., 0]) ** 2)
        mse_imag = np.mean((preds_unscaled[..., 1] - eval_labels_unscaled[..., 1]) ** 2)
        
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
        ckpt_dir = os.path.join(args.save_dir, f"meta_model_nway_{args.n_way}")
        summary_csv_path = os.path.join(ckpt_dir, f"iMAML_{args.K_shot}shot_LAM{args.lam}_mse_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nMSE summary saved to: {summary_csv_path}")
        print()

if __name__ == '__main__':
    import sys
    from pathlib import Path

    _WC_ROOT = Path(__file__).resolve().parents[2]
    if str(_WC_ROOT) not in sys.path:
        sys.path.insert(0, str(_WC_ROOT))
    from paths import default_dataset_umi_siso_folder, default_imaml_save_dir, default_tdl_updated_model

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, default=default_dataset_umi_siso_folder(), help="Path to the data directory")
    argparser.add_argument('--device', type=str, default='cuda:1', help="Device to run on")
    argparser.add_argument('--save_dir', type=str, default=default_imaml_save_dir(), help="Directory where checkpoints and results are saved")
    argparser.add_argument('--step', type=int, default= 4000, help="Checkpoint step to load")
    argparser.add_argument('--epoch', type=int, default=500, help="Number of epochs for fine-tuning")
    argparser.add_argument('--batchsz', type=int, default=8, help="Batch size for fine-tuning")
    argparser.add_argument('--k_qry', type=int, default=5, help="Number of query (shot) samples for fine-tuning")
    argparser.add_argument('--k_spt', type=int, default=5, help="Number of support samples (if used)")
    argparser.add_argument('--update_lr', type=float, default=1e-3, help="Learning rate for fine-tuning")
    argparser.add_argument('--meta_lr', type=float, default=1e-4, help="Meta learning rate (for information)")
    argparser.add_argument('--n_way', type=int, default=4, help="Number of classes")
    argparser.add_argument('--lam', type=int, default=3.0, help="regularization term")
    argparser.add_argument('--K_shot', type=int, default=5, help="Number of instances per class (shot) used in training")
    argparser.add_argument('--inner_lr', type=float, default=1e-3, help="Inner loop learning rate")
    argparser.add_argument('--outer_lr', type=float, default=1e-4, help="Outer loop learning rate")
    argparser.add_argument('--update_step', type=int, default=2, help="Number of fine-tuning update steps (if applicable)")
    argparser.add_argument('--scaler_dir', type=str,
                           default=default_tdl_updated_model(),
                           help="Directory containing ChannelNet minmax_params.npz for scaling")
    
    args = argparser.parse_args()
    fine_tune(args)