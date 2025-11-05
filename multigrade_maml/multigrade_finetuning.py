import os
import sys
import torch
import numpy as np
import argparse

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Utils
from multigrade_meta import MultigradeMeta

def _scale_with_params(x, params, eps=1e-8):
    """
    Apply [-1,1] scaling using provided per-channel params
    """
    real = 2.0 * (x[..., 0] - params["min_real"]) / (params["max_real"] - params["min_real"] + eps) - 1.0
    imag = 2.0 * (x[..., 1] - params["min_imag"]) / (params["max_imag"] - params["min_imag"] + eps) - 1.0
    return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)

def _to_torch_ch_first(x_np, device):
    """
    Convert to torch tensor with channel-first format
    """
    x_cf = np.transpose(x_np, (0, 3, 1, 2))
    return torch.from_numpy(x_cf).to(device)

def fine_tune(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print("Multigrade MAML Fine-tuning")
    print("=" * 50)
    print(f"Number of grades: {args.num_grades}")
    print(f"Fine-tuning epochs: {args.epoch}")
    print("=" * 50)

    # Define the model configuration
    config = [
        ('conv2d', [64, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [64]),
        ('conv2d', [256, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [512, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [512]),
        ('conv2d', [256, 512, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [32, 256, 3, 3, 1, 1]),
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
    maml_finetuning = MultigradeMeta(args, config, num_grades=args.num_grades).to(device)

    # Load data
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    fine_tune_file_names = list(data_dict.keys())[4:]

    for outer_channel_name in fine_tune_file_names:
        print(f"\n=== Fine-tuning on channel: {outer_channel_name} ===")

        # Load channel data
        x_all = data_dict[outer_channel_name].astype(np.float32)
        y_all = labels_dict[outer_channel_name].astype(np.float32)

        # Split data
        x_pool, y_pool = x_all[:30], y_all[:30]
        x_eval, y_eval = x_all[30:], y_all[30:]

        # Scale data
        x_pool_s, x_params = Utils.standard_scaling(x_pool)
        y_pool_s, y_params = Utils.standard_scaling(y_pool)
        x_eval_s = _scale_with_params(x_eval, x_params)
        y_eval_s = _scale_with_params(y_eval, y_params)

        # k-shot subset
        x_shot_s, y_shot_s = x_pool_s[:args.k_qry], y_pool_s[:args.k_qry]

        # Load meta-trained weights
        ckpt_path = os.path.join(
            args.save_init,
            f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}",
            f"MultigradeMAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{args.step-1}"
            + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
        )
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            maml_finetuning.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from: {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        # Convert to torch tensors
        x_shot_t = _to_torch_ch_first(x_shot_s, device)
        y_shot_t = _to_torch_ch_first(y_shot_s, device)
        x_eval_t = _to_torch_ch_first(x_eval_s, device)
        y_eval_t = _to_torch_ch_first(y_eval_s, device)

        # Fine-tune using all grades
        optimizer = torch.optim.AdamW(maml_finetuning.parameters(), lr=args.update_lr, weight_decay=0.01)
        loss_path = os.path.join(args.save_init, f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}",
                                 f"MultigradeMAML_{args.k_qry}_shot_{outer_channel_name}_loss.npy")
        
        maml_finetuning.finetuning(optimizer, x_shot_t, y_shot_t, args.epoch, args.batchsz, device, loss_path)

        # Save fine-tuned model
        os.makedirs(os.path.join(args.save_init, f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}"), exist_ok=True)
        torch.save({'state_dict': maml_finetuning.state_dict()},
                   os.path.join(args.save_init, f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}",
                                f"MultigradeMAML_{args.k_qry}_shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}.pth"))

        # Evaluate
        eval_save_path_scaled = os.path.join(args.save_init, f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}",
                                             f"MultigradeMAML_{args.k_qry}_shot_{outer_channel_name}_predictions_scaled.npy")
        eval_loss, preds_scaled_cf = maml_finetuning.evaluate(x_eval_t, y_eval_t, args.batchsz, device,
                                                              save_path=eval_save_path_scaled)
        print(f"Eval (scaled space) loss: {eval_loss:.6f}")

        # Unscale predictions
        preds_scaled_cl = np.transpose(preds_scaled_cf, (0, 2, 3, 1))
        preds_unscaled = Utils.unscale_standard(preds_scaled_cl, y_params)

        eval_save_path_unscaled = os.path.join(args.save_init, f"multigrade_meta_model_nway_{args.n_way}_grades_{args.num_grades}",
                                               f"MultigradeMAML_{args.k_qry}_shot_{outer_channel_name}_predictions.npy")
        np.save(eval_save_path_unscaled, preds_unscaled)
        print(f"Predictions (unscaled) saved to {eval_save_path_unscaled}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, 
                          default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    argparser.add_argument('--device', type=str, default='cuda:0')
    argparser.add_argument('--save_init', type=str, 
                          default="/home/rghasemi/Wireless_communication/multigrade_maml_results")
    argparser.add_argument('--step', type=int, default=2000)
    argparser.add_argument('--epoch', type=int, default=1000)
    argparser.add_argument('--batchsz', type=int, default=8)
    argparser.add_argument('--k_qry', type=int, default=5)
    argparser.add_argument('--k_spt', type=int, default=5)
    argparser.add_argument('--update_lr', type=float, default=1e-3)
    argparser.add_argument('--meta_lr', type=float, default=1e-4)
    argparser.add_argument('--n_way', type=int, default=5)
    argparser.add_argument('--update_step', type=int, default=2)
    argparser.add_argument('--num_grades', type=int, default=3, 
                          help='Number of grades in multigrade learning')

    args = argparser.parse_args()
    fine_tune(args)
