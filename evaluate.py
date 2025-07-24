import os
import torch
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from meta import Meta
from metrics import Metric, extract_snr
from utils import Utils

def mse(y_true, y_pred):
    """Computes mean squared error"""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    if y_true.shape != y_pred.shape:
        # This can happen if the batch size is 1
        y_pred = y_pred.squeeze(0)
    return torch.mean(torch.square(torch.abs(y_true - y_pred))).item()

def plot_evaluation_results(results_df, output_dir):
    """Plots MSE and BER vs. SNR for each channel."""
    
    # Create a new column for the legend
    results_df['legend'] = results_df.apply(
        lambda row: f"{row['method']} ({int(row['n_shot'])}-shot)" if pd.notna(row['n_shot']) else row['method'],
        axis=1
    )
    
    channels = results_df['channel'].unique()
    
    for channel in channels:
        channel_df = results_df[results_df['channel'] == channel]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot MSE
        for method in channel_df['legend'].unique():
            method_df = channel_df[channel_df['legend'] == method]
            ax1.plot(method_df['snr'], method_df['mse'], marker='o', linestyle='-', label=method)
        
        ax1.set_title(f'MSE vs. SNR for {channel}')
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('MSE')
        ax1.grid(True)
        ax1.legend()
        
        # Plot BER
        for method in channel_df['legend'].unique():
            method_df = channel_df[channel_df['legend'] == method]
            ax2.plot(method_df['snr'], method_df['ber'], marker='x', linestyle='--', label=method)
            
        ax2.set_title(f'BER vs. SNR for {channel}')
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('BER')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"evaluation_plot_{channel}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for {channel} to {plot_path}")

def ber(metric, y_pred, snr, modulation='16QAM'):
    """Computes bit error rate"""
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return metric.bit_error_rate(y_pred, snr, modulation=modulation)

def ls_estimation(eval_data, pilot_indices, grid_indices):
    """Performs LS estimation using linear interpolation for a batch of data."""
    eval_data_np = eval_data.cpu().numpy()
    num_samples = eval_data_np.shape[0]
    num_subcarriers = eval_data_np.shape[1]
    num_symbols = eval_data_np.shape[2]
    
    interpolated_batch = []

    for i in range(num_samples):
        sample = eval_data_np[i]
        
        # Extract pilot values at pilot locations
        pilot_values = sample[pilot_indices[:, 0], pilot_indices[:, 1]]
        
        # Interpolate real and imaginary parts
        interpolated_grid = griddata(pilot_indices, pilot_values, grid_indices, method='linear')
        
        # Fill NaNs using nearest neighbor interpolation for points outside the convex hull
        nan_mask = np.isnan(interpolated_grid).any(axis=1)
        if np.any(nan_mask):
            nearest_fill = griddata(pilot_indices, pilot_values, grid_indices[nan_mask], method='nearest')
            interpolated_grid[nan_mask] = nearest_fill

        interpolated_sample = interpolated_grid.reshape(num_subcarriers, num_symbols, 2)
        interpolated_batch.append(interpolated_sample)

    return torch.from_numpy(np.array(interpolated_batch)).float()

def lmmse_estimation(h_ls, true_channels, snr):
    """Simplified LMMSE estimation"""
    # Convert to complex tensors
    h_ls_complex = torch.complex(h_ls[..., 0], h_ls[..., 1])
    true_channels_complex = torch.complex(true_channels[..., 0], true_channels[..., 1])
    
    h_ls_flat = h_ls_complex.view(h_ls_complex.shape[0], -1)
    true_channels_flat = true_channels_complex.view(true_channels_complex.shape[0], -1)
    
    # Rhh = torch.mean(torch.einsum('bi,bj->bij', true_channels_flat, true_channels_flat.conj()), axis=0)
    # The above line is memory inefficient, let's use matrix multiplication
    Rhh = (true_channels_flat.conj().T @ true_channels_flat) / true_channels_flat.shape[0]
    noise_var = 1 / (10**(snr / 10))
    # Assuming noise is white, so Rnn is diagonal
    Rnn = torch.eye(Rhh.shape[0], device=h_ls.device) * noise_var
    
    lmmse_filter = Rhh @ torch.inverse(Rhh + Rnn)
    h_lmmse_flat = torch.einsum('ij,bj->bi', lmmse_filter, h_ls_flat)
    
    h_lmmse_complex = h_lmmse_flat.view(true_channels_complex.shape)
    
    # Convert back to real tensor representation
    h_lmmse = torch.stack([h_lmmse_complex.real, h_lmmse_complex.imag], dim=-1)
    return h_lmmse

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()
    test_file_names = list(data_dict.keys())[10:]
    
    # Initialize results storage
    results_df = pd.DataFrame(columns=["snr", "mse", "ber", "method", "channel", "n_shot"])
    
    metric = Metric()
    
    # Define model config
    config = [
        ('conv2d', [16, 2, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [16]),
        ('conv2d', [8, 16, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [8]),
        ('conv2d', [args.batchsz, 8, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [args.batchsz]),
        ('conv2d', [16, args.batchsz, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [16]),
        ('conv2d', [16, 16, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [16]),
        ('conv2d', [args.batchsz, 16, 3, 3, 1, 1]), ('tanh', [True]), ('avg_pool2d', [3, 1, 1]), ('bn', [args.batchsz]),
        ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    ]

    for channel_name in test_file_names:
        print(f"--- Evaluating channel: {channel_name} ---")
        
        # Load data for the current channel
        channel_data = data_dict[channel_name]
        channel_labels = labels_dict[channel_name]
        snr = extract_snr(channel_name)
        
        # Split data into train/eval and ensure correct dtype
        train_data, eval_data = torch.from_numpy(channel_data[:30]).float(), torch.from_numpy(channel_data[30:]).float()
        train_labels, eval_labels = torch.from_numpy(channel_labels[:30]).float(), torch.from_numpy(channel_labels[30:]).float()
        
        # Define pilot and grid indices for interpolation
        num_subcarriers = eval_data.shape[1]
        num_symbols = eval_data.shape[2]
        pilot_rows, pilot_cols = np.mgrid[0:num_subcarriers:4, 0:num_symbols:2]
        pilot_indices = np.vstack([pilot_rows.ravel(), pilot_cols.ravel()]).T
        grid_rows, grid_cols = np.mgrid[0:num_subcarriers, 0:num_symbols]
        grid_indices = np.vstack([grid_rows.ravel(), grid_cols.ravel()]).T

        # LS Baseline
        ls_preds = ls_estimation(eval_data, pilot_indices, grid_indices)
        ls_mse = mse(eval_labels, ls_preds)
        ls_ber = ber(metric, ls_preds, snr)
        results_df.loc[len(results_df)] = [snr, ls_mse, ls_ber, "LS", channel_name, np.nan]

        # LMMSE Baseline
        lmmse_preds = lmmse_estimation(ls_preds, eval_labels, snr)
        lmmse_mse = mse(eval_labels, lmmse_preds)
        lmmse_ber = ber(metric, lmmse_preds, snr)
        results_df.loc[len(results_df)] = [snr, lmmse_mse, lmmse_ber, "LMMSE", channel_name, np.nan]
        
        # MAML evaluation
        for n_shot in args.k_qry:
            maml_model = Meta(args, config).to(device)
            checkpoint_path = os.path.join(
                args.save_init, f"meta_model_nway_{args.n_way}",
                f"MAML_{n_shot}_shot_fine_tuned_model_{channel_name}_lr{args.update_lr}.pth"
            )

            if not os.path.exists(checkpoint_path):
                print(f"Skipping MAML {n_shot}-shot for {channel_name}, file not found.")
                continue

            maml_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
            
            _, predictions = maml_model.evaluate(eval_data.permute(0, 3, 1, 2), eval_labels.permute(0, 3, 1, 2), args.batchsz, device)
            
            current_mse = mse(eval_labels.permute(0, 3, 1, 2).cpu(), predictions.cpu())
            current_ber = ber(metric, predictions.cpu(), snr)
            
            results_df.loc[len(results_df)] = [snr, current_mse, current_ber, f"MAML", channel_name, n_shot]

    print("\n--- Evaluation Summary ---")
    print(results_df)
    results_df.to_csv(os.path.join(args.save_init, "evaluation_summary.csv"), index=False)
    print(f"\nResults saved to {os.path.join(args.save_init, 'evaluation_summary.csv')}")

    # Plot results
    plot_evaluation_results(results_df, args.save_init)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='path to processed_data dir', default="new_data")
    argparser.add_argument('--device', type=str, help='device to run the process', default='cuda:0')
    argparser.add_argument('--save_init', type=str, help='path to save directory', default="results")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3000)
    argparser.add_argument('--n_way', type=int, help='n way', default=7)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, nargs='+', help='k shot for query set, can be multiple', default=[5, 15])
    argparser.add_argument('--batchsz', type=int, help='meta batch size', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--seed', type=int, help='random seed', default=222)
    args = argparser.parse_args()
    main(args)
