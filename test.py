import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from metrics import Metric
import pdb

def calculate_and_save_ber(metric, predictions, snrs, modulation, output_path):
    bers = []
    for snr in snrs:
        ber = metric.bit_error_rate(predictions, snr, modulation=modulation)
        bers.append(ber)
    np.save(output_path, bers)
    return bers


def plot_ber_comparison(snrs, results, channel_name, output_dir, epoch):
    plt.figure(figsize=(8, 6))
    for method, data in results.items():
        plt.plot(snrs, data, marker='o', label=f"BER for {method}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"BER Comparison for Channel: {channel_name}")
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(output_dir, f"ber_comparison_plot_{channel_name}_epoch{epoch}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_2d_subplot(predictions, output_path, sample_idx=0):
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError("Predictions must have shape (226, 612, 14, 2).")
    sample = predictions[sample_idx]
    real_component = sample[..., 0]
    imaginary_component = sample[..., 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(real_component, cmap='viridis', aspect='auto')
    axes[0].set_title(f"Real Component - Sample {sample_idx}")
    axes[0].set_xlabel("Dimension 14")
    axes[0].set_ylabel("Dimension 612")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(imaginary_component, cmap='plasma', aspect='auto')
    axes[1].set_title(f"Imaginary Component - Sample {sample_idx}")
    axes[1].set_xlabel("Dimension 14")
    axes[1].set_ylabel("Dimension 612")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved subplot at: {output_path}")


def evaluation_result(args):
    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    fine_tune_file_names = list(data_dict.keys())[10:]
    n_shots = [5, 10, 15]
    snrs = [-5, 0, 5, 10, 20, 30]
    metric = Metric()

    for outer_channel_name in fine_tune_file_names:
        results = {}
        channel_output_dir = os.path.join(args.root, "results",f"meta_model_nway_{args.n_way}", outer_channel_name)
        os.makedirs(channel_output_dir, exist_ok=True)

        # LS
        LS = data_dict[outer_channel_name]
        LS_eval = LS[30:]
        ls_bers = calculate_and_save_ber(metric, LS_eval, snrs, '16QAM',
                                         os.path.join(channel_output_dir, f"LS_BER.npy"))
        results[f"LS_Estimator"] = ls_bers
        plot_prediction_2d_subplot(LS, os.path.join(channel_output_dir, f"LS_plot.png"), sample_idx=0)

        # MAML
        for n_shot in n_shots:
            maml_path = os.path.join(args.save_init, f"MAML_{n_shot}_shot_{outer_channel_name}_predictions.npy")
            if not os.path.exists(maml_path):
                print(f"Skipping missing file: {maml_path}")
                continue

            maml_predictions = np.load(maml_path)
            # pdb.set_trace()
            maml_bers = calculate_and_save_ber(metric, maml_predictions, snrs, '16QAM',
                                               os.path.join(channel_output_dir, f"MAML_{n_shot}_BER.npy"))
            plot_prediction_2d_subplot(maml_predictions.transpose(0,2,3,1), os.path.join(channel_output_dir, f"MAML_{n_shot}_plot.png"),
                                        sample_idx=0)
            results[f"MAML_{n_shot}_shot"] = maml_bers

            # ChannelNet
            channel_net_path = os.path.join(args.save_init, f"{n_shot}shot_{outer_channel_name}_DNCNN_predictions.npy")
            if os.path.exists(channel_net_path):
                channel_net_predictions = np.load(channel_net_path)
                channel_net_bers = calculate_and_save_ber(metric, channel_net_predictions, snrs, '16QAM',
                                                        os.path.join(channel_output_dir, f"ChannelNet_{n_shot}shot_BER.npy"))
                plot_prediction_2d_subplot(channel_net_predictions,
                                            os.path.join(channel_output_dir, f"ChannelNet_{n_shot}shot_plot.png"), sample_idx=0)
                results[f"ChannelNet{n_shot}shot"] = channel_net_bers

        # Plot BER comparison
        plot_ber_comparison(snrs, results, outer_channel_name, channel_output_dir, args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="new_data")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--save_init', type=str, default="saved_init")
    parser.add_argument('--n_way', type=int, default=5)   
    parser.add_argument('--epoch', type=int, default=500)
    args = parser.parse_args()

    evaluation_result(args)


