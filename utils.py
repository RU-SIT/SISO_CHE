import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Utils:
    @staticmethod
    def save_checkpoint(state, filename="checkpoint.pth.tar"):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    @staticmethod
    def visualize(input, output, num_samples=5, sub_title_1="sub_title_1", sub_title_2="sub_title_2", path="visualizations", fig_path="fig_path", title="Sample Visualization"):
        save_dir = os.path.dirname(path)
        fig_path = os.path.join(save_dir, fig_path)
        
        fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(16, 4 * num_samples))
        fig.suptitle(title, fontsize=12)
        
        cmap = sns.color_palette("viridis", as_cmap=True)

        for i in range(num_samples):
            # Plot real part
            real_sample_input = input[0, i, 0, :, :]
            real_sample_output = output[0, i, 0, :, :]
            imag_sample_input = input[0, i, 1, :, :]
            imag_sample_output = output[0, i, 1, :, :]

            sns.heatmap(real_sample_input, ax=axes[i, 0], cmap=cmap, cbar=False)
            axes[i, 0].set_title(sub_title_1 + f' Real Part {i + 1}')
            axes[i, 0].axis('off')

            sns.heatmap(real_sample_output, ax=axes[i, 1], cmap=cmap, cbar=False)
            axes[i, 1].set_title(sub_title_2 + f' Real Part {i + 1}')
            axes[i, 1].axis('off')

            sns.heatmap(imag_sample_input, ax=axes[i, 2], cmap=cmap, cbar=False)
            axes[i, 2].set_title(sub_title_1 + f' Imag Part {i + 1}')
            axes[i, 2].axis('off')

            sns.heatmap(imag_sample_output, ax=axes[i, 3], cmap=cmap, cbar=False)
            axes[i, 3].set_title(sub_title_2 + f' Imag Part {i + 1}')
            axes[i, 3].axis('off')

        # Save the figure
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved to {fig_path}")

    @staticmethod
    def mmse_channel_estimation(received_signal, transmitted_signal, noise_variance, channel_correlation):
        """
        Perform MMSE channel estimation in a SISO system.

        Parameters:
        - received_signal (torch.Tensor): The received signal (y), of shape [batch_size, signal_length]
        - transmitted_signal (torch.Tensor): The transmitted signal (x), of shape [batch_size, signal_length]
        - noise_variance (float): The variance of the noise (sigma_n^2)
        - channel_correlation (torch.Tensor): The channel autocorrelation (R_h), shape [batch_size, signal_length, signal_length]

        Returns:
        - h_mmse (torch.Tensor): The estimated channel using MMSE, shape [batch_size, signal_length]
        """
        # Transmit signal conjugate transpose (Hermitian)
        transmitted_signal_H = transmitted_signal.conj().transpose(-1, -2)  # Hermitian of transmitted signal

        # Calculate the denominator term: (x^H * R_h * x + sigma_n^2)
        denom = torch.matmul(torch.matmul(transmitted_signal_H, channel_correlation), transmitted_signal)
        denom = denom + noise_variance

        # Calculate the numerator term: R_h * x^H
        numerator = torch.matmul(channel_correlation, transmitted_signal_H)

        # MMSE estimation: h_mmse = (numerator) / denom
        h_mmse = torch.matmul(numerator, received_signal) / denom

        return h_mmse

    @staticmethod
    def unit_scaling(data, label):
        """
        Scale each real and imaginary component pixelwise.
        :param data: Data to be scaled.
        :param label: Labels to be scaled.
        :return: Scaled data and labels.
        """
        data = np.array(data)
        label = np.array(label)

        denom = np.sqrt(np.sum(data ** 2, axis=-1, keepdims=True)) + 1e-8  # Avoid division by zero
        scaled_data = data / denom
        scaled_label = label / denom

        return scaled_data, scaled_label, denom
        
    @staticmethod
    def trch_unit_scaling(data, label):
        """
        Scale each real and imaginary component pixelwise using PyTorch tensors.
        :param data: Data to be scaled (torch.Tensor).
        :param label: Labels to be scaled (torch.Tensor).
        :return: Scaled data, scaled labels, and denominator used for scaling.
        """
        # Ensure inputs are tensors
        data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        label = torch.tensor(label, dtype=torch.float32) if not isinstance(label, torch.Tensor) else label

        # Compute denominator for scaling
        denom = torch.sqrt(torch.sum(data ** 2, dim=-1, keepdim=True)) + 1e-8  # Avoid division by zero
        scaled_data = data / denom
        scaled_label = label / denom

        return scaled_data, scaled_label, denom

    def save_unique_samples(self, save_path):
        """
        Save the unique file-to-sample mappings to a file.
        """
        np.save(os.path.join(save_path, "unique_file_samples.npy"), self.unique_file_samples)