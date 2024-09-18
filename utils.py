import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns 

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def visualize(input, output, num_samples=5, sub_title_1="sub_title_1", sub_title_2="sub_title_2", path="visualizations", fig_path="fig_path", title="Sample Visualization"):
    save_dir = os.path.dirname(path)
    fig_path = os.path.join(save_dir, fig_path)

    fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(16, 4*num_samples))
    fig.suptitle(title, fontsize=12)
    
    cmap = sns.color_palette("viridis", as_cmap=True)

    for i in range(num_samples):
        # Plot real part
        real_sample_input = input[0, i, 0, :, :]
        real_sample_output = output[0, i, 0, :, :]
        imag_sample_input = input[0, i, 1, :, :]
        imag_sample_output = output[0, i, 1, :, :]

        sns.heatmap(real_sample_input, ax=axes[i, 0], cmap=cmap, cbar=False)
        axes[i, 0].set_title(sub_title_1 + f'Real Part {i + 1}')
        axes[i, 0].axis('off')

        sns.heatmap(real_sample_output, ax=axes[i, 1], cmap=cmap, cbar=False)
        axes[i, 1].set_title(sub_title_2 + f'Real Part {i + 1}')
        axes[i, 1].axis('off')

        sns.heatmap(imag_sample_input, ax=axes[i, 2], cmap=cmap, cbar=False)
        axes[i, 2].set_title(sub_title_1 + f'Imag Part {i + 1}')
        axes[i, 2].axis('off')

        sns.heatmap(imag_sample_output, ax=axes[i, 3], cmap=cmap, cbar=False)
        axes[i, 3].set_title(sub_title_2 + f'Imag Part {i + 1}')
        axes[i, 3].axis('off')

    # Save the figure
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

       