import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mse_vs_snr(csv_file):
    # Read the CSV file into a DataFrame
    results = pd.read_csv(csv_file)
    
    # Extract the "train_output_..." folder from the file path for the title
    parts = os.path.normpath(csv_file).split(os.sep)
    train_folder = next((part for part in parts if "train_output" in part), "Unknown")
    
    # Remove "train_output_" prefix from the folder name
    train_folder = train_folder.replace("train_output_", "")
    
    # Define plot parameters
    markers = {"ChannelNet": "o", "LS": "s", "LMMSE": "^", "ALMMSE": "D"}
    order = ["ChannelNet", "LS", "LMMSE", "ALMMSE"]
    palette = sns.color_palette("tab10", len(order))
    x_values = results["snr"].unique()
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    sns.lineplot(
        x="snr",
        y="mse",
        hue="method",
        style="method",
        markers=markers,
        dashes=False,
        data=results,
        hue_order=order,
        palette=palette,
        ax=ax,
    )
    ax.set(yscale="log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_xticks(x_values)
    ax.set_xticklabels(list(map(int, x_values)))
    ax.set_xlim(int(min(x_values)), int(max(x_values)))
    
    # Automatically adjust y-axis limits to include all data
    ax.set_ylim(None, max(results["mse"]))
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=5, frameon=False)
    
    # Set plot title with the modified folder name
    plt.title(f"MSE vs SNR - {train_foldern}", loc="center", pad=20)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter if needed
    
    # Save the plot in the same directory as the CSV file
    output_dir = os.path.dirname(os.path.abspath(csv_file))
    output_file = os.path.join(output_dir, "mse_vs_snr_plot.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MSE vs SNR from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    args = parser.parse_args()
    plot_mse_vs_snr(args.csv_file)
