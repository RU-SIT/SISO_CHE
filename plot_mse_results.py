#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_csv(path):
    """
    Reads CSV with header: SNR_dB,Model,MSE
    Returns: dict mapping Model->(snr_list, mse_list)
    """
    data = defaultdict(lambda: {"snrs": [], "mses": []})
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = float(row["SNR_dB"])
            model = row["Model"]
            mse = float(row["MSE"])
            data[model]["snrs"].append(snr)
            data[model]["mses"].append(mse)
    # ensure each model is sorted by increasing SNR
    out = {}
    for model, dat in data.items():
        pairs = sorted(zip(dat["snrs"], dat["mses"]))
        snrs_sorted, mses_sorted = zip(*pairs)
        out[model] = (np.array(snrs_sorted), np.array(mses_sorted))
    return out

def main(root_dir, channel, shots, out_file):
    fig, axes = plt.subplots(1, len(shots), figsize=(5*len(shots), 4), sharey=True)
    
    for ax, k in zip(axes, shots):
        csv_path = os.path.join(root_dir, f"{channel}_{k}shot_results.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        results = read_csv(csv_path)
        for model, (snrs, mses) in results.items():
            ax.semilogy(snrs, mses, marker="o", label=model)
        ax.set_title(f"{k}-shot")
        ax.set_xlabel("SNR (dB)")
        ax.grid(which="both", linestyle="--")
        if ax is axes[0]:
            ax.set_ylabel("MSE")
        ax.legend(loc="best")
    
    plt.suptitle(f"MSE vs SNR ({channel.upper()})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved figure to {out_file}")
    plt.show()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default=".", 
                   help="Directory containing the CSV files")
    p.add_argument("--channel", type=str, required=True,
                   help="Channel prefix in CSV filenames (e.g. 'tdl' or 'umi')")
    p.add_argument("--shots", nargs="+", type=int, default=[5,10,15],
                   help="List of shot values to plot (e.g. 5 10 15)")
    p.add_argument("--out_file", type=str, default="mse_vs_snr.png",
                   help="Filename for the saved figure")
    args = p.parse_args()
    main(args.root_dir, args.channel, args.shots, args.out_file)
