#!/usr/bin/env python3
"""
Plot predictions from MAML, iMAML, ChannelNet, and multigrade-MAML models
for both TDL and UMi experiments.

For a given channel model (e.g., MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A),
this script will find and plot predictions from all four models.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from pathlib import Path
import pdb
from paths import (
    default_dataset_umi_pspacing,
    default_prediction_plots_dir,
    default_tdl_init_dir,
    model_paths_plot,
)

# Set random seed for reproducibility (optional)
np.random.seed(42)
#TOD: plot GT disribution


# Define paths for each model (portable; see paths.py / README)
MODEL_PATHS = model_paths_plot()

# File naming patterns for each model
FILE_PATTERNS = {
    'TDL': {
        # 'MAML': '*MAML_5_shot_train_data_{channel_model}_predictions.npy',
        'iMAML': '*wireless_IMAML_5_shot_LAM3.0_train_data_{channel_model}_predictions.npy',
        'ChannelNet': '*5shot_train_data_{channel_model}_DNCNN_predictions.npy',
        'MultigradeMAML': '*MultigradeMAML_5shot_train_data_{channel_model}_predictions.npy',
        # 'tiny_MAML': '*MAML_5_shot_train_data_{channel_model}_predictions.npy',
        # 'SNR_MAML': '*MAML_5_shot_train_data_{channel_model}_predictions.npy',
        'CG_MAML': '*MAML_5_shot_train_data_{channel_model}_predictions.npy',
    },
    'UMi': {
        # 'MAML': '*MAML_5_shot_data_snr5.hdf5_predictions.npy',
        'iMAML': '*wireless_IMAML_5_shot_LAM3.0_data_snr5.hdf5_predictions.npy',
        'ChannelNet': '*5shot_data_snr5.hdf5_DNCNN_predictions.npy',
        'MultigradeMAML': '*MultigradeMAML_5shot_data_snr5.hdf5_predictions.npy',
        # 'tiny_MAML': '*MAML_5_shot_data_snr5.hdf5_predictions.npy'
    }
}


def find_prediction_file(channel_type, model_name, channel_model=None):
    """
    Find prediction file for a given model and channel model.
    
    Args:
        channel_type: 'TDL' or 'UMi'
        model_name: 'MAML', 'iMAML', 'ChannelNet', 'MultigradeMAML', 'tiny_MAML', or 'SNR_MAML'
        channel_model: Channel model identifier (e.g., 'MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A')
                       Required for TDL, optional for UMi
    
    Returns:
        Path to prediction file or None if not found
    """
    base_path = MODEL_PATHS[channel_type][model_name]
    
    if not os.path.exists(base_path):
        return None
    
    if channel_type == 'TDL':
        if channel_model is None:
            raise ValueError("channel_model is required for TDL experiments")
        pattern = FILE_PATTERNS[channel_type][model_name].format(channel_model=channel_model)
        # pdb.set_trace()
    else:  # UMi
        pattern = FILE_PATTERNS[channel_type][model_name]
    
    # Search for files matching the pattern
    search_path = os.path.join(base_path, pattern)
    matches = glob.glob(search_path)
    # pdb.set_trace()
    
    if matches:
        # For UMi ChannelNet, prefer 5shot over 15shot
        if channel_type == 'UMi' and model_name == 'ChannelNet':
            # First try to find 5shot file specifically (not 15shot)
            # Filter matches to only include files with "5shot" but not "15shot"
            five_shot_matches = [m for m in matches if '5shot' in os.path.basename(m) and '15shot' not in os.path.basename(m)]
            if five_shot_matches:
                return five_shot_matches[0]
            # Fall back to any match if 5shot not found
        return matches[0]
    
    # Try alternative search if exact match not found
    # For TDL, try without the train_data prefix variations
    if channel_type == 'TDL':
        # Try with just the channel model part
        alt_pattern = f'*{channel_model}*predictions.npy'
        alt_search_path = os.path.join(base_path, alt_pattern)
        matches = glob.glob(alt_search_path)
        if matches:
            return matches[0]
    
    return None


def load_predictions(file_path):
    """Load predictions from numpy file."""
    if file_path is None or not os.path.exists(file_path):
        return None
    try:
        predictions = np.load(file_path)
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_ground_truth_file(channel_type, channel_model=None):
    """Find ground truth (channel_label_dict.npy) file for a given channel model."""
    if channel_type == 'TDL':
        gt_path = os.path.join(default_tdl_init_dir(), 'channel_label_dict.npy')
        if os.path.exists(gt_path):
            return gt_path
    else:  # UMi
        gt_path = os.path.join(default_dataset_umi_pspacing(), 'channel_label_dict.npy')
        if os.path.exists(gt_path):
            return gt_path
    return None


def find_input_from_channel_data_dict(channel_type, channel_model=None, root_dir=None, umi_root_dir=None):
    """
    Load input data from channel_data_dict.npy for a given channel model.
    The input is sliced from index 30 onwards to match evaluation samples.
    
    Args:
        channel_type: 'TDL' or 'UMi'
        channel_model: Channel model identifier (required for TDL)
        root_dir: Root directory containing channel_data_dict.npy (optional, uses default paths if None)
        umi_root_dir: UMi root directory (optional, uses default path if None)
    
    Returns:
        Input data array (N, H, W, 2) or None if not found
    """
    # Use specific paths provided by user
    if channel_type == 'TDL':
        channel_data_dict_path = os.path.join(default_tdl_init_dir(), 'channel_data_dict.npy')
    else:  # UMi
        channel_data_dict_path = os.path.join(default_dataset_umi_pspacing(), 'channel_data_dict.npy')
    
    # Allow override with root_dir parameter
    if root_dir is not None:
        channel_data_dict_path = os.path.join(root_dir, "channel_data_dict.npy")
    elif umi_root_dir is not None and channel_type == 'UMi':
        channel_data_dict_path = os.path.join(umi_root_dir, "channel_data_dict.npy")
    
    if not os.path.exists(channel_data_dict_path):
        return None
    
    try:
        x_dict = np.load(channel_data_dict_path, allow_pickle=True).item()
        
        if channel_type == 'TDL':
            if channel_model is None:
                return None
            # Extract channel model name from the full identifier
            # Keys in dict have "train_data_" prefix, e.g., "train_data_MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A.mat"
            # channel_model is "MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A.mat"
            channel_key = None
            
            # First try exact match with "train_data_" prefix
            train_data_key = f"train_data_{channel_model}"
            if train_data_key in x_dict:
                channel_key = train_data_key
            else:
                # Try without .mat extension
                channel_model_no_ext = channel_model.replace('.mat', '')
                train_data_key_no_ext = f"train_data_{channel_model_no_ext}"
                if train_data_key_no_ext in x_dict:
                    channel_key = train_data_key_no_ext
                else:
                    # Try to find by matching the unique part (e.g., TDL-A, TDL-C, etc.)
                    for key in x_dict.keys():
                        # Extract the unique part (e.g., TDL-A, TDL-C, etc.)
                        if 'TDL-A' in channel_model and 'TDL-A' in key:
                            channel_key = key
                            break
                        elif 'TDL-C' in channel_model and 'TDL-C' in key:
                            channel_key = key
                            break
                        elif 'TDL-D' in channel_model and 'TDL-D' in key:
                            channel_key = key
                            break
                        elif 'TDL-E' in channel_model and 'TDL-E' in key:
                            channel_key = key
                            break
            
            if channel_key is None:
                return None
            
            input_data = x_dict[channel_key]
            # pdb.set_trace()
            # Slice from index 30 onwards
            if input_data.ndim == 4:
                input_data = input_data[30:]
            else:
                input_data = input_data[30:]
            
            return input_data
        else:  # UMi
            # For UMi, typically use the last key
            keys = list(x_dict.keys())
            if not keys:
                return None
            channel_key = keys[-1]
            input_data = x_dict[channel_key]
            # Slice from index 30 onwards
            if input_data.ndim == 4:
                input_data = input_data[30:]
            else:
                input_data = input_data[30:]
            
            return input_data
    except Exception as e:
        print(f"Error loading input from channel_data_dict.npy: {e}")
        return None


def per_sample_mse_split(pred_cl, gt_cl):
    """
    Returns per-sample MSEs:
      total: E[(Re)^2 + (Im)^2] over (H,W)
      real : E[(Re)^2]           over (H,W)
      imag : E[(Im)^2]           over (H,W)
    Shapes: pred_cl, gt_cl = (N,H,W,2)
    """
    if pred_cl is None or gt_cl is None:
        return None, None, None
    diff_r = pred_cl[..., 0] - gt_cl[..., 0]
    diff_i = pred_cl[..., 1] - gt_cl[..., 1]
    mse_r  = np.mean(diff_r * diff_r, axis=(1, 2))
    mse_i  = np.mean(diff_i * diff_i, axis=(1, 2))
    mse    = mse_r + mse_i
    
    return mse, mse_r, mse_i


def _imshow(ax, arr2d, vmin, vmax, title=None):
    """Plot 2D array with style matching evaluation.py"""
    im = ax.imshow(arr2d.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return im


def _annotate_mse(ax, val, where="top-left"):
    """Overlay readable MSE text inside the axes."""
    if val is None:
        return
    # choose corner
    if where == "top-left":
        xy = (0.02, 0.98); va = "top"; ha = "left"
    elif where == "bottom-left":
        xy = (0.02, 0.02); va = "bottom"; ha = "left"
    else:
        xy = (0.98, 0.98); va = "top"; ha = "right"
    ax.text(
        xy[0], xy[1], f"MSE={val:.2e}",
        transform=ax.transAxes, ha=ha, va=va, fontsize=8,
        color="w", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none")
    )


def discover_channel_models(channel_type):
    """
    Discover all available channel models for a given channel type.
    
    Args:
        channel_type: 'TDL' or 'UMi'
    
    Returns:
        List of channel model identifiers (for TDL) or None (for UMi)
    """
    if channel_type == 'TDL':
        # Find all TDL channel models by looking at MAML prediction files
        maml_path = MODEL_PATHS['TDL']['CG_MAML']
        if not os.path.exists(maml_path):
            return []
        
        files = glob.glob(os.path.join(maml_path, '*predictions.npy'))
        channel_models = set()
        
        for f in files:
            basename = os.path.basename(f)
            # Extract channel model from filename
            # Pattern: MAML_5_shot_train_data_CHANNEL_MODEL_predictions.npy
            if 'train_data_' in basename:
                parts = basename.split('train_data_')
                if len(parts) > 1:
                    channel = parts[1].split('_predictions')[0]
                    channel_models.add(channel)
        
        return sorted(list(channel_models))
    
    else:  # UMi
        # UMi has a single prediction file, so return None to indicate it's available
        return None


def plot_all_channel_models(channel_type, output_dir=None, root_dir=None, umi_root_dir=None):
    """
    Plot predictions for all available channel models.
    
    Args:
        channel_type: 'TDL' or 'UMi'
        output_dir: Directory to save plots (default: current directory)
    """
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    if channel_type == 'TDL':
        channel_models = discover_channel_models('TDL')
        if not channel_models:
            print(f"No TDL channel models found!")
            return
        
        print(f"Found {len(channel_models)} TDL channel models:")
        for ch in channel_models:
            print(f"  - {ch}")
        
        print(f"\nPlotting predictions for all {len(channel_models)} channel models...\n")
        
        for i, channel_model in enumerate(channel_models, 1):
            print(f"\n[{i}/{len(channel_models)}] Processing: {channel_model}")
            safe_name = channel_model.replace('/', '_').replace('\\', '_').replace('.mat', '')
            save_path = os.path.join(output_dir, f'predictions_comparison_{channel_type}_{safe_name}.png')
            plot_predictions(
                channel_type=channel_type,
                channel_model=channel_model,
                save_path=save_path,
                root_dir=root_dir,
                umi_root_dir=umi_root_dir
            )
        
        print(f"\n✓ All plots saved to: {output_dir}")
    
    else:  # UMi
        print("Plotting predictions for UMi...\n")
        save_path = os.path.join(output_dir, f'predictions_comparison_{channel_type}.png')
        plot_predictions(
            channel_type=channel_type,
            channel_model=None,
            save_path=save_path,
            root_dir=root_dir,
            umi_root_dir=umi_root_dir
        )
        print(f"\n✓ Plot saved to: {save_path}")


def plot_predictions(channel_type, channel_model=None, save_path=None, root_dir=None, umi_root_dir=None):
    """
    Plot predictions from all four models for a given channel model.
    Follows the plotting style from evaluation.py.
    
    Args:
        channel_type: 'TDL' or 'UMi'
        channel_model: Channel model identifier (required for TDL)
        num_samples: Number of samples to plot (for visualization)
        save_path: Path to save the figure
    """
    # Define model names based on channel type
    if channel_type == 'TDL':
        model_names = ['iMAML', 'ChannelNet', 'MultigradeMAML',  "CG_MAML", ]
    else:  # UMi
        model_names = ['iMAML', 'ChannelNet', 'MultigradeMAML',]
    
    # Find all prediction files
    prediction_files = {}
    predictions_data = {}
    
    for model in model_names:
        file_path = find_prediction_file(channel_type, model, channel_model)
        prediction_files[model] = file_path
        
        if file_path:
            print(f"Found {model} predictions: {file_path}")
            pred_data = load_predictions(file_path)
            if pred_data is not None:
                predictions_data[model] = pred_data
            else:
                print(f"Warning: Could not load predictions for {model}")
        else:
            print(f"Warning: Could not find prediction file for {model}")
    
    if not predictions_data:
        print("Error: No prediction files found!")
        return
    
    # Determine the shape of predictions
    first_model = list(predictions_data.keys())[0]
    first_pred = predictions_data[first_model]
    
    # Load ground truth from channel_label_dict.npy
    gt_file = find_ground_truth_file(channel_type, channel_model)
    gt_data = None
    if gt_file:
        print(f"Found ground truth: {gt_file}")
        try:
            # Load the dictionary
            gt_dict = np.load(gt_file, allow_pickle=True).item()
            
            if channel_type == 'TDL':
                if channel_model is None:
                    print("Warning: channel_model is required for TDL ground truth")
                else:
                    # Extract channel model name from the full identifier
                    # Keys in dict have "train_data_" prefix, e.g., "train_data_MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A.mat"
                    # channel_model is "MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A.mat"
                    channel_key = None
                    
                    # First try exact match with "train_data_" prefix
                    train_data_key = f"train_data_{channel_model}"
                    if train_data_key in gt_dict:
                        channel_key = train_data_key
                    else:
                        # Try without .mat extension
                        channel_model_no_ext = channel_model.replace('.mat', '')
                        train_data_key_no_ext = f"train_data_{channel_model_no_ext}"
                        if train_data_key_no_ext in gt_dict:
                            channel_key = train_data_key_no_ext
                        else:
                            # Try to find by matching the unique part (e.g., TDL-A, TDL-C, etc.)
                            for key in gt_dict.keys():
                                if 'TDL-A' in channel_model and 'TDL-A' in key:
                                    channel_key = key
                                    break
                                elif 'TDL-C' in channel_model and 'TDL-C' in key:
                                    channel_key = key
                                    break
                                elif 'TDL-D' in channel_model and 'TDL-D' in key:
                                    channel_key = key
                                    break
                                elif 'TDL-E' in channel_model and 'TDL-E' in key:
                                    channel_key = key
                                    break
                    
                    if channel_key is not None:
                        gt_data = gt_dict[channel_key]
                        print(f"Found GT data for key '{channel_key}' with shape {gt_data.shape}")
                        # Slice from index 30 onwards (same as input)
                        if gt_data.ndim == 4:
                            gt_data = gt_data[30:]
                        else:
                            gt_data = gt_data[30:]
                        print(f"GT data after slicing: shape {gt_data.shape}")
                    else:
                        print(f"Warning: Could not find channel model '{channel_model}' in ground truth dictionary")
                        print(f"Available keys in GT dictionary: {list(gt_dict.keys())[:5]}...")  # Show first 5 keys
            else:  # UMi
                # For UMi, typically use the last key
                keys = list(gt_dict.keys())
                if keys:
                    channel_key = keys[-1]
                    gt_data = gt_dict[channel_key]
                    # Slice from index 30 onwards (same as input)
                    if gt_data.ndim == 4:
                        gt_data = gt_data[30:]
                    else:
                        gt_data = gt_data[30:]
                else:
                    print("Warning: Ground truth dictionary is empty")
        except Exception as e:
            print(f"Error loading ground truth from {gt_file}: {e}")
            gt_data = None
    else:
        print("Warning: Could not find ground truth file")
    
    # Load input from channel_data_dict.npy (sliced from index 30 onwards)
    input_data = find_input_from_channel_data_dict(channel_type, channel_model, root_dir=root_dir, umi_root_dir=umi_root_dir)
    if input_data is not None:
        print(f"Found input data from channel_data_dict.npy (shape: {input_data.shape})")
        # Ensure input data has the same shape as predictions
        if input_data.shape != first_pred.shape:
            print(f"Warning: Input shape {input_data.shape} doesn't match predictions shape {first_pred.shape}")
            # Try to match the number of samples
            if input_data.shape[0] > first_pred.shape[0]:
                input_data = input_data[:first_pred.shape[0]]
            elif input_data.shape[0] < first_pred.shape[0]:
                print(f"Warning: Input has fewer samples ({input_data.shape[0]}) than predictions ({first_pred.shape[0]})")
    else:
        print("Warning: Could not find input data from channel_data_dict.npy")
    
    # Ensure GT data has the same shape as predictions (if available)
    if gt_data is not None:
        if gt_data.shape != first_pred.shape:
            print(f"Warning: GT shape {gt_data.shape} doesn't match predictions shape {first_pred.shape}")
            # Try to match the number of samples
            if gt_data.shape[0] > first_pred.shape[0]:
                gt_data = gt_data[:first_pred.shape[0]]
                print(f"GT data trimmed to match predictions: shape {gt_data.shape}")
            elif gt_data.shape[0] < first_pred.shape[0]:
                print(f"Warning: GT has fewer samples ({gt_data.shape[0]}) than predictions ({first_pred.shape[0]})")
        else:
            print(f"GT data shape matches predictions: {gt_data.shape}")
    
    # Check if data is complex (N, H, W, 2)
    if first_pred.ndim != 4 or first_pred.shape[-1] != 2:
        print(f"Error: Expected predictions with shape (N, H, W, 2), got {first_pred.shape}")
        return
    
    # Select a random sample for visualization
    n_samples = first_pred.shape[0]
    random_sample_idx = np.random.randint(0, n_samples)
    print(f"\nSelected random sample index: {random_sample_idx} (out of {n_samples} samples)")
    
    # Prepare arrays dict in the format expected by evaluation.py style plotting
    arrays_dict = {
        # "MAML": predictions_data.get("MAML"),
        "iMAML": predictions_data.get("iMAML"),
        "ChannelNet": predictions_data.get("ChannelNet"),
        "MultigradeMAML": predictions_data.get("MultigradeMAML"),
        # "tiny_MAML": predictions_data.get("tiny_MAML"),
        "CG_MAML": predictions_data.get("CG_MAML"),
        # "SNR_MAML": predictions_data.get("SNR_MAML"),
        "Input": input_data,
        "GT": gt_data,
    }
    
    # Compute MSEs for each model (if GT is available)
    mses_total = {}
    mses_real = {}
    mses_imag = {}
    if gt_data is not None:
        for name, arr in arrays_dict.items():
            if name == "GT" or arr is None:
                mses_total[name] = mses_real[name] = mses_imag[name] = None
            else:
                mt, mr, mi = per_sample_mse_split(arr, gt_data)
                mses_total[name] = mt[random_sample_idx]
                mses_real[name] = mr[random_sample_idx]
                mses_imag[name] = mi[random_sample_idx]
    
    # Determine color range from GT (if available) or from first available prediction
    if gt_data is not None:
        vmin_r = gt_data[random_sample_idx, ..., 0].min()
        vmax_r = gt_data[random_sample_idx, ..., 0].max()
        vmin_i = gt_data[random_sample_idx, ..., 1].min()
        vmax_i = gt_data[random_sample_idx, ..., 1].max()
    else:
        # Use first available prediction to set color range
        first_available = next(iter(predictions_data.values()))
        vmin_r = first_available[random_sample_idx, ..., 0].min()
        vmax_r = first_available[random_sample_idx, ..., 0].max()
        vmin_i = first_available[random_sample_idx, ..., 1].min()
        vmax_i = first_available[random_sample_idx, ..., 1].max()
    
    # Define column order: models + Input + GT
    if channel_type == 'TDL':
        cols = [ "iMAML", "ChannelNet", "MultigradeMAML", "CG_MAML", "Input", "GT"]
    else:  # UMi
        cols = ["iMAML", "ChannelNet", "MultigradeMAML", "Input", "GT"]
    arrs = [arrays_dict.get(k) for k in cols]
    
    # Create figure with main image plot (2 rows for real/imag) and distribution plots below
    n_cols = len([a for a in arrs if a is not None])
    # Create a figure with 3 rows: 2 for images + 1 for distribution
    fig = plt.figure(figsize=(3*n_cols, 10))
    
    # Top section: Image comparison (2 rows x n_cols)
    gs = fig.add_gridspec(3, n_cols, hspace=0.3, wspace=0.3)
    axes = np.zeros((2, n_cols), dtype=object)
    
    col_idx = 0
    for c, (name, arr) in enumerate(zip(cols, arrs)):
        if arr is None:
            continue
        
        # REAL row (row 0)
        axes[0, col_idx] = fig.add_subplot(gs[0, col_idx])
        im0 = _imshow(axes[0, col_idx], arr[random_sample_idx, ..., 0], vmin_r, vmax_r, title=name)
        # Show per-model real-part MSE in the top-left corner
        if name != "GT" and name != "Input" and mses_real.get(name) is not None:
            _annotate_mse(axes[0, col_idx], mses_real[name], where="top-left")
        
        # IMAG row (row 1)
        axes[1, col_idx] = fig.add_subplot(gs[1, col_idx])
        im1 = _imshow(axes[1, col_idx], arr[random_sample_idx, ..., 1], vmin_i, vmax_i)
        # Show per-model imag-part MSE in the bottom-left corner
        if name != "GT" and name != "Input" and mses_imag.get(name) is not None:
            _annotate_mse(axes[1, col_idx], mses_imag[name], where="bottom-left")
        
        # Colorbars ONLY for GT column (both rows) - shared scale based on GT
        if name == "GT":
            fig.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.01)
            fig.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.01)
        
        col_idx += 1
    
    # Bottom section: Distribution comparison (all samples)
    ax_dist = fig.add_subplot(gs[2, :])
    
    # Plot histograms for real parts (all samples)
    for model in model_names:
        if model not in predictions_data:
            continue
        pred = predictions_data[model]
        real_flat = pred[..., 0].flatten()
        ax_dist.hist(real_flat, bins=50, alpha=0.5, label=f'{model} (Real)', density=True)
    
    # Plot histograms for imaginary parts (all samples)
    for model in model_names:
        if model not in predictions_data:
            continue
        pred = predictions_data[model]
        imag_flat = pred[..., 1].flatten()
        ax_dist.hist(imag_flat, bins=50, alpha=0.5, label=f'{model} (Imag)', density=True, linestyle='--')
    
    # Add Input distribution if available
    if input_data is not None:
        input_real_flat = input_data[..., 0].flatten()
        input_imag_flat = input_data[..., 1].flatten()
        ax_dist.hist(input_real_flat, bins=50, alpha=0.5, label='Input (Real)', density=True, color='purple')
        ax_dist.hist(input_imag_flat, bins=50, alpha=0.5, label='Input (Imag)', density=True, linestyle='--', color='purple')
    
    # Add GT distribution if available
    if gt_data is not None:
        gt_real_flat = gt_data[..., 0].flatten()
        gt_imag_flat = gt_data[..., 1].flatten()
        ax_dist.hist(gt_real_flat, bins=50, alpha=0.7, label='GT (Real)', density=True, color='black', histtype='step', linewidth=2)
        ax_dist.hist(gt_imag_flat, bins=50, alpha=0.7, label='GT (Imag)', density=True, linestyle='--', color='black', histtype='step', linewidth=2)
    
    ax_dist.set_xlabel('Prediction Value')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('Prediction Distribution Comparison (All Samples)')
    ax_dist.legend(ncol=3, loc='upper right', fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    
    # Set main title
    title = f'Model Predictions Comparison - {channel_type}'
    if channel_model:
        title += f' — Sample {random_sample_idx} — {channel_model}'
    else:
        title += f' — Sample {random_sample_idx}'
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if save_path is None:
        if channel_model:
            safe_name = channel_model.replace('/', '_').replace('\\', '_').replace('.mat', '')
            save_path = f'predictions_comparison_{channel_type}_{safe_name}.png'
        else:
            save_path = f'predictions_comparison_{channel_type}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()
    
    # Print summary statistics for all samples
    print("\n=== Prediction Statistics (All Samples) ===")
    
    # Calculate and print MSE for all models and input
    mse_results = []
    if gt_data is not None:
        print("\n=== Overall MSE (Entire Test Set) ===")
        
        # Calculate MSE for all models
        for model in model_names:
            if model not in predictions_data:
                continue
            pred = predictions_data[model]
            mt, mr, mi = per_sample_mse_split(pred, gt_data)
            total_mse = np.mean(mt)
            real_mse = np.mean(mr)
            imag_mse = np.mean(mi)
            print(f"{model}:")
            print(f"  Total MSE: {total_mse:.6e}")
            print(f"  Real MSE:  {real_mse:.6e}")
            print(f"  Imag MSE:  {imag_mse:.6e}")
            mse_results.append({
                'Model': model,
                'Total_MSE': total_mse,
                'Real_MSE': real_mse,
                'Imag_MSE': imag_mse
            })
        
        # Calculate MSE for Input
        if input_data is not None:
            mt_input, mr_input, mi_input = per_sample_mse_split(input_data, gt_data)
            total_mse_input = np.mean(mt_input)
            real_mse_input = np.mean(mr_input)
            imag_mse_input = np.mean(mi_input)
            print(f"Input:")
            print(f"  Total MSE: {total_mse_input:.6e}")
            print(f"  Real MSE:  {real_mse_input:.6e}")
            print(f"  Imag MSE:  {imag_mse_input:.6e}")
            mse_results.append({
                'Model': 'Input',
                'Total_MSE': total_mse_input,
                'Real_MSE': real_mse_input,
                'Imag_MSE': imag_mse_input
            })
        
        # Save MSE results to CSV
        if save_path is not None:
            csv_path = save_path.replace('.png', '_mse.csv')
            # Ensure we save in the same directory as the plot
            csv_dir = os.path.dirname(save_path)
            if csv_dir:
                csv_path = os.path.join(csv_dir, os.path.basename(csv_path))
        else:
            if channel_model:
                safe_name = channel_model.replace('/', '_').replace('\\', '_').replace('.mat', '')
                csv_path = f'predictions_comparison_{channel_type}_{safe_name}_mse.csv'
            else:
                csv_path = f'predictions_comparison_{channel_type}_mse.csv'
        
        # Write MSE results to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Model', 'Total_MSE', 'Real_MSE', 'Imag_MSE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in mse_results:
                writer.writerow(row)
        
        print(f"\n✓ MSE results saved to: {csv_path}")
        
        print("\n=== Detailed Statistics ===")
    
    for model in model_names:
        if model not in predictions_data:
            continue
        pred = predictions_data[model]
        real_flat = pred[..., 0].flatten()
        imag_flat = pred[..., 1].flatten()
        complex_flat = (pred[..., 0] + 1j * pred[..., 1]).flatten()
        
        print(f"\n{model}:")
        print(f"  Shape: {pred.shape}")
        print(f"  Real Part:")
        print(f"    Mean: {np.mean(real_flat):.6f}")
        print(f"    Std:  {np.std(real_flat):.6f}")
        print(f"    Min:  {np.min(real_flat):.6f}")
        print(f"    Max:  {np.max(real_flat):.6f}")
        print(f"  Imaginary Part:")
        print(f"    Mean: {np.mean(imag_flat):.6f}")
        print(f"    Std:  {np.std(imag_flat):.6f}")
        print(f"    Min:  {np.min(imag_flat):.6f}")
        print(f"    Max:  {np.max(imag_flat):.6f}")
    
    # Print Input statistics
    if input_data is not None:
        input_real_flat = input_data[..., 0].flatten()
        input_imag_flat = input_data[..., 1].flatten()
        print(f"\nInput:")
        print(f"  Shape: {input_data.shape}")
        print(f"  Real Part:")
        print(f"    Mean: {np.mean(input_real_flat):.6f}")
        print(f"    Std:  {np.std(input_real_flat):.6f}")
        print(f"    Min:  {np.min(input_real_flat):.6f}")
        print(f"    Max:  {np.max(input_real_flat):.6f}")
        print(f"  Imaginary Part:")
        print(f"    Mean: {np.mean(input_imag_flat):.6f}")
        print(f"    Std:  {np.std(input_imag_flat):.6f}")
        print(f"    Min:  {np.min(input_imag_flat):.6f}")
        print(f"    Max:  {np.max(input_imag_flat):.6f}")
    
    # Print GT statistics
    if gt_data is not None:
        gt_real_flat = gt_data[..., 0].flatten()
        gt_imag_flat = gt_data[..., 1].flatten()
        print(f"\nGround Truth:")
        print(f"  Shape: {gt_data.shape}")
        print(f"  Real Part:")
        print(f"    Mean: {np.mean(gt_real_flat):.6f}")
        print(f"    Std:  {np.std(gt_real_flat):.6f}")
        print(f"    Min:  {np.min(gt_real_flat):.6f}")
        print(f"    Max:  {np.max(gt_real_flat):.6f}")
        print(f"  Imaginary Part:")
        print(f"    Mean: {np.mean(gt_imag_flat):.6f}")
        print(f"    Std:  {np.std(gt_imag_flat):.6f}")
        print(f"    Min:  {np.min(gt_imag_flat):.6f}")
        print(f"    Max:  {np.max(gt_imag_flat):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot predictions from MAML, iMAML, ChannelNet, and multigrade-MAML models'
    )
    parser.add_argument(
        '--channel_type',
        type=str,
        required=False,
        choices=['TDL', 'UMi'],
        help='Channel type: TDL or UMi'
    )
    parser.add_argument(
        '--channel_model',
        type=str,
        default=None,
        help='Channel model identifier (required for TDL, e.g., MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to plot (default: 100)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save the figure (default: auto-generated)'
    )
    parser.add_argument(
        '--list_available',
        action='store_true',
        help='List available channel models and exit'
    )
    parser.add_argument(
        '--plot_all',
        action='store_true',
        help='Plot all available channel models for TDL and UMi'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_prediction_plots_dir(),
        help='Directory to save plots when using --plot_all (default: current directory)'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=default_tdl_init_dir(),
        help='Root directory containing channel_data_dict.npy (default: auto-detect)'
    )
    
    parser.add_argument(
        '--umi_root_dir',
        type=str,
        default=default_dataset_umi_pspacing(),
        help='Root directory containing channel_data_dict.npy (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    if args.list_available:
        print("=== Available TDL Channel Models ===")
        channel_models = discover_channel_models('TDL')
        if channel_models:
            for ch in channel_models:
                print(f"  {ch}")
        else:
            print("  No TDL channel models found")
        
        print("\n=== Available UMi Predictions ===")
        for model in MODEL_PATHS['UMi'].keys():
            path = MODEL_PATHS['UMi'][model]
            if os.path.exists(path):
                files = glob.glob(os.path.join(path, '*predictions.npy'))
                if files:
                    print(f"  {model}: {len(files)} file(s) found")
                    for f in files[:5]:  # Show first 5
                        print(f"    {os.path.basename(f)}")
        return
    
    # Plot all channel models if requested
    if args.plot_all:
        print("=" * 60)
        print("Plotting all available channel models for TDL and UMi")
        print("=" * 60)
        
        # Plot all TDL channel models
        print("\n" + "=" * 60)
        print("TDL Channel Models")
        print("=" * 60)
        plot_all_channel_models(
            channel_type='TDL',
            output_dir=args.output_dir,
            root_dir=args.root_dir,
            umi_root_dir=args.umi_root_dir
        )
        
        # Plot UMi
        print("\n" + "=" * 60)
        print("UMi Channel Model")
        print("=" * 60)
        plot_all_channel_models(
            channel_type='UMi',
            output_dir=args.output_dir,
            root_dir=args.umi_root_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ All plots completed!")
        print("=" * 60)
        return
    
    # Single channel model plotting
    if args.channel_type is None:
        parser.error("--channel_type is required (use --list_available to see available options or --plot_all to plot all)")
    
    if args.channel_type == 'TDL' and args.channel_model is None:
        parser.error("--channel_model is required for TDL experiments (use --plot_all to plot all channel models)")
    
    plot_predictions(
        channel_type=args.channel_type,
        channel_model=args.channel_model,
        save_path=args.save_path,
        root_dir=args.root_dir,
        umi_root_dir=args.umi_root_dir
    )


if __name__ == '__main__':
    main()

