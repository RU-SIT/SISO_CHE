#!/usr/bin/env python3

import numpy as np
import os
import glob
import csv
import argparse

from paths import gt_paths_dict, input_paths_dict, model_paths_calculate_mse

# Model paths (portable; see paths.py)
MODEL_PATHS = model_paths_calculate_mse()
GT_PATHS = gt_paths_dict()
INPUT_PATHS = input_paths_dict()


def calculate_mse(pred, gt):
    """Calculate MSE: total, real, imaginary."""
    if pred is None or gt is None:
        return None, None, None
    
    # Ensure same shape
    min_samples = min(pred.shape[0], gt.shape[0])
    pred = pred[:min_samples]
    gt = gt[:min_samples]
    
    # Calculate differences
    diff_real = pred[..., 0] - gt[..., 0]
    diff_imag = pred[..., 1] - gt[..., 1]
    
    # MSE for real and imaginary parts
    mse_real = np.mean(diff_real ** 2)
    mse_imag = np.mean(diff_imag ** 2)
    mse_total = mse_real + mse_imag
    
    return mse_total, mse_real, mse_imag


def find_prediction_file(model_path, model_name, channel_model):
    """Find prediction file for a model and channel model."""
    # Search for files containing the channel model
    pattern = f'*{channel_model}*predictions.npy'
    files = glob.glob(os.path.join(model_path, pattern))
    
    if not files:
        return None
    
    # For ChannelNet, must have DNCNN in filename
    if model_name == 'ChannelNet':
        for f in files:
            if 'DNCNN' in os.path.basename(f):
                return f
        return None
    
    # For other models, exclude DNCNN files
    for f in files:
        if 'DNCNN' not in os.path.basename(f):
            return f
    
    return None


def get_channel_models_from_predictions(experiment_type):
    """Find all channel models that have predictions."""
    channel_models = set()
    
    for model_name, model_path in MODEL_PATHS[experiment_type].items():
        if not os.path.exists(model_path):
            continue
        
        # Find all prediction files
        pattern = '*predictions.npy'
        files = glob.glob(os.path.join(model_path, pattern))
        
        for file in files:
            basename = os.path.basename(file)
            
            if experiment_type == 'TDL':
                # TDL format: ...train_data_MAXDopS_50_DS_3e-7_SNR_0db_mod_16QAM_TDL-A.mat...predictions.npy
                # Extract everything between train_data_ and .mat (or end if no .mat)
                if 'train_data_' in basename:
                    start_idx = basename.find('train_data_') + len('train_data_')
                    # Find .mat or end of channel model part
                    if '.mat' in basename[start_idx:]:
                        end_idx = basename.find('.mat', start_idx) + len('.mat')
                        channel_model = basename[start_idx:end_idx]
                        channel_models.add(channel_model)
            else:  # UMi
                # UMi format: ...data_snr5.hdf5...predictions.npy
                if 'data_snr' in basename and '.hdf5' in basename:
                    start_idx = basename.find('data_snr')
                    end_idx = basename.find('.hdf5', start_idx) + len('.hdf5')
                    channel_model = basename[start_idx:end_idx]
                    channel_models.add(channel_model)
    
    return sorted(list(channel_models))


def process_experiment(experiment_type):
    """Process all channel models for an experiment type."""
    print(f"\n{'='*60}")
    print(f"Processing {experiment_type} experiment")
    print(f"{'='*60}\n")
    
    # Load ground truth and input dictionaries
    gt_dict = np.load(GT_PATHS[experiment_type], allow_pickle=True).item()
    input_dict = np.load(INPUT_PATHS[experiment_type], allow_pickle=True).item()
    
    print(f"Loaded GT dict with {len(gt_dict)} keys")
    print(f"Loaded input dict with {len(input_dict)} keys")
    
    # Get all channel models from predictions
    channel_models = get_channel_models_from_predictions(experiment_type)
    print(f"Found {len(channel_models)} channel models: {channel_models}\n")
    
    # Process each channel model
    for channel_model in channel_models:
        print(f"Processing {channel_model}...")
        
        # Find GT key
        gt_key = None
        if experiment_type == 'TDL':
            # Key format: train_data_CHANNEL_MODEL
            gt_key = f"train_data_{channel_model}"
            if gt_key not in gt_dict:
                # Try without .mat extension
                gt_key = f"train_data_{channel_model.replace('.mat', '')}"
        else:  # UMi
            # UMi typically has one key, find the one matching channel_model
            for key in gt_dict.keys():
                if channel_model in key:
                    gt_key = key
                    break
            if gt_key is None and gt_dict:
                gt_key = list(gt_dict.keys())[0]  # Fallback to first key
        
        if gt_key is None or gt_key not in gt_dict:
            print(f"  Warning: Could not find GT key for {channel_model}")
            continue
        
        # Load and slice GT (from index 30)
        gt_data = gt_dict[gt_key][30:]
        
        # Find input key (same logic as GT)
        input_key = None
        if experiment_type == 'TDL':
            input_key = f"train_data_{channel_model}"
            if input_key not in input_dict:
                input_key = f"train_data_{channel_model.replace('.mat', '')}"
        else:  # UMi
            for key in input_dict.keys():
                if channel_model in key:
                    input_key = key
                    break
            if input_key is None and input_dict:
                input_key = list(input_dict.keys())[0]
        
        input_data = input_dict[input_key][30:] if input_key and input_key in input_dict else None
        
        # Collect MSE results
        results = []
        
        # Process each model
        for model_name, model_path in MODEL_PATHS[experiment_type].items():
            if not os.path.exists(model_path):
                continue
            
            # Find prediction file
            pred_file = find_prediction_file(model_path, model_name, channel_model)
            if pred_file is None:
                print(f"  {model_name}: No prediction file found")
                continue
            
            # Load predictions
            try:
                pred_data = np.load(pred_file)
                print(f"  {model_name}: Loaded {pred_data.shape}")
                
                # Calculate MSE
                mse_total, mse_real, mse_imag = calculate_mse(pred_data, gt_data)
                
                if mse_total is not None:
                    results.append({
                        'Model': model_name,
                        'Channel_Model': channel_model,
                        'Total_MSE': mse_total,
                        'Real_MSE': mse_real,
                        'Imag_MSE': mse_imag
                    })
                    print(f"    MSE Total: {mse_total:.6e}, Real: {mse_real:.6e}, Imag: {mse_imag:.6e}")
            except Exception as e:
                print(f"  {model_name}: Error loading - {e}")
        
        # Calculate MSE for input
        if input_data is not None and gt_data is not None:
            mse_total, mse_real, mse_imag = calculate_mse(input_data, gt_data)
            if mse_total is not None:
                results.append({
                    'Model': 'Input',
                    'Channel_Model': channel_model,
                    'Total_MSE': mse_total,
                    'Real_MSE': mse_real,
                    'Imag_MSE': mse_imag
                })
                print(f"  Input: MSE Total: {mse_total:.6e}, Real: {mse_real:.6e}, Imag: {mse_imag:.6e}")
        
        # Save CSV for this channel model
        if results:
            safe_name = channel_model.replace('/', '_').replace('\\', '_').replace('.mat', '').replace('.hdf5', '')
            csv_file = f'mse_results_{experiment_type}_{safe_name}.csv'
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Model', 'Channel_Model', 'Total_MSE', 'Real_MSE', 'Imag_MSE'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"  Saved results to {csv_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Calculate MSE for model predictions')
    parser.add_argument('--experiment', type=str, choices=['TDL', 'UMi'], required=True,
                       help='Experiment type: TDL or UMi')
    args = parser.parse_args()
    
    process_experiment(args.experiment)


if __name__ == '__main__':
    main()

