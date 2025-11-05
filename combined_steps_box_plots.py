#!/usr/bin/env python3
"""
Create box plots combining all steps (0, 1, 2) for each epoch.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_combined_steps_box_plots():
    """Create box plots combining all steps for each epoch."""
    
    tracking_dir = "inner_loop_tracking_data_tdl"
    output_dir = "combined_steps_box_plots_tdl"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CREATING COMBINED STEPS BOX PLOTS")
    print("="*60)
    
    # Load all tracking data
    all_data = []
    tracking_files = list(Path(tracking_dir).glob("tracking_data_epoch_*.json"))
    
    print(f"Found {len(tracking_files)} tracking files")
    
    for file in tracking_files:
        print(f"Loading {file}")
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    if len(all_data) == 0:
        print("No tracking data found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(all_data)} tracking entries")
    print(f"Unique channels: {df['channel_name'].unique()}")
    print(f"Epoch range: {df['epoch'].min()} - {df['epoch'].max()}")
    
    # Create combined steps box plots for each channel
    unique_channels = df['channel_name'].unique()
    
    for channel in unique_channels:
        print(f"\nCreating combined steps box plots for: {channel}")
        create_channel_combined_steps_box_plots(df, channel, output_dir)
    
    # Create combined plot for all channels
    create_all_channels_combined_steps_box_plots(df, output_dir)
    
    print(f"\n✓ Combined steps box plots saved to: {output_dir}")

def create_channel_combined_steps_box_plots(df, channel, output_dir):
    """Create combined steps box plots for a specific channel."""
    
    # Filter data for this channel
    channel_data = df[df['channel_name'] == channel].sort_values('epoch')
    
    if len(channel_data) == 0:
        print(f"No data found for channel: {channel}")
        return
    
    print(f"Channel {channel}: {len(channel_data)} samples across {channel_data['epoch'].nunique()} epochs")
    
    # Get unique epochs
    unique_epochs = sorted(channel_data['epoch'].unique())
    
    # Prepare data for each epoch (combining all steps)
    epoch_data = []
    epoch_labels = []
    
    for epoch in unique_epochs:
        epoch_samples = channel_data[channel_data['epoch'] == epoch]
        
        if len(epoch_samples) > 0:
            # Combine all steps for this epoch
            combined_losses = []
            combined_losses.extend(epoch_samples['step_0_loss'].values)
            combined_losses.extend(epoch_samples['step_1_loss'].values)
            combined_losses.extend(epoch_samples['step_2_loss'].values)
            
            epoch_data.append(combined_losses)
            epoch_labels.append(f'Epoch {epoch}')
    
    if len(epoch_data) == 0:
        print(f"No epoch data found for channel: {channel}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(epoch_labels) * 0.8), 8))
    
    # Create box plot
    box_plot = ax.boxplot(epoch_data, patch_artist=True, labels=epoch_labels)
    
    # Color the boxes
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Color other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black')
    
    ax.set_title(f'{channel}\nCombined Steps (0, 1, 2) Learning Progression', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Epoch', fontsize=14)
    ax.set_ylabel('Loss (All Steps Combined)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many epochs
    if len(epoch_labels) > 10:
        ax.tick_params(axis='x', rotation=45)
    
    # Add statistics text
    all_values = np.concatenate(epoch_data)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    n_samples = len(all_values)
    
    stats_text = f'Total N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    safe_channel_name = channel.replace('/', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f'combined_steps_{safe_channel_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined steps box plots saved to: {output_file}")
    
    # Print statistics for this channel
    print_channel_combined_statistics(channel_data, channel)

def create_all_channels_combined_steps_box_plots(df, output_dir):
    """Create combined steps box plots for all channels together."""
    
    print("\nCreating combined steps box plots for all channels...")
    
    # Get all unique epochs
    all_epochs = sorted(df['epoch'].unique())
    
    # Prepare data for each epoch (combining all steps and all channels)
    epoch_data = []
    epoch_labels = []
    
    for epoch in all_epochs:
        epoch_samples = df[df['epoch'] == epoch]
        
        if len(epoch_samples) > 0:
            # Combine all steps for this epoch across all channels
            combined_losses = []
            combined_losses.extend(epoch_samples['step_0_loss'].values)
            combined_losses.extend(epoch_samples['step_1_loss'].values)
            combined_losses.extend(epoch_samples['step_2_loss'].values)
            
            epoch_data.append(combined_losses)
            epoch_labels.append(f'Epoch {epoch}')
    
    if len(epoch_data) == 0:
        print("No epoch data found for all channels")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(max(15, len(epoch_labels) * 0.8), 8))
    
    # Create box plot
    box_plot = ax.boxplot(epoch_data, patch_artist=True, labels=epoch_labels)
    
    # Color the boxes
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    # Color other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black')
    
    ax.set_title('All Channels Combined\nCombined Steps (0, 1, 2) Learning Progression', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Epoch', fontsize=14)
    ax.set_ylabel('Loss (All Steps Combined)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many epochs
    if len(epoch_labels) > 10:
        ax.tick_params(axis='x', rotation=45)
    
    # Add statistics text
    all_values = np.concatenate(epoch_data)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    n_samples = len(all_values)
    
    stats_text = f'Total N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'all_channels_combined_steps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All channels combined steps box plots saved to: {output_file}")

def print_channel_combined_statistics(channel_data, channel):
    """Print detailed statistics for a channel with combined steps."""
    
    print(f"\n--- Combined Steps Statistics for {channel} ---")
    
    # Get all losses combined
    all_step_0 = channel_data['step_0_loss'].values
    all_step_1 = channel_data['step_1_loss'].values
    all_step_2 = channel_data['step_2_loss'].values
    
    # Combine all steps
    all_combined = np.concatenate([all_step_0, all_step_1, all_step_2])
    
    print(f"Total samples: {len(channel_data)}")
    print(f"Total combined steps: {len(all_combined)}")
    print(f"Epochs: {channel_data['epoch'].min()} - {channel_data['epoch'].max()}")
    
    print(f"\nCombined Steps Statistics:")
    print(f"Mean: {np.mean(all_combined):.6f}")
    print(f"Std: {np.std(all_combined):.6f}")
    print(f"Min: {np.min(all_combined):.6f}")
    print(f"Max: {np.max(all_combined):.6f}")
    print(f"Q25: {np.percentile(all_combined, 25):.6f}")
    print(f"Q50: {np.percentile(all_combined, 50):.6f}")
    print(f"Q75: {np.percentile(all_combined, 75):.6f}")
    
    # Per-epoch statistics
    print(f"\nPer-Epoch Combined Statistics:")
    for epoch in sorted(channel_data['epoch'].unique()):
        epoch_data = channel_data[channel_data['epoch'] == epoch]
        if len(epoch_data) > 0:
            # Combine all steps for this epoch
            epoch_combined = np.concatenate([
                epoch_data['step_0_loss'].values,
                epoch_data['step_1_loss'].values,
                epoch_data['step_2_loss'].values
            ])
            
            print(f"  Epoch {epoch}: {len(epoch_combined)} combined samples")
            print(f"    Mean: {np.mean(epoch_combined):.4f} ± {np.std(epoch_combined):.4f}")
            print(f"    Range: {np.min(epoch_combined):.4f} - {np.max(epoch_combined):.4f}")

if __name__ == '__main__':
    create_combined_steps_box_plots()

