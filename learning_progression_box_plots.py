#!/usr/bin/env python3
"""
Create learning progression box plots showing loss distributions over training epochs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_learning_progression_box_plots():
    """Create box plots showing learning progression over training epochs."""
    
    tracking_dir = "inner_loop_tracking_data_umi"
    output_dir = "learning_progression_box_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CREATING LEARNING PROGRESSION BOX PLOTS")
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
    
    # Create learning progression box plots for each channel
    unique_channels = df['channel_name'].unique()
    
    for channel in unique_channels:
        print(f"\nCreating learning progression box plots for: {channel}")
        create_channel_learning_progression(df, channel, output_dir)
    
    print(f"\n✓ Learning progression box plots saved to: {output_dir}")

def create_channel_learning_progression(df, channel, output_dir):
    """Create learning progression box plots for a specific channel."""
    
    # Filter data for this channel
    channel_data = df[df['channel_name'] == channel].sort_values('epoch')
    
    if len(channel_data) == 0:
        print(f"No data found for channel: {channel}")
        return
    
    print(f"Channel {channel}: {len(channel_data)} samples across {channel_data['epoch'].nunique()} epochs")
    
    # Get unique epochs
    unique_epochs = sorted(channel_data['epoch'].unique())
    n_epochs = len(unique_epochs)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare data for each step
    step_0_data = []
    step_1_data = []
    step_2_data = []
    epoch_labels = []
    
    for epoch in unique_epochs:
        epoch_data = channel_data[channel_data['epoch'] == epoch]
        
        if len(epoch_data) > 0:
            step_0_data.append(epoch_data['step_0_loss'].values)
            step_1_data.append(epoch_data['step_1_loss'].values)
            step_2_data.append(epoch_data['step_2_loss'].values)
            epoch_labels.append(f'Epoch {epoch}')
    
    # Create box plots for each step
    steps = ['Step 0 (Initial)', 'Step 1 (After 1 update)', 'Step 2 (After 2 updates)']
    step_data = [step_0_data, step_1_data, step_2_data]
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    
    for i, (step_name, data, color) in enumerate(zip(steps, step_data, colors)):
        ax = axes[i]
        
        if len(data) > 0:
            # Create box plot
            box_plot = ax.boxplot(data, patch_artist=True, labels=epoch_labels)
            
            # Color the boxes
            for patch in box_plot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Color other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_plot[element], color='black')
            
            ax.set_title(f'{channel}\n{step_name}\nLearning Progression', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if there are many epochs
            if len(epoch_labels) > 5:
                ax.tick_params(axis='x', rotation=45)
            
            # Add statistics text
            if len(data) > 0:
                all_values = np.concatenate(data)
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                n_samples = len(all_values)
                
                stats_text = f'Total N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.set_title(f'{channel}\n{step_name}\nNo Data Available', 
                        fontsize=14, fontweight='bold')
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    safe_channel_name = channel.replace('/', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f'learning_progression_{safe_channel_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning progression box plots saved to: {output_file}")
    
    # Print statistics for this channel
    print_channel_statistics(channel_data, channel)

def print_channel_statistics(channel_data, channel):
    """Print detailed statistics for a channel."""
    
    print(f"\n--- Statistics for {channel} ---")
    
    # Overall statistics
    step_0_all = channel_data['step_0_loss'].values
    step_1_all = channel_data['step_1_loss'].values
    step_2_all = channel_data['step_2_loss'].values
    
    print(f"Total samples: {len(channel_data)}")
    print(f"Epochs: {channel_data['epoch'].min()} - {channel_data['epoch'].max()}")
    
    print(f"\nOverall Statistics:")
    print(f"Step 0 - Mean: {np.mean(step_0_all):.6f}, Std: {np.std(step_0_all):.6f}")
    print(f"Step 1 - Mean: {np.mean(step_1_all):.6f}, Std: {np.std(step_1_all):.6f}")
    print(f"Step 2 - Mean: {np.mean(step_2_all):.6f}, Std: {np.std(step_2_all):.6f}")
    
    # Improvements
    improvement_0_to_1 = np.mean(step_0_all) - np.mean(step_1_all)
    improvement_1_to_2 = np.mean(step_1_all) - np.mean(step_2_all)
    improvement_0_to_2 = np.mean(step_0_all) - np.mean(step_2_all)
    
    print(f"\nImprovements:")
    print(f"Step 0→1: {improvement_0_to_1:.6f} ({(improvement_0_to_1/np.mean(step_0_all)*100):.2f}%)")
    print(f"Step 1→2: {improvement_1_to_2:.6f} ({(improvement_1_to_2/np.mean(step_1_all)*100):.2f}%)")
    print(f"Step 0→2: {improvement_0_to_2:.6f} ({(improvement_0_to_2/np.mean(step_0_all)*100):.2f}%)")
    
    # Per-epoch statistics
    print(f"\nPer-Epoch Statistics:")
    for epoch in sorted(channel_data['epoch'].unique()):
        epoch_data = channel_data[channel_data['epoch'] == epoch]
        if len(epoch_data) > 0:
            step_0 = epoch_data['step_0_loss'].values
            step_1 = epoch_data['step_1_loss'].values
            step_2 = epoch_data['step_2_loss'].values
            
            print(f"  Epoch {epoch}: {len(epoch_data)} samples")
            print(f"    Step 0: {np.mean(step_0):.4f} ± {np.std(step_0):.4f}")
            print(f"    Step 1: {np.mean(step_1):.4f} ± {np.std(step_1):.4f}")
            print(f"    Step 2: {np.mean(step_2):.4f} ± {np.std(step_2):.4f}")

def create_combined_learning_progression(df, output_dir):
    """Create combined learning progression box plots for all channels."""
    
    print("\nCreating combined learning progression box plots...")
    
    # Get all unique epochs
    all_epochs = sorted(df['epoch'].unique())
    n_epochs = len(all_epochs)
    
    if n_epochs == 0:
        print("No epochs found!")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Prepare data for each step across all channels
    step_0_data = []
    step_1_data = []
    step_2_data = []
    epoch_labels = []
    
    for epoch in all_epochs:
        epoch_data = df[df['epoch'] == epoch]
        
        if len(epoch_data) > 0:
            step_0_data.append(epoch_data['step_0_loss'].values)
            step_1_data.append(epoch_data['step_1_loss'].values)
            step_2_data.append(epoch_data['step_2_loss'].values)
            epoch_labels.append(f'Epoch {epoch}')
    
    # Create box plots for each step
    steps = ['Step 0 (Initial)', 'Step 1 (After 1 update)', 'Step 2 (After 2 updates)']
    step_data = [step_0_data, step_1_data, step_2_data]
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    
    for i, (step_name, data, color) in enumerate(zip(steps, step_data, colors)):
        ax = axes[i]
        
        if len(data) > 0:
            # Create box plot
            box_plot = ax.boxplot(data, patch_artist=True, labels=epoch_labels)
            
            # Color the boxes
            for patch in box_plot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'All Channels Combined\n{step_name}\nLearning Progression', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if there are many epochs
            if len(epoch_labels) > 5:
                ax.tick_params(axis='x', rotation=45)
            
            # Add statistics text
            all_values = np.concatenate(data)
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            n_samples = len(all_values)
            
            stats_text = f'Total N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'combined_learning_progression_box_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined learning progression box plots saved to: {output_file}")

if __name__ == '__main__':
    create_learning_progression_box_plots()

