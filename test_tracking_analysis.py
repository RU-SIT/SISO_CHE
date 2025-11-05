#!/usr/bin/env python3
"""
Test script to analyze existing tracking data and create box plots.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_existing_data():
    """Analyze the existing tracking data."""
    
    tracking_dir = "inner_loop_tracking_data"
    output_dir = "test_analysis_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("ANALYZING EXISTING TRACKING DATA")
    print("="*60)
    
    # Load tracking data
    tracking_files = list(Path(tracking_dir).glob("tracking_data_epoch_*.json"))
    print(f"Found {len(tracking_files)} tracking files")
    
    all_data = []
    for file in tracking_files:
        print(f"Loading {file}")
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    print(f"Total tracking entries: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No tracking data found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique channels: {df['channel_name'].unique()}")
    print(f"Epoch range: {df['epoch'].min()} - {df['epoch'].max()}")
    
    # Print sample data
    print("\nSample data:")
    print(df.head())
    
    # Create simple box plots
    create_simple_box_plots(df, output_dir)
    
    # Print statistics
    print_statistics(df)

def create_simple_box_plots(df, output_dir):
    """Create simple box plots."""
    
    print("\nCreating box plots...")
    
    # Get unique channels
    unique_channels = df['channel_name'].unique()
    print(f"Channels: {unique_channels}")
    
    # Create figure
    fig, axes = plt.subplots(len(unique_channels), 3, figsize=(15, 5 * len(unique_channels)))
    if len(unique_channels) == 1:
        axes = axes.reshape(1, -1)
    
    for i, channel in enumerate(unique_channels):
        channel_data = df[df['channel_name'] == channel]
        
        # Extract losses
        step_0_losses = channel_data['step_0_loss'].values
        step_1_losses = channel_data['step_1_loss'].values
        step_2_losses = channel_data['step_2_loss'].values
        
        print(f"\n{channel}:")
        print(f"  Step 0: {len(step_0_losses)} samples, mean={np.mean(step_0_losses):.4f}")
        print(f"  Step 1: {len(step_1_losses)} samples, mean={np.mean(step_1_losses):.4f}")
        print(f"  Step 2: {len(step_2_losses)} samples, mean={np.mean(step_2_losses):.4f}")
        
        # Create box plots
        steps = ['Step 0 (Initial)', 'Step 1 (After 1 update)', 'Step 2 (After 2 updates)']
        step_data = [step_0_losses, step_1_losses, step_2_losses]
        
        for j, (step_name, data) in enumerate(zip(steps, step_data)):
            ax = axes[i, j]
            
            # Create box plot
            box_plot = ax.boxplot(data, patch_artist=True)
            
            # Color the boxes
            for patch in box_plot['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_title(f'{channel}\n{step_name}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            n_samples = len(data)
            
            stats_text = f'N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'simple_box_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to: {output_file}")

def print_statistics(df):
    """Print detailed statistics."""
    
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    
    for channel in df['channel_name'].unique():
        channel_data = df[df['channel_name'] == channel]
        
        print(f"\n{channel}:")
        print(f"  Total appearances: {len(channel_data)}")
        
        # Step 0 statistics
        step_0 = channel_data['step_0_loss'].values
        print(f"  Step 0 - Mean: {np.mean(step_0):.6f}, Std: {np.std(step_0):.6f}")
        print(f"  Step 0 - Min: {np.min(step_0):.6f}, Max: {np.max(step_0):.6f}")
        
        # Step 1 statistics
        step_1 = channel_data['step_1_loss'].values
        print(f"  Step 1 - Mean: {np.mean(step_1):.6f}, Std: {np.std(step_1):.6f}")
        print(f"  Step 1 - Min: {np.min(step_1):.6f}, Max: {np.max(step_1):.6f}")
        
        # Step 2 statistics
        step_2 = channel_data['step_2_loss'].values
        print(f"  Step 2 - Mean: {np.mean(step_2):.6f}, Std: {np.std(step_2):.6f}")
        print(f"  Step 2 - Min: {np.min(step_2):.6f}, Max: {np.max(step_2):.6f}")
        
        # Improvements
        improvement_0_to_1 = np.mean(step_0) - np.mean(step_1)
        improvement_1_to_2 = np.mean(step_1) - np.mean(step_2)
        improvement_0_to_2 = np.mean(step_0) - np.mean(step_2)
        
        print(f"  Improvement 0→1: {improvement_0_to_1:.6f} ({(improvement_0_to_1/np.mean(step_0)*100):.2f}%)")
        print(f"  Improvement 1→2: {improvement_1_to_2:.6f} ({(improvement_1_to_2/np.mean(step_1)*100):.2f}%)")
        print(f"  Improvement 0→2: {improvement_0_to_2:.6f} ({(improvement_0_to_2/np.mean(step_0)*100):.2f}%)")
        
        # Check if learning is happening
        if improvement_0_to_2 > 0:
            print(f"  ✓ LEARNING IS HAPPENING: {improvement_0_to_2:.6f} improvement")
        else:
            print(f"  ❌ NO LEARNING: {improvement_0_to_2:.6f} change")

if __name__ == '__main__':
    analyze_existing_data()
