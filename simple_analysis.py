#!/usr/bin/env python3
"""
Simple analysis script that works with limited data.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_tracking_data():
    """Analyze the tracking data and create visualizations."""
    
    tracking_dir = "inner_loop_tracking_data"
    output_dir = "simple_analysis_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("SIMPLE ANALYSIS OF TRACKING DATA")
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
    
    # Create box plots
    create_box_plots(df, output_dir)
    
    # Create summary statistics
    create_summary_statistics(df, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")

def create_box_plots(df, output_dir):
    """Create box plots for the data."""
    
    print("\nCreating box plots...")
    
    unique_channels = df['channel_name'].unique()
    n_channels = len(unique_channels)
    
    # Create figure
    fig, axes = plt.subplots(n_channels, 3, figsize=(15, 5 * n_channels))
    if n_channels == 1:
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
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        
        for j, (step_name, data, color) in enumerate(zip(steps, step_data, colors)):
            ax = axes[i, j]
            
            # Create box plot
            box_plot = ax.boxplot(data, patch_artist=True)
            
            # Color the boxes
            for patch in box_plot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{channel}\n{step_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            n_samples = len(data)
            
            stats_text = f'N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'channel_box_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to: {output_file}")

def create_summary_statistics(df, output_dir):
    """Create summary statistics."""
    
    print("\nCreating summary statistics...")
    
    # Create summary data
    summary_data = []
    
    for channel in df['channel_name'].unique():
        channel_data = df[df['channel_name'] == channel]
        
        # Calculate statistics
        step_0 = channel_data['step_0_loss'].values
        step_1 = channel_data['step_1_loss'].values
        step_2 = channel_data['step_2_loss'].values
        
        # Improvements
        improvement_0_to_1 = np.mean(step_0) - np.mean(step_1)
        improvement_1_to_2 = np.mean(step_1) - np.mean(step_2)
        improvement_0_to_2 = np.mean(step_0) - np.mean(step_2)
        
        # Improvement percentages
        improvement_pct_0_to_1 = (improvement_0_to_1 / np.mean(step_0)) * 100
        improvement_pct_1_to_2 = (improvement_1_to_2 / np.mean(step_1)) * 100
        improvement_pct_0_to_2 = (improvement_0_to_2 / np.mean(step_0)) * 100
        
        summary_data.append({
            'channel_name': channel,
            'total_appearances': len(channel_data),
            'step_0_mean': np.mean(step_0),
            'step_0_std': np.std(step_0),
            'step_1_mean': np.mean(step_1),
            'step_1_std': np.std(step_1),
            'step_2_mean': np.mean(step_2),
            'step_2_std': np.std(step_2),
            'improvement_0_to_1': improvement_0_to_1,
            'improvement_1_to_2': improvement_1_to_2,
            'improvement_0_to_2': improvement_0_to_2,
            'improvement_pct_0_to_1': improvement_pct_0_to_1,
            'improvement_pct_1_to_2': improvement_pct_1_to_2,
            'improvement_pct_0_to_2': improvement_pct_0_to_2
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['channel_name']}:")
        print(f"  Total appearances: {row['total_appearances']}")
        print(f"  Step 0 - Mean: {row['step_0_mean']:.6f}, Std: {row['step_0_std']:.6f}")
        print(f"  Step 1 - Mean: {row['step_1_mean']:.6f}, Std: {row['step_1_std']:.6f}")
        print(f"  Step 2 - Mean: {row['step_2_mean']:.6f}, Std: {row['step_2_std']:.6f}")
        print(f"  Improvement 0→1: {row['improvement_0_to_1']:.6f} ({row['improvement_pct_0_to_1']:.2f}%)")
        print(f"  Improvement 1→2: {row['improvement_1_to_2']:.6f} ({row['improvement_pct_1_to_2']:.2f}%)")
        print(f"  Improvement 0→2: {row['improvement_0_to_2']:.6f} ({row['improvement_pct_0_to_2']:.2f}%)")
        
        # Check if learning is happening
        if row['improvement_0_to_2'] > 0:
            print(f"  ✓ LEARNING IS HAPPENING: {row['improvement_0_to_2']:.6f} improvement")
        else:
            print(f"  ❌ NO LEARNING: {row['improvement_0_to_2']:.6f} change")
    
    print(f"\nSummary statistics saved to: {csv_file}")

if __name__ == '__main__':
    analyze_tracking_data()

