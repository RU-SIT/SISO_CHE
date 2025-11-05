#!/usr/bin/env python3
"""
Box Plot Analysis for MAML Inner Loop Losses

This script analyzes the tracked inner loop losses and creates box plots
for each channel across all training epochs, as requested by the advisor.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BoxPlotAnalyzer:
    """Analyzes tracked MAML losses and creates box plots."""
    
    def __init__(self, tracking_dir, output_dir):
        self.tracking_dir = Path(tracking_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all tracking data
        self.tracking_data = self.load_all_tracking_data()
        self.df = pd.DataFrame(self.tracking_data)
        
        print(f"Loaded {len(self.tracking_data)} tracking entries")
        print(f"Unique channels: {self.df['channel_name'].nunique()}")
        print(f"Epoch range: {self.df['epoch'].min()} - {self.df['epoch'].max()}")
        
    def load_all_tracking_data(self):
        """Load all tracking data from JSON files."""
        all_data = []
        
        # Find all tracking data files
        tracking_files = list(self.tracking_dir.glob("tracking_data_epoch_*.json"))
        
        if not tracking_files:
            print(f"No tracking data files found in {self.tracking_dir}")
            return []
        
        print(f"Found {len(tracking_files)} tracking data files")
        
        for file in sorted(tracking_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return all_data
    
    def create_individual_channel_box_plots(self):
        """Create box plots for each individual channel."""
        print("\nCreating individual channel box plots...")
        
        unique_channels = self.df['channel_name'].unique()
        n_channels = len(unique_channels)
        
        # Create subplots
        fig, axes = plt.subplots(n_channels, 3, figsize=(15, 5 * n_channels))
        if n_channels == 1:
            axes = axes.reshape(1, -1)
        
        for i, channel in enumerate(unique_channels):
            channel_data = self.df[self.df['channel_name'] == channel]
            
            if len(channel_data) == 0:
                continue
                
            # Extract losses for each step
            step_0_losses = channel_data['step_0_loss'].values
            step_1_losses = channel_data['step_1_loss'].values
            step_2_losses = channel_data['step_2_loss'].values
            
            # Create box plots for each step
            steps = ['Step 0 (Initial)', 'Step 1 (After 1 update)', 'Step 2 (After 2 updates)']
            step_data = [step_0_losses, step_1_losses, step_2_losses]
            
            for j, (step_name, data) in enumerate(zip(steps, step_data)):
                ax = axes[i, j]
                
                # Create box plot
                box_plot = ax.boxplot(data, patch_artist=True, 
                                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                                     medianprops=dict(color='red', linewidth=2),
                                     whiskerprops=dict(color='black', linewidth=1.5),
                                     capprops=dict(color='black', linewidth=1.5),
                                     flierprops=dict(marker='o', markerfacecolor='red', 
                                                   markeredgecolor='red', markersize=4))
                
                ax.set_title(f'{channel}\n{step_name}', fontsize=10, fontweight='bold')
                ax.set_ylabel('Loss', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = np.mean(data)
                std_val = np.std(data)
                n_samples = len(data)
                
                stats_text = f'N={n_samples}\nMean={mean_val:.4f}\nStd={std_val:.4f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'individual_channel_box_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual channel box plots saved to: {output_file}")
    
    def create_combined_box_plots(self):
        """Create combined box plots for all channels."""
        print("\nCreating combined box plots...")
        
        # Prepare data for combined plots
        step_0_data = []
        step_1_data = []
        step_2_data = []
        channel_labels = []
        
        unique_channels = self.df['channel_name'].unique()
        
        for channel in unique_channels:
            channel_data = self.df[self.df['channel_name'] == channel]
            if len(channel_data) > 0:
                step_0_data.extend(channel_data['step_0_loss'].values)
                step_1_data.extend(channel_data['step_1_loss'].values)
                step_2_data.extend(channel_data['step_2_loss'].values)
                channel_labels.extend([channel] * len(channel_data))
        
        # Create combined box plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Step 0 (Initial)
        axes[0].boxplot(step_0_data, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2))
        axes[0].set_title('Step 0 (Initial Loss)\nAll Channels Combined', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_0 = np.mean(step_0_data)
        std_0 = np.std(step_0_data)
        axes[0].text(0.02, 0.98, f'Mean={mean_0:.4f}\nStd={std_0:.4f}\nN={len(step_0_data)}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Step 1 (After 1 update)
        axes[1].boxplot(step_1_data, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='darkgreen', linewidth=2))
        axes[1].set_title('Step 1 (After 1 Update)\nAll Channels Combined', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_1 = np.mean(step_1_data)
        std_1 = np.std(step_1_data)
        axes[1].text(0.02, 0.98, f'Mean={mean_1:.4f}\nStd={std_1:.4f}\nN={len(step_1_data)}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Step 2 (After 2 updates)
        axes[2].boxplot(step_2_data, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='darkblue', linewidth=2))
        axes[2].set_title('Step 2 (After 2 Updates)\nAll Channels Combined', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Add statistics
        mean_2 = np.mean(step_2_data)
        std_2 = np.std(step_2_data)
        axes[2].text(0.02, 0.98, f'Mean={mean_2:.4f}\nStd={std_2:.4f}\nN={len(step_2_data)}',
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'combined_box_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined box plots saved to: {output_file}")
    
    def create_learning_progression_plots(self):
        """Create plots showing learning progression."""
        print("\nCreating learning progression plots...")
        
        unique_channels = self.df['channel_name'].unique()
        n_channels = len(unique_channels)
        
        # Create subplots for each channel
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 4 * n_channels))
        if n_channels == 1:
            axes = [axes]
        
        for i, channel in enumerate(unique_channels):
            channel_data = self.df[self.df['channel_name'] == channel].sort_values('epoch')
            
            if len(channel_data) == 0:
                continue
            
            ax = axes[i]
            
            # Plot learning progression over epochs
            epochs = channel_data['epoch'].values
            step_0_losses = channel_data['step_0_loss'].values
            step_1_losses = channel_data['step_1_loss'].values
            step_2_losses = channel_data['step_2_loss'].values
            
            ax.plot(epochs, step_0_losses, 'o-', label='Step 0 (Initial)', alpha=0.7, markersize=3)
            ax.plot(epochs, step_1_losses, 's-', label='Step 1 (After 1 update)', alpha=0.7, markersize=3)
            ax.plot(epochs, step_2_losses, '^-', label='Step 2 (After 2 updates)', alpha=0.7, markersize=3)
            
            ax.set_title(f'{channel}\nLearning Progression Over Training', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add trend lines
            if len(epochs) > 1:
                z0 = np.polyfit(epochs, step_0_losses, 1)
                z1 = np.polyfit(epochs, step_1_losses, 1)
                z2 = np.polyfit(epochs, step_2_losses, 1)
                
                p0 = np.poly1d(z0)
                p1 = np.poly1d(z1)
                p2 = np.poly1d(z2)
                
                ax.plot(epochs, p0(epochs), "--", alpha=0.5, color='blue')
                ax.plot(epochs, p1(epochs), "--", alpha=0.5, color='green')
                ax.plot(epochs, p2(epochs), "--", alpha=0.5, color='red')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'learning_progression_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Learning progression plots saved to: {output_file}")
    
    def create_statistical_summary(self):
        """Create statistical summary and save to file."""
        print("\nCreating statistical summary...")
        
        summary_data = []
        
        for channel in self.df['channel_name'].unique():
            channel_data = self.df[self.df['channel_name'] == channel]
            
            if len(channel_data) == 0:
                continue
            
            # Calculate statistics
            step_0_stats = {
                'mean': np.mean(channel_data['step_0_loss']),
                'std': np.std(channel_data['step_0_loss']),
                'min': np.min(channel_data['step_0_loss']),
                'max': np.max(channel_data['step_0_loss']),
                'q25': np.percentile(channel_data['step_0_loss'], 25),
                'q50': np.percentile(channel_data['step_0_loss'], 50),
                'q75': np.percentile(channel_data['step_0_loss'], 75)
            }
            
            step_1_stats = {
                'mean': np.mean(channel_data['step_1_loss']),
                'std': np.std(channel_data['step_1_loss']),
                'min': np.min(channel_data['step_1_loss']),
                'max': np.max(channel_data['step_1_loss']),
                'q25': np.percentile(channel_data['step_1_loss'], 25),
                'q50': np.percentile(channel_data['step_1_loss'], 50),
                'q75': np.percentile(channel_data['step_1_loss'], 75)
            }
            
            step_2_stats = {
                'mean': np.mean(channel_data['step_2_loss']),
                'std': np.std(channel_data['step_2_loss']),
                'min': np.min(channel_data['step_2_loss']),
                'max': np.max(channel_data['step_2_loss']),
                'q25': np.percentile(channel_data['step_2_loss'], 25),
                'q50': np.percentile(channel_data['step_2_loss'], 50),
                'q75': np.percentile(channel_data['step_2_loss'], 75)
            }
            
            # Calculate improvements
            improvement_0_to_1 = step_0_stats['mean'] - step_1_stats['mean']
            improvement_1_to_2 = step_1_stats['mean'] - step_2_stats['mean']
            improvement_0_to_2 = step_0_stats['mean'] - step_2_stats['mean']
            
            # Improvement percentages
            improvement_pct_0_to_1 = (improvement_0_to_1 / step_0_stats['mean']) * 100
            improvement_pct_1_to_2 = (improvement_1_to_2 / step_1_stats['mean']) * 100
            improvement_pct_0_to_2 = (improvement_0_to_2 / step_0_stats['mean']) * 100
            
            summary_data.append({
                'channel_name': channel,
                'total_appearances': len(channel_data),
                'epoch_range': f"{channel_data['epoch'].min()}-{channel_data['epoch'].max()}",
                'step_0_mean': step_0_stats['mean'],
                'step_0_std': step_0_stats['std'],
                'step_1_mean': step_1_stats['mean'],
                'step_1_std': step_1_stats['std'],
                'step_2_mean': step_2_stats['mean'],
                'step_2_std': step_2_stats['std'],
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
        csv_file = self.output_dir / 'statistical_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        
        # Save detailed statistics to JSON
        json_file = self.output_dir / 'detailed_statistics.json'
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Statistical summary saved to: {csv_file}")
        print(f"Detailed statistics saved to: {json_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['channel_name']}:")
            print(f"  Total appearances: {row['total_appearances']}")
            print(f"  Epoch range: {row['epoch_range']}")
            print(f"  Step 0 - Mean: {row['step_0_mean']:.6f}, Std: {row['step_0_std']:.6f}")
            print(f"  Step 1 - Mean: {row['step_1_mean']:.6f}, Std: {row['step_1_std']:.6f}")
            print(f"  Step 2 - Mean: {row['step_2_mean']:.6f}, Std: {row['step_2_std']:.6f}")
            print(f"  Improvement 0→1: {row['improvement_0_to_1']:.6f} ({row['improvement_pct_0_to_1']:.2f}%)")
            print(f"  Improvement 1→2: {row['improvement_1_to_2']:.6f} ({row['improvement_pct_1_to_2']:.2f}%)")
            print(f"  Improvement 0→2: {row['improvement_0_to_2']:.6f} ({row['improvement_pct_0_to_2']:.2f}%)")
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete box plot analysis."""
        print("Starting complete box plot analysis...")
        print(f"Tracking data directory: {self.tracking_dir}")
        print(f"Output directory: {self.output_dir}")
        
        if len(self.tracking_data) == 0:
            print("No tracking data found. Please run MAML training with tracking first.")
            return
        
        # Create all visualizations
        self.create_individual_channel_box_plots()
        self.create_combined_box_plots()
        self.create_learning_progression_plots()
        summary_df = self.create_statistical_summary()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All visualizations and statistics saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MAML inner loop losses and create box plots')
    parser.add_argument('--tracking_dir', type=str, default='inner_loop_tracking_data',
                       help='Directory containing tracking data')
    parser.add_argument('--output_dir', type=str, default='box_plot_analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = BoxPlotAnalyzer(args.tracking_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
