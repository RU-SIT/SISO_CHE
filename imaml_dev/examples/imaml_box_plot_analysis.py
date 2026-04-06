#!/usr/bin/env python3
"""
Box Plot Analysis for iMAML Inner Loop Losses

This script analyzes the tracked inner loop losses and creates box plots
for each channel across all training epochs for iMAML.
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

class iMAMLBoxPlotAnalyzer:
    """Analyzes tracked iMAML losses and creates box plots."""
    
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
        
        for file in tracking_files:
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
                
            # Extract losses for each step (first and last only for iMAML)
            first_step_losses = channel_data['first_step_loss'].values
            last_step_losses = channel_data['last_step_loss'].values
            
            # Create box plots for each step
            steps = ['First Step (Initial)', 'Last Step (Final)']
            step_data = [first_step_losses, last_step_losses]
            
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
        output_file = self.output_dir / 'imaml_individual_channel_box_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"iMAML individual channel box plots saved to: {output_file}")
    
    def create_combined_box_plots(self):
        """Create combined box plots for all channels."""
        print("\nCreating combined box plots...")
        
        # Prepare data for combined plots (first and last steps only for iMAML)
        first_step_data = []
        last_step_data = []
        channel_labels = []
        
        unique_channels = self.df['channel_name'].unique()
        
        for channel in unique_channels:
            channel_data = self.df[self.df['channel_name'] == channel]
            if len(channel_data) > 0:
                first_step_data.extend(channel_data['first_step_loss'].values)
                last_step_data.extend(channel_data['last_step_loss'].values)
                channel_labels.extend([channel] * len(channel_data))
        
        # Create combined box plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # First Step (Initial)
        axes[0].boxplot(first_step_data, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2))
        axes[0].set_title('iMAML First Step (Initial Loss)\nAll Channels Combined', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_first = np.mean(first_step_data)
        std_first = np.std(first_step_data)
        axes[0].text(0.02, 0.98, f'Mean={mean_first:.4f}\nStd={std_first:.4f}\nN={len(first_step_data)}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Last Step (Final)
        axes[1].boxplot(last_step_data, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='darkgreen', linewidth=2))
        axes[1].set_title('iMAML Last Step (Final Loss)\nAll Channels Combined', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_last = np.mean(last_step_data)
        std_last = np.std(last_step_data)
        axes[1].text(0.02, 0.98, f'Mean={mean_last:.4f}\nStd={std_last:.4f}\nN={len(last_step_data)}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'imaml_combined_box_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"iMAML combined box plots saved to: {output_file}")
    
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
            
            # Plot learning progression over epochs (first and last steps only for iMAML)
            epochs = channel_data['epoch'].values
            first_step_losses = channel_data['first_step_loss'].values
            last_step_losses = channel_data['last_step_loss'].values
            
            ax.plot(epochs, first_step_losses, 'o-', label='First Step (Initial)', alpha=0.7, markersize=3)
            ax.plot(epochs, last_step_losses, 's-', label='Last Step (Final)', alpha=0.7, markersize=3)
            
            ax.set_title(f'iMAML {channel}\nLearning Progression Over Training', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add trend lines (first and last steps only for iMAML)
            if len(epochs) > 1 and len(set(epochs)) > 1:  # Check for sufficient data and variation
                try:
                    z_first = np.polyfit(epochs, first_step_losses, 1)
                    z_last = np.polyfit(epochs, last_step_losses, 1)
                    
                    p_first = np.poly1d(z_first)
                    p_last = np.poly1d(z_last)
                    
                    ax.plot(epochs, p_first(epochs), "--", alpha=0.5, color='blue')
                    ax.plot(epochs, p_last(epochs), "--", alpha=0.5, color='green')
                except np.linalg.LinAlgError:
                    # Skip trend lines if linear regression fails
                    pass
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'imaml_learning_progression_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"iMAML learning progression plots saved to: {output_file}")
    
    def create_statistical_summary(self):
        """Create statistical summary and save to file."""
        print("\nCreating statistical summary...")
        
        summary_data = []
        
        for channel in self.df['channel_name'].unique():
            channel_data = self.df[self.df['channel_name'] == channel]
            
            if len(channel_data) == 0:
                continue
            
            # Calculate statistics (first and last steps only for iMAML)
            first_step_stats = {
                'mean': np.mean(channel_data['first_step_loss']),
                'std': np.std(channel_data['first_step_loss']),
                'min': np.min(channel_data['first_step_loss']),
                'max': np.max(channel_data['first_step_loss']),
                'q25': np.percentile(channel_data['first_step_loss'], 25),
                'q50': np.percentile(channel_data['first_step_loss'], 50),
                'q75': np.percentile(channel_data['first_step_loss'], 75)
            }
            
            last_step_stats = {
                'mean': np.mean(channel_data['last_step_loss']),
                'std': np.std(channel_data['last_step_loss']),
                'min': np.min(channel_data['last_step_loss']),
                'max': np.max(channel_data['last_step_loss']),
                'q25': np.percentile(channel_data['last_step_loss'], 25),
                'q50': np.percentile(channel_data['last_step_loss'], 50),
                'q75': np.percentile(channel_data['last_step_loss'], 75)
            }
            
            # Calculate improvements
            improvement_first_to_last = first_step_stats['mean'] - last_step_stats['mean']
            
            # Improvement percentage
            improvement_pct_first_to_last = (improvement_first_to_last / first_step_stats['mean']) * 100
            
            summary_data.append({
                'channel_name': channel,
                'total_appearances': len(channel_data),
                'epoch_range': f"{channel_data['epoch'].min()}-{channel_data['epoch'].max()}",
                'first_step_mean': first_step_stats['mean'],
                'first_step_std': first_step_stats['std'],
                'last_step_mean': last_step_stats['mean'],
                'last_step_std': last_step_stats['std'],
                'improvement_first_to_last': improvement_first_to_last,
                'improvement_pct_first_to_last': improvement_pct_first_to_last
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_file = self.output_dir / 'imaml_statistical_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        
        # Save detailed statistics to JSON
        json_file = self.output_dir / 'imaml_detailed_statistics.json'
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"iMAML statistical summary saved to: {csv_file}")
        print(f"iMAML detailed statistics saved to: {json_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("iMAML STATISTICAL SUMMARY")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['channel_name']}:")
            print(f"  Total appearances: {row['total_appearances']}")
            print(f"  Epoch range: {row['epoch_range']}")
            print(f"  First Step - Mean: {row['first_step_mean']:.6f}, Std: {row['first_step_std']:.6f}")
            print(f"  Last Step - Mean: {row['last_step_mean']:.6f}, Std: {row['last_step_std']:.6f}")
            print(f"  Improvement First→Last: {row['improvement_first_to_last']:.6f} ({row['improvement_pct_first_to_last']:.2f}%)")
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete box plot analysis."""
        print("Starting complete iMAML box plot analysis...")
        print(f"Tracking data directory: {self.tracking_dir}")
        print(f"Output directory: {self.output_dir}")
        
        if len(self.tracking_data) == 0:
            print("No tracking data found. Please run iMAML training with tracking first.")
            return
        
        # Create all visualizations
        self.create_individual_channel_box_plots()
        self.create_combined_box_plots()
        self.create_learning_progression_plots()
        summary_df = self.create_statistical_summary()
        
        print("\n" + "="*60)
        print("iMAML ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All visualizations and statistics saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze iMAML inner loop losses and create box plots')
    parser.add_argument('--tracking_dir', type=str, default='inner_loop_tracking_data_imaml',
                       help='Directory containing tracking data')
    parser.add_argument('--output_dir', type=str, default='imaml_box_plot_analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = iMAMLBoxPlotAnalyzer(args.tracking_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
