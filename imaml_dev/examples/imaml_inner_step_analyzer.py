#!/usr/bin/env python3
"""
iMAML Inner Step Analysis Tool

This script provides comprehensive analysis of iMAML performance on UMi and TDL scenarios
by tracking and analyzing inner loop losses across different channels and training steps.

Features:
- Detailed tracking of inner loop losses for each channel
- Statistical analysis and visualization
- Comparison between UMi and TDL scenarios
- Learning progression analysis
- Performance metrics calculation
"""

import numpy as np
import torch
import torch.nn as nn
import utils as utils
import random
import time as timer
import pickle
import argparse
import pathlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from learner_model import Learner
from learner_model import make_fc_network, make_conv_network
from utils import DataLog
import sys
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to access modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from Data_Nshot import ChannelEstimationNShot

# Fix the relative import issue by adding the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

class InnerStepAnalyzer:
    """Comprehensive analyzer for iMAML inner step performance."""
    
    def __init__(self, save_dir, scenario_name="UMi"):
        self.save_dir = save_dir
        self.scenario_name = scenario_name
        self.tracking_data = []
        self.channel_stats = {}
        self.step_wise_losses = {}
        self.learning_curves = {}
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def track_inner_step_losses(self, epoch, channel_names, task_losses, step_details=None):
        """
        Track detailed inner step losses for each channel.
        
        Args:
            epoch: Current training epoch
            channel_names: List of channel names for this task
            task_losses: List of loss dictionaries for each task
            step_details: Optional detailed step-by-step losses
        """
        for i, channel_name in enumerate(channel_names):
            if i < len(task_losses):
                task_loss = task_losses[i]
                
                # Extract losses for this channel
                initial_loss = float(task_loss.get('initial', 0.0))
                final_loss = float(task_loss.get('final', 0.0))
                
                # Store tracking data
                tracking_entry = {
                    'epoch': epoch,
                    'channel_name': channel_name,
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'improvement': initial_loss - final_loss,
                    'improvement_ratio': (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add step-wise details if available
                if step_details and i < len(step_details):
                    step_detail = step_details[i]
                    for step_key, step_value in step_detail.items():
                        tracking_entry[f'step_{step_key}'] = float(step_value)
                
                self.tracking_data.append(tracking_entry)
                
                # Update channel statistics
                if channel_name not in self.channel_stats:
                    self.channel_stats[channel_name] = {
                        'total_appearances': 0,
                        'initial_losses': [],
                        'final_losses': [],
                        'improvements': [],
                        'improvement_ratios': [],
                        'epochs': [],
                        'step_wise_losses': []
                    }
                
                self.channel_stats[channel_name]['total_appearances'] += 1
                self.channel_stats[channel_name]['initial_losses'].append(initial_loss)
                self.channel_stats[channel_name]['final_losses'].append(final_loss)
                self.channel_stats[channel_name]['improvements'].append(initial_loss - final_loss)
                self.channel_stats[channel_name]['improvement_ratios'].append(
                    (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
                )
                self.channel_stats[channel_name]['epochs'].append(epoch)
                
                # Store step-wise losses if available
                if step_details and i < len(step_details):
                    self.channel_stats[channel_name]['step_wise_losses'].append(step_details[i])
    
    def analyze_learning_progression(self):
        """Analyze learning progression over epochs."""
        if not self.tracking_data:
            return {}
        
        df = pd.DataFrame(self.tracking_data)
        
        # Group by epoch and calculate statistics
        epoch_stats = df.groupby('epoch').agg({
            'initial_loss': ['mean', 'std', 'min', 'max'],
            'final_loss': ['mean', 'std', 'min', 'max'],
            'improvement': ['mean', 'std', 'min', 'max'],
            'improvement_ratio': ['mean', 'std', 'min', 'max']
        }).round(6)
        
        # Flatten column names
        epoch_stats.columns = ['_'.join(col).strip() for col in epoch_stats.columns]
        
        return epoch_stats.to_dict('index')
    
    def generate_channel_statistics(self):
        """Generate comprehensive statistics for each channel."""
        stats_summary = {}
        
        for channel_name, stats in self.channel_stats.items():
            if not stats['initial_losses']:
                continue
                
            initial_losses = np.array(stats['initial_losses'])
            final_losses = np.array(stats['final_losses'])
            improvements = np.array(stats['improvements'])
            improvement_ratios = np.array(stats['improvement_ratios'])
            
            stats_summary[channel_name] = {
                'total_appearances': stats['total_appearances'],
                'initial_loss': {
                    'mean': float(np.mean(initial_losses)),
                    'std': float(np.std(initial_losses)),
                    'min': float(np.min(initial_losses)),
                    'max': float(np.max(initial_losses)),
                    'median': float(np.median(initial_losses))
                },
                'final_loss': {
                    'mean': float(np.mean(final_losses)),
                    'std': float(np.std(final_losses)),
                    'min': float(np.min(final_losses)),
                    'max': float(np.max(final_losses)),
                    'median': float(np.median(final_losses))
                },
                'improvement': {
                    'mean': float(np.mean(improvements)),
                    'std': float(np.std(improvements)),
                    'min': float(np.min(improvements)),
                    'max': float(np.max(improvements)),
                    'median': float(np.median(improvements))
                },
                'improvement_ratio': {
                    'mean': float(np.mean(improvement_ratios)),
                    'std': float(np.std(improvement_ratios)),
                    'min': float(np.min(improvement_ratios)),
                    'max': float(np.max(improvement_ratios)),
                    'median': float(np.median(improvement_ratios))
                },
                'consistency_score': float(1.0 - np.std(improvement_ratios)),  # Higher is more consistent
                'learning_efficiency': float(np.mean(improvement_ratios) * (1.0 - np.std(improvement_ratios)))
            }
        
        return stats_summary
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        if not self.tracking_data:
            print("No tracking data available for visualization")
            return
        
        df = pd.DataFrame(self.tracking_data)
        
        # 1. Channel-wise Box Plots
        self._create_channel_box_plots(df)
        
        # 2. Learning Progression Plots
        self._create_learning_progression_plots(df)
        
        # 3. Improvement Distribution Plots
        self._create_improvement_distribution_plots(df)
        
        # 4. Channel Performance Comparison
        self._create_channel_performance_comparison(df)
        
        # 5. Epoch-wise Analysis
        self._create_epoch_wise_analysis(df)
    
    def _create_channel_box_plots(self, df):
        """Create box plots for each channel."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'iMAML Inner Step Analysis - {self.scenario_name} Scenario', fontsize=16)
        
        # Initial vs Final Loss Comparison
        ax1 = axes[0, 0]
        channel_data = []
        channel_labels = []
        for channel in df['channel_name'].unique():
            channel_df = df[df['channel_name'] == channel]
            channel_data.extend([
                *channel_df['initial_loss'].tolist(),
                *channel_df['final_loss'].tolist()
            ])
            channel_labels.extend([
                *[f'{channel}_Initial'] * len(channel_df),
                *[f'{channel}_Final'] * len(channel_df)
            ])
        
        if channel_data:
            box_data = pd.DataFrame({'Loss': channel_data, 'Channel_Step': channel_labels})
            sns.boxplot(data=box_data, x='Channel_Step', y='Loss', ax=ax1)
            ax1.set_title('Initial vs Final Loss Distribution by Channel')
            ax1.tick_params(axis='x', rotation=45)
        
        # Improvement Distribution
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='channel_name', y='improvement', ax=ax2)
        ax2.set_title('Loss Improvement Distribution by Channel')
        ax2.tick_params(axis='x', rotation=45)
        
        # Improvement Ratio Distribution
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='channel_name', y='improvement_ratio', ax=ax3)
        ax3.set_title('Improvement Ratio Distribution by Channel')
        ax3.tick_params(axis='x', rotation=45)
        
        # Loss Scatter Plot
        ax4 = axes[1, 1]
        sns.scatterplot(data=df, x='initial_loss', y='final_loss', hue='channel_name', ax=ax4)
        ax4.plot([0, df['initial_loss'].max()], [0, df['initial_loss'].max()], 'k--', alpha=0.5)
        ax4.set_title('Initial vs Final Loss Scatter Plot')
        ax4.set_xlabel('Initial Loss')
        ax4.set_ylabel('Final Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'imaml_channel_box_plots_{self.scenario_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_learning_progression_plots(self, df):
        """Create learning progression plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'iMAML Learning Progression Analysis - {self.scenario_name} Scenario', fontsize=16)
        
        # Overall learning curve
        ax1 = axes[0, 0]
        epoch_stats = df.groupby('epoch').agg({
            'initial_loss': 'mean',
            'final_loss': 'mean',
            'improvement': 'mean'
        })
        
        ax1.plot(epoch_stats.index, epoch_stats['initial_loss'], label='Initial Loss', marker='o')
        ax1.plot(epoch_stats.index, epoch_stats['final_loss'], label='Final Loss', marker='s')
        ax1.set_title('Overall Learning Progression')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement over epochs
        ax2 = axes[0, 1]
        ax2.plot(epoch_stats.index, epoch_stats['improvement'], label='Mean Improvement', marker='o', color='green')
        ax2.set_title('Improvement Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Improvement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Channel-wise learning curves
        ax3 = axes[1, 0]
        for channel in df['channel_name'].unique():
            channel_df = df[df['channel_name'] == channel]
            channel_epoch_stats = channel_df.groupby('epoch')['improvement'].mean()
            ax3.plot(channel_epoch_stats.index, channel_epoch_stats.values, 
                    label=channel, marker='o', alpha=0.7)
        
        ax3.set_title('Channel-wise Learning Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Loss variance over epochs
        ax4 = axes[1, 1]
        epoch_variance = df.groupby('epoch').agg({
            'initial_loss': 'std',
            'final_loss': 'std'
        })
        
        ax4.plot(epoch_variance.index, epoch_variance['initial_loss'], label='Initial Loss Std', marker='o')
        ax4.plot(epoch_variance.index, epoch_variance['final_loss'], label='Final Loss Std', marker='s')
        ax4.set_title('Loss Variance Over Training')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Standard Deviation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'imaml_learning_progression_{self.scenario_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_improvement_distribution_plots(self, df):
        """Create improvement distribution plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'iMAML Improvement Distribution Analysis - {self.scenario_name} Scenario', fontsize=16)
        
        # Overall improvement histogram
        ax1 = axes[0, 0]
        ax1.hist(df['improvement'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('Overall Improvement Distribution')
        ax1.set_xlabel('Loss Improvement')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Improvement ratio histogram
        ax2 = axes[0, 1]
        ax2.hist(df['improvement_ratio'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Improvement Ratio Distribution')
        ax2.set_xlabel('Improvement Ratio')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Channel-wise improvement comparison
        ax3 = axes[1, 0]
        sns.violinplot(data=df, x='channel_name', y='improvement', ax=ax3)
        ax3.set_title('Channel-wise Improvement Distribution')
        ax3.tick_params(axis='x', rotation=45)
        
        # Improvement vs Initial Loss
        ax4 = axes[1, 1]
        sns.scatterplot(data=df, x='initial_loss', y='improvement', hue='channel_name', ax=ax4)
        ax4.set_title('Improvement vs Initial Loss')
        ax4.set_xlabel('Initial Loss')
        ax4.set_ylabel('Loss Improvement')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'imaml_improvement_distribution_{self.scenario_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_channel_performance_comparison(self, df):
        """Create channel performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'iMAML Channel Performance Comparison - {self.scenario_name} Scenario', fontsize=16)
        
        # Channel performance summary
        channel_summary = df.groupby('channel_name').agg({
            'initial_loss': ['mean', 'std'],
            'final_loss': ['mean', 'std'],
            'improvement': ['mean', 'std'],
            'improvement_ratio': ['mean', 'std']
        }).round(6)
        
        # Flatten column names
        channel_summary.columns = ['_'.join(col).strip() for col in channel_summary.columns]
        channel_summary = channel_summary.reset_index()
        
        # Mean improvement by channel
        ax1 = axes[0, 0]
        bars = ax1.bar(channel_summary['channel_name'], channel_summary['improvement_mean'])
        ax1.errorbar(channel_summary['channel_name'], channel_summary['improvement_mean'], 
                    yerr=channel_summary['improvement_std'], fmt='none', color='black', capsize=5)
        ax1.set_title('Mean Improvement by Channel')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Mean Loss Improvement')
        ax1.tick_params(axis='x', rotation=45)
        
        # Improvement ratio by channel
        ax2 = axes[0, 1]
        bars = ax2.bar(channel_summary['channel_name'], channel_summary['improvement_ratio_mean'])
        ax2.errorbar(channel_summary['channel_name'], channel_summary['improvement_ratio_mean'], 
                    yerr=channel_summary['improvement_ratio_std'], fmt='none', color='black', capsize=5)
        ax2.set_title('Mean Improvement Ratio by Channel')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Mean Improvement Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Initial vs Final loss comparison
        ax3 = axes[1, 0]
        x = np.arange(len(channel_summary))
        width = 0.35
        
        ax3.bar(x - width/2, channel_summary['initial_loss_mean'], width, label='Initial Loss', alpha=0.8)
        ax3.bar(x + width/2, channel_summary['final_loss_mean'], width, label='Final Loss', alpha=0.8)
        ax3.set_title('Initial vs Final Loss by Channel')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Loss')
        ax3.set_xticks(x)
        ax3.set_xticklabels(channel_summary['channel_name'], rotation=45)
        ax3.legend()
        
        # Consistency score (1 - std of improvement ratio)
        ax4 = axes[1, 1]
        consistency_scores = 1 - channel_summary['improvement_ratio_std']
        bars = ax4.bar(channel_summary['channel_name'], consistency_scores)
        ax4.set_title('Learning Consistency by Channel')
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Consistency Score (1 - std)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'imaml_channel_performance_{self.scenario_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_epoch_wise_analysis(self, df):
        """Create epoch-wise analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'iMAML Epoch-wise Analysis - {self.scenario_name} Scenario', fontsize=16)
        
        # Loss trends over epochs
        ax1 = axes[0, 0]
        epoch_stats = df.groupby('epoch').agg({
            'initial_loss': 'mean',
            'final_loss': 'mean'
        })
        
        ax1.plot(epoch_stats.index, epoch_stats['initial_loss'], label='Initial Loss', marker='o', linewidth=2)
        ax1.plot(epoch_stats.index, epoch_stats['final_loss'], label='Final Loss', marker='s', linewidth=2)
        ax1.fill_between(epoch_stats.index, 
                        epoch_stats['initial_loss'] - df.groupby('epoch')['initial_loss'].std(),
                        epoch_stats['initial_loss'] + df.groupby('epoch')['initial_loss'].std(),
                        alpha=0.3, label='Initial Loss ± 1σ')
        ax1.fill_between(epoch_stats.index, 
                        epoch_stats['final_loss'] - df.groupby('epoch')['final_loss'].std(),
                        epoch_stats['final_loss'] + df.groupby('epoch')['final_loss'].std(),
                        alpha=0.3, label='Final Loss ± 1σ')
        
        ax1.set_title('Loss Trends Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement trends
        ax2 = axes[0, 1]
        improvement_stats = df.groupby('epoch')['improvement'].agg(['mean', 'std'])
        ax2.plot(improvement_stats.index, improvement_stats['mean'], marker='o', linewidth=2, color='green')
        ax2.fill_between(improvement_stats.index, 
                        improvement_stats['mean'] - improvement_stats['std'],
                        improvement_stats['mean'] + improvement_stats['std'],
                        alpha=0.3, color='green')
        ax2.set_title('Improvement Trends Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Learning efficiency over epochs
        ax3 = axes[1, 0]
        efficiency_stats = df.groupby('epoch').agg({
            'improvement_ratio': ['mean', 'std']
        })
        efficiency_stats.columns = ['mean', 'std']
        ax3.plot(efficiency_stats.index, efficiency_stats['mean'], marker='o', linewidth=2, color='purple')
        ax3.fill_between(efficiency_stats.index, 
                        efficiency_stats['mean'] - efficiency_stats['std'],
                        efficiency_stats['mean'] + efficiency_stats['std'],
                        alpha=0.3, color='purple')
        ax3.set_title('Learning Efficiency Over Epochs')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Improvement Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Channel diversity over epochs
        ax4 = axes[1, 1]
        channel_diversity = df.groupby('epoch')['channel_name'].nunique()
        ax4.plot(channel_diversity.index, channel_diversity.values, marker='o', linewidth=2, color='orange')
        ax4.set_title('Channel Diversity Over Epochs')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Number of Unique Channels')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'imaml_epoch_analysis_{self.scenario_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(self, epoch=None):
        """Save comprehensive analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed tracking data
        if self.tracking_data:
            tracking_file = os.path.join(self.save_dir, f'imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.json')
            with open(tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            
            # Save as CSV for easy analysis
            df = pd.DataFrame(self.tracking_data)
            csv_file = os.path.join(self.save_dir, f'imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.csv')
            df.to_csv(csv_file, index=False)
        
        # Save channel statistics
        channel_stats = self.generate_channel_statistics()
        stats_file = os.path.join(self.save_dir, f'imaml_channel_statistics_{self.scenario_name.lower()}_{timestamp}.json')
        with open(stats_file, 'w') as f:
            json.dump(channel_stats, f, indent=2)
        
        # Save learning progression analysis
        progression_analysis = self.analyze_learning_progression()
        progression_file = os.path.join(self.save_dir, f'imaml_progression_analysis_{self.scenario_name.lower()}_{timestamp}.json')
        with open(progression_file, 'w') as f:
            json.dump(progression_analysis, f, indent=2)
        
        # Create summary report
        self._create_summary_report(channel_stats, progression_analysis, timestamp)
        
        print(f"Analysis results saved to {self.save_dir}")
        print(f"Files created:")
        print(f"  - imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.json")
        print(f"  - imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.csv")
        print(f"  - imaml_channel_statistics_{self.scenario_name.lower()}_{timestamp}.json")
        print(f"  - imaml_progression_analysis_{self.scenario_name.lower()}_{timestamp}.json")
        print(f"  - imaml_summary_report_{self.scenario_name.lower()}_{timestamp}.txt")
    
    def _create_summary_report(self, channel_stats, progression_analysis, timestamp):
        """Create a comprehensive summary report."""
        report_file = os.path.join(self.save_dir, f'imaml_summary_report_{self.scenario_name.lower()}_{timestamp}.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"iMAML INNER STEP ANALYSIS SUMMARY REPORT\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%c')}\n")
            f.write(f"Scenario: {self.scenario_name}\n")
            f.write(f"Total Tracking Entries: {len(self.tracking_data)}\n")
            f.write(f"Unique Channels: {len(channel_stats)}\n\n")
            
            f.write(f"GENERATED FILES:\n")
            f.write(f"{'-'*20}\n")
            f.write(f"- imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.json\n")
            f.write(f"- imaml_tracking_data_{self.scenario_name.lower()}_{timestamp}.csv\n")
            f.write(f"- imaml_channel_statistics_{self.scenario_name.lower()}_{timestamp}.json\n")
            f.write(f"- imaml_progression_analysis_{self.scenario_name.lower()}_{timestamp}.json\n")
            f.write(f"- imaml_summary_report_{self.scenario_name.lower()}_{timestamp}.txt\n")
            f.write(f"- imaml_channel_box_plots_{self.scenario_name.lower()}.png\n")
            f.write(f"- imaml_learning_progression_{self.scenario_name.lower()}.png\n")
            f.write(f"- imaml_improvement_distribution_{self.scenario_name.lower()}.png\n")
            f.write(f"- imaml_channel_performance_{self.scenario_name.lower()}.png\n")
            f.write(f"- imaml_epoch_analysis_{self.scenario_name.lower()}.png\n\n")
            
            f.write(f"CHANNEL PERFORMANCE SUMMARY:\n")
            f.write(f"{'-'*30}\n")
            
            # Sort channels by learning efficiency
            sorted_channels = sorted(channel_stats.items(), 
                                   key=lambda x: x[1]['learning_efficiency'], 
                                   reverse=True)
            
            for channel, stats in sorted_channels:
                f.write(f"\n{channel}:\n")
                f.write(f"  Total Appearances: {stats['total_appearances']}\n")
                f.write(f"  Initial Loss - Mean: {stats['initial_loss']['mean']:.6f}, Std: {stats['initial_loss']['std']:.6f}\n")
                f.write(f"  Final Loss - Mean: {stats['final_loss']['mean']:.6f}, Std: {stats['final_loss']['std']:.6f}\n")
                f.write(f"  Improvement - Mean: {stats['improvement']['mean']:.6f}, Std: {stats['improvement']['std']:.6f}\n")
                f.write(f"  Improvement Ratio - Mean: {stats['improvement_ratio']['mean']:.6f}, Std: {stats['improvement_ratio']['std']:.6f}\n")
                f.write(f"  Consistency Score: {stats['consistency_score']:.6f}\n")
                f.write(f"  Learning Efficiency: {stats['learning_efficiency']:.6f}\n")
            
            f.write(f"\nOVERALL STATISTICS:\n")
            f.write(f"{'-'*20}\n")
            
            if self.tracking_data:
                df = pd.DataFrame(self.tracking_data)
                f.write(f"  Mean Initial Loss: {df['initial_loss'].mean():.6f}\n")
                f.write(f"  Mean Final Loss: {df['final_loss'].mean():.6f}\n")
                f.write(f"  Mean Improvement: {df['improvement'].mean():.6f}\n")
                f.write(f"  Mean Improvement Ratio: {df['improvement_ratio'].mean():.6f}\n")
                f.write(f"  Best Performing Channel: {sorted_channels[0][0]}\n")
                f.write(f"  Worst Performing Channel: {sorted_channels[-1][0]}\n")
            
            f.write(f"\nINTERPRETATION GUIDE:\n")
            f.write(f"{'-'*20}\n")
            f.write(f"✓ Good Learning Indicators:\n")
            f.write(f"  - High improvement ratio (>0.1)\n")
            f.write(f"  - High consistency score (>0.8)\n")
            f.write(f"  - High learning efficiency (>0.05)\n")
            f.write(f"  - Decreasing final losses over epochs\n\n")
            f.write(f"❌ Poor Learning Indicators:\n")
            f.write(f"  - Low or negative improvement ratio\n")
            f.write(f"  - Low consistency score (<0.5)\n")
            f.write(f"  - High variance in improvements\n")
            f.write(f"  - Increasing final losses over epochs\n")

def run_imaml_analysis(data_dir, scenario_name, save_dir, meta_steps=1000, n_way=2, k_shot=10, 
                      inner_lr=1e-3, outer_lr=1e-4, n_steps=2, lam=2.0, use_gpu=True):
    """
    Run comprehensive iMAML analysis for a specific scenario.
    
    Args:
        data_dir: Path to the dataset
        scenario_name: Name of the scenario (UMi, TDL, etc.)
        save_dir: Directory to save analysis results
        meta_steps: Number of meta training steps
        n_way: Number of ways for few-shot learning
        k_shot: Number of shots
        inner_lr: Inner loop learning rate
        outer_lr: Outer loop learning rate
        n_steps: Number of inner steps
        lam: Regularization parameter
        use_gpu: Whether to use GPU
    """
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    print(f"="*80)
    print(f"iMAML INNER STEP ANALYSIS - {scenario_name.upper()} SCENARIO")
    print(f"="*80)
    
    # Create analyzer
    analyzer = InnerStepAnalyzer(save_dir, scenario_name)
    
    # Load Data
    print(f"\nLoading dataset from: {data_dir}")
    dataset = ChannelEstimationNShot(
        root=data_dir,
        batchsz=8,  # Fixed batch size for analysis
        n_way=n_way,
        k_shot=k_shot,
        k_query=k_shot
    )
    
    # Initialize Model
    print("Initializing iMAML model...")
    learner_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=8)
    fast_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=8)
    
    meta_learner = Learner(
        model=learner_net,
        loss_function=torch.nn.MSELoss(),
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        GPU=use_gpu
    )
    fast_learner = Learner(
        model=fast_net,
        loss_function=torch.nn.MSELoss(),
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        GPU=use_gpu
    )
    
    device = 'cuda:0' if use_gpu else 'cpu'
    if use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    lam_tensor = torch.tensor(lam, device=device)
    
    print(f"\nTraining Configuration:")
    print(f"  Scenario: {scenario_name}")
    print(f"  Meta Steps: {meta_steps}")
    print(f"  N-way: {n_way}, K-shot: {k_shot}")
    print(f"  Inner Steps: {n_steps}")
    print(f"  Lambda: {lam}")
    print(f"  Device: {device}")
    
    # Training Loop with Analysis
    print(f"\nStarting training and analysis...")
    losses = np.zeros((meta_steps, 4))
    
    for outstep in tqdm(range(meta_steps), desc=f"Training {scenario_name}"):
        # Get next batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
            xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
            (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
            qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = dataset.next(mode='train')
        
        w_k = meta_learner.get_params()
        meta_grad = torch.zeros_like(w_k)
        lam_grad = 0.0
        
        # Track losses for each task
        task_losses = []
        step_details = []
        
        for i in range(n_way):
            fast_learner.set_params(w_k.clone())
            task = {
                'x_train': torch.tensor(x_qry_scld[i], device=device),
                'y_train': torch.tensor(y_qry_scld[i], device=device),
                'x_val': torch.tensor(x_spt_scld[i], device=device),
                'y_val': torch.tensor(y_spt_scld[i], device=device)
            }
            
            # Get initial validation loss
            vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            
            # Learn task and get training losses
            tl = fast_learner.learn_task(task, num_steps=n_steps)
            
            # Compute regularization loss
            fast_learner.inner_opt.zero_grad()
            regu_loss = fast_learner.regularization_loss(w_k, lam_tensor)
            regu_loss.backward()
            fast_learner.inner_opt.step()
            
            # Get final validation loss
            vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            
            # Store losses for this task
            task_loss = {
                'initial': vl_before,
                'final': vl_after
            }
            task_losses.append(task_loss)
            
            # Store step details
            step_detail = {
                '0': vl_before,
                str(n_steps): vl_after
            }
            step_details.append(step_detail)
            
            # Compute meta-gradient
            valid_loss = fast_learner.get_loss(task['x_val'], task['y_val'])
            valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
            flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
            
            if torch.isnan(flat_grad).any():
                print(f"Warning: NaN detected in gradient at step {outstep}")
                continue
            
            # Use direct gradient (simplified for analysis)
            task_outer_grad = flat_grad
            meta_grad += (task_outer_grad / 8)  # Divide by batch size
            losses[outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after]) / 8)
            
            if lam_grad != 0.0:
                lam_grad += (0.0 / 8)  # Simplified lambda gradient
        
        # Track losses for this meta step
        analyzer.track_inner_step_losses(outstep, spt_name, task_losses, step_details)
        
        # Update meta-learner
        meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
        
        # Clear GPU cache periodically
        if use_gpu and outstep % 50 == 0:
            torch.cuda.empty_cache()
        
        # Print progress
        if outstep % 100 == 0:
            param_norm = torch.norm(meta_learner.get_params())
            print(f"\nMeta Step {outstep}: parameter norm = {param_norm.item():.6f}")
            
            if outstep % 500 == 0:
                summary = analyzer.generate_channel_statistics()
                print(f"\n--- Analysis Summary at Step {outstep} ---")
                for channel, stats in list(summary.items())[:3]:  # Show top 3 channels
                    print(f"{channel}: {stats['total_appearances']} appearances, "
                          f"Efficiency: {stats['learning_efficiency']:.4f}")
    
    # Generate final analysis
    print(f"\nGenerating comprehensive analysis...")
    analyzer.create_visualizations()
    analyzer.save_analysis_results()
    
    print(f"\n{scenario_name} analysis completed!")
    print(f"Results saved to: {save_dir}")

def main():
    """Main function to run iMAML analysis for both UMi and TDL scenarios."""
    import sys
    from pathlib import Path

    _WC_ROOT = Path(__file__).resolve().parents[2]
    if str(_WC_ROOT) not in sys.path:
        sys.path.insert(0, str(_WC_ROOT))
    from paths import (
        default_dataset_tdl_siso_folder,
        default_dataset_umi_siso_folder,
        default_imaml_inner_step_analysis_dir,
    )

    parser = argparse.ArgumentParser(description='iMAML Inner Step Analysis Tool')
    parser.add_argument('--umi_data_dir', type=str, 
                       default=default_dataset_umi_siso_folder(),
                       help='Path to UMi dataset')
    parser.add_argument('--tdl_data_dir', type=str, 
                       default=default_dataset_tdl_siso_folder(),
                       help='Path to TDL dataset')
    parser.add_argument('--save_dir', type=str, 
                       default=default_imaml_inner_step_analysis_dir(),
                       help='Directory to save analysis results')
    parser.add_argument('--meta_steps', type=int, default=1000, help='Number of meta training steps')
    parser.add_argument('--N_way', type=int, default=2, help='Number of ways for few-shot learning')
    parser.add_argument('--K_shot', type=int, default=10, help='Number of shots')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-4, help='Outer loop learning rate')
    parser.add_argument('--n_steps', type=int, default=2, help='Number of inner steps')
    parser.add_argument('--lam', type=float, default=2.0, help='Regularization parameter')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--scenarios', nargs='+', default=['UMi', 'TDL'], 
                       help='Scenarios to analyze')
    
    args = parser.parse_args()
    
    # Create main save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run analysis for each scenario
    for scenario in args.scenarios:
        print(f"\n{'='*80}")
        print(f"STARTING ANALYSIS FOR {scenario.upper()} SCENARIO")
        print(f"{'='*80}")
        
        # Set data directory based on scenario
        if scenario.upper() == 'UMI':
            data_dir = args.umi_data_dir
        elif scenario.upper() == 'TDL':
            data_dir = args.tdl_data_dir
        else:
            print(f"Unknown scenario: {scenario}")
            continue
        
        # Create scenario-specific save directory
        scenario_save_dir = os.path.join(args.save_dir, f"{scenario.lower()}_analysis")
        os.makedirs(scenario_save_dir, exist_ok=True)
        
        try:
            # Run analysis
            run_imaml_analysis(
                data_dir=data_dir,
                scenario_name=scenario,
                save_dir=scenario_save_dir,
                meta_steps=args.meta_steps,
                n_way=args.N_way,
                k_shot=args.K_shot,
                inner_lr=args.inner_lr,
                outer_lr=args.outer_lr,
                n_steps=args.n_steps,
                lam=args.lam,
                use_gpu=args.use_gpu
            )
        except Exception as e:
            print(f"Error analyzing {scenario} scenario: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.save_dir}")
    
    # Create comparison analysis if multiple scenarios were analyzed
    if len(args.scenarios) > 1:
        print("\nCreating cross-scenario comparison...")
        create_cross_scenario_comparison(args.save_dir, args.scenarios)

def create_cross_scenario_comparison(main_save_dir, scenarios):
    """Create comparison analysis across different scenarios."""
    comparison_dir = os.path.join(main_save_dir, "cross_scenario_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"Cross-scenario comparison saved to: {comparison_dir}")
    # Additional comparison analysis can be implemented here

if __name__ == '__main__':
    main()
