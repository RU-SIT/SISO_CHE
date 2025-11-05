#!/usr/bin/env python3
"""
Corrected MAML Task Performance Analysis
Analyzes inner loop performance per task for both UMI and TDL channel models.
Tracks first/last occurrence and step 2 performance for each task.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from datetime import datetime

class TaskPerformanceAnalyzer:
    def __init__(self, tracking_dir, output_dir):
        self.tracking_dir = tracking_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.task_data = {}  # task_name -> task_info
        
    def load_tracking_data(self):
        """Load all tracking data from JSON files"""
        print("Loading tracking data...")
        
        # Find all tracking files
        tracking_files = []
        for root, dirs, files in os.walk(self.tracking_dir):
            for file in files:
                if file.startswith('channel_stats_epoch_') and file.endswith('.json'):
                    tracking_files.append(os.path.join(root, file))
        
        tracking_files.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
        print(f"Found {len(tracking_files)} tracking files")
        
        # Load data from each file
        for file_path in tracking_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process each task in this file
                for task_name, task_stats in data.items():
                    if isinstance(task_stats, dict) and 'step_2_losses' in task_stats:
                        # This is the correct structure: step_0_losses, step_1_losses, step_2_losses, epochs
                        if task_name not in self.task_data:
                            self.task_data[task_name] = {
                                'step_0_losses': [],
                                'step_1_losses': [],
                                'step_2_losses': [],
                                'epochs': []
                            }
                        
                        # Extract data for this task
                        step_0 = task_stats.get('step_0_losses', [])
                        step_1 = task_stats.get('step_1_losses', [])
                        step_2 = task_stats.get('step_2_losses', [])
                        epochs = task_stats.get('epochs', [])
                        
                        # Add to our data
                        self.task_data[task_name]['step_0_losses'].extend(step_0)
                        self.task_data[task_name]['step_1_losses'].extend(step_1)
                        self.task_data[task_name]['step_2_losses'].extend(step_2)
                        self.task_data[task_name]['epochs'].extend(epochs)
                            
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded data for {len(self.task_data)} unique tasks")
        
    def analyze_task_performance(self):
        """Analyze performance metrics for each task"""
        print("Analyzing task performance...")
        
        results = {}
        
        for task_name, task_info in self.task_data.items():
            step_0_losses = np.array(task_info['step_0_losses'])
            step_1_losses = np.array(task_info['step_1_losses'])
            step_2_losses = np.array(task_info['step_2_losses'])
            epochs = np.array(task_info['epochs'])
            
            if len(step_2_losses) == 0:
                print(f"Warning: No step 2 losses for task {task_name}")
                continue
            
            # Basic statistics
            first_epoch = int(np.min(epochs)) if len(epochs) > 0 else 0
            last_epoch = int(np.max(epochs)) if len(epochs) > 0 else 0
            total_occurrences = len(step_2_losses)
            
            # Step 2 performance statistics
            step2_mean = np.mean(step_2_losses)
            step2_std = np.std(step_2_losses)
            step2_min = np.min(step_2_losses)
            step2_max = np.max(step_2_losses)
            
            # Performance improvement (first vs last occurrence)
            first_step2 = step_2_losses[0]
            last_step2 = step_2_losses[-1]
            improvement = first_step2 - last_step2
            improvement_pct = (improvement / first_step2) * 100 if first_step2 > 0 else 0
            
            # Learning trend (linear regression on step2 losses over epochs)
            if len(epochs) > 1:
                try:
                    trend_slope, trend_intercept = np.polyfit(epochs, step_2_losses, 1)
                    trend_direction = "improving" if trend_slope < 0 else "degrading"
                except:
                    trend_slope = 0
                    trend_direction = "stable"
            else:
                trend_slope = 0
                trend_direction = "stable"
            
            # Inner loop improvement (step 0 to step 2)
            if len(step_0_losses) > 0 and len(step_2_losses) > 0:
                inner_loop_improvement = np.mean(step_0_losses) - np.mean(step_2_losses)
                inner_loop_improvement_pct = (inner_loop_improvement / np.mean(step_0_losses)) * 100 if np.mean(step_0_losses) > 0 else 0
            else:
                inner_loop_improvement = 0
                inner_loop_improvement_pct = 0
            
            results[task_name] = {
                'first_occurrence': first_epoch,
                'last_occurrence': last_epoch,
                'total_occurrences': total_occurrences,
                'step2_mean': step2_mean,
                'step2_std': step2_std,
                'step2_min': step2_min,
                'step2_max': step2_max,
                'first_step2_loss': first_step2,
                'last_step2_loss': last_step2,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'trend_slope': trend_slope,
                'trend_direction': trend_direction,
                'inner_loop_improvement': inner_loop_improvement,
                'inner_loop_improvement_pct': inner_loop_improvement_pct,
                'step0_mean': np.mean(step_0_losses) if len(step_0_losses) > 0 else 0,
                'step1_mean': np.mean(step_1_losses) if len(step_1_losses) > 0 else 0
            }
        
        return results
    
    def create_task_summary_plots(self, results):
        """Create comprehensive summary plots"""
        print("Creating task summary plots...")
        
        # Convert results to DataFrame for easier plotting
        df_data = []
        for task_name, stats in results.items():
            df_data.append({
                'task_name': task_name,
                'first_occurrence': stats['first_occurrence'],
                'last_occurrence': stats['last_occurrence'],
                'total_occurrences': stats['total_occurrences'],
                'step2_mean': stats['step2_mean'],
                'step2_std': stats['step2_std'],
                'improvement_pct': stats['improvement_pct'],
                'trend_slope': stats['trend_slope'],
                'inner_loop_improvement_pct': stats['inner_loop_improvement_pct']
            })
        
        df = pd.DataFrame(df_data)
        
        # 1. Task occurrence timeline
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # First vs Last occurrence
        axes[0,0].scatter(df['first_occurrence'], df['last_occurrence'], 
                         s=df['total_occurrences']*10, alpha=0.6)
        axes[0,0].set_xlabel('First Occurrence (Epoch)')
        axes[0,0].set_ylabel('Last Occurrence (Epoch)')
        axes[0,0].set_title('Task Occurrence Timeline')
        axes[0,0].plot([df['first_occurrence'].min(), df['first_occurrence'].max()],
                      [df['first_occurrence'].min(), df['first_occurrence'].max()], 
                      'r--', alpha=0.5, label='y=x')
        axes[0,0].legend()
        
        # Step 2 performance distribution
        axes[0,1].hist(df['step2_mean'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Mean Step 2 Loss')
        axes[0,1].set_ylabel('Number of Tasks')
        axes[0,1].set_title('Step 2 Performance Distribution')
        
        # Improvement percentage
        axes[1,0].bar(range(len(df)), df['improvement_pct'], alpha=0.7)
        axes[1,0].set_xlabel('Task Index')
        axes[1,0].set_ylabel('Improvement %')
        axes[1,0].set_title('Task Improvement Percentage')
        axes[1,0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Learning trend
        colors = ['green' if x < 0 else 'red' for x in df['trend_slope']]
        axes[1,1].scatter(df['total_occurrences'], df['trend_slope'], 
                         c=colors, alpha=0.6)
        axes[1,1].set_xlabel('Total Occurrences')
        axes[1,1].set_ylabel('Learning Trend Slope')
        axes[1,1].set_title('Learning Trend vs Occurrences')
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'task_performance_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_task_plots(self, results):
        """Create detailed plots for each task"""
        print("Creating individual task plots...")
        
        # Create subdirectory for individual task plots
        task_plots_dir = os.path.join(self.output_dir, 'individual_tasks')
        os.makedirs(task_plots_dir, exist_ok=True)
        
        for task_name, task_info in self.task_data.items():
            if task_name not in results:
                continue
                
            step_0_losses = np.array(task_info['step_0_losses'])
            step_1_losses = np.array(task_info['step_1_losses'])
            step_2_losses = np.array(task_info['step_2_losses'])
            epochs = np.array(task_info['epochs'])
            
            if len(step_2_losses) == 0:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Step 2 loss over time
            axes[0,0].plot(epochs, step_2_losses, 'o-', alpha=0.7)
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Step 2 Loss')
            axes[0,0].set_title(f'{task_name}\nStep 2 Performance Over Time')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(epochs) > 1:
                try:
                    z = np.polyfit(epochs, step_2_losses, 1)
                    p = np.poly1d(z)
                    axes[0,0].plot(epochs, p(epochs), "r--", alpha=0.8, 
                                 label=f'Trend: {z[0]:.2e}x + {z[1]:.2e}')
                    axes[0,0].legend()
                except:
                    pass
            
            # All steps performance (average across all occurrences)
            if len(step_0_losses) > 0 and len(step_1_losses) > 0 and len(step_2_losses) > 0:
                steps = ['Step 0', 'Step 1', 'Step 2']
                means = [np.mean(step_0_losses), np.mean(step_1_losses), np.mean(step_2_losses)]
                stds = [np.std(step_0_losses), np.std(step_1_losses), np.std(step_2_losses)]
                
                axes[0,1].errorbar(steps, means, yerr=stds, fmt='o-', capsize=5, alpha=0.7)
                axes[0,1].set_ylabel('Loss')
                axes[0,1].set_title('Average Inner Loop Performance')
                axes[0,1].grid(True, alpha=0.3)
            
            # Step 2 loss distribution
            axes[1,0].hist(step_2_losses, bins=min(10, len(step_2_losses)), 
                          alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Step 2 Loss')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Step 2 Loss Distribution')
            
            # Performance statistics
            stats = results[task_name]
            stats_text = f"""
            First Occurrence: Epoch {stats['first_occurrence']}
            Last Occurrence: Epoch {stats['last_occurrence']}
            Total Occurrences: {stats['total_occurrences']}
            
            Step 2 Performance:
            Mean: {stats['step2_mean']:.4f}
            Std: {stats['step2_std']:.4f}
            Min: {stats['step2_min']:.4f}
            Max: {stats['step2_max']:.4f}
            
            Improvement: {stats['improvement']:.4f} ({stats['improvement_pct']:.1f}%)
            Trend: {stats['trend_direction']}
            
            Inner Loop Improvement: {stats['inner_loop_improvement']:.4f} ({stats['inner_loop_improvement_pct']:.1f}%)
            """
            
            axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
            axes[1,1].set_title('Task Statistics')
            
            plt.tight_layout()
            
            # Clean task name for filename
            clean_name = task_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '_')
            plt.savefig(os.path.join(task_plots_dir, f'{clean_name}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_comparative_analysis(self, results):
        """Create comparative analysis between different task types"""
        print("Creating comparative analysis...")
        
        # Categorize tasks by type (UMI vs TDL)
        umi_tasks = []
        tdl_tasks = []
        
        for task_name in results:
            if 'UMI' in task_name.upper() or 'UMI' in task_name:
                umi_tasks.append(task_name)
            elif 'TDL' in task_name.upper() or 'TDL' in task_name:
                tdl_tasks.append(task_name)
        
        if not umi_tasks and not tdl_tasks:
            print("No UMI/TDL categorization found, using all tasks")
            umi_tasks = list(results.keys())
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data for comparison
        umi_data = [results[task] for task in umi_tasks if task in results]
        tdl_data = [results[task] for task in tdl_tasks if task in results]
        
        # Step 2 performance comparison
        if umi_data:
            umi_step2 = [d['step2_mean'] for d in umi_data]
            axes[0,0].hist(umi_step2, alpha=0.7, label='UMI', bins=15)
        if tdl_data:
            tdl_step2 = [d['step2_mean'] for d in tdl_data]
            axes[0,0].hist(tdl_step2, alpha=0.7, label='TDL', bins=15)
        
        axes[0,0].set_xlabel('Mean Step 2 Loss')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Step 2 Performance Distribution')
        axes[0,0].legend()
        
        # Improvement comparison
        if umi_data:
            umi_improvement = [d['improvement_pct'] for d in umi_data]
            axes[0,1].hist(umi_improvement, alpha=0.7, label='UMI', bins=15)
        if tdl_data:
            tdl_improvement = [d['improvement_pct'] for d in tdl_data]
            axes[0,1].hist(tdl_improvement, alpha=0.7, label='TDL', bins=15)
        
        axes[0,1].set_xlabel('Improvement %')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Improvement Distribution')
        axes[0,1].legend()
        axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Occurrence frequency
        if umi_data:
            umi_occurrences = [d['total_occurrences'] for d in umi_data]
            axes[0,2].hist(umi_occurrences, alpha=0.7, label='UMI', bins=15)
        if tdl_data:
            tdl_occurrences = [d['total_occurrences'] for d in tdl_data]
            axes[0,2].hist(tdl_occurrences, alpha=0.7, label='TDL', bins=15)
        
        axes[0,2].set_xlabel('Total Occurrences')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Task Occurrence Frequency')
        axes[0,2].legend()
        
        # Box plots for detailed comparison
        comparison_data = []
        comparison_labels = []
        
        if umi_data:
            comparison_data.append([d['step2_mean'] for d in umi_data])
            comparison_labels.append('UMI')
        if tdl_data:
            comparison_data.append([d['step2_mean'] for d in tdl_data])
            comparison_labels.append('TDL')
        
        if comparison_data:
            axes[1,0].boxplot(comparison_data, labels=comparison_labels)
            axes[1,0].set_ylabel('Step 2 Loss')
            axes[1,0].set_title('Step 2 Performance Box Plot')
        
        # Improvement box plot
        improvement_data = []
        if umi_data:
            improvement_data.append([d['improvement_pct'] for d in umi_data])
        if tdl_data:
            improvement_data.append([d['improvement_pct'] for d in tdl_data])
        
        if improvement_data:
            axes[1,1].boxplot(improvement_data, labels=comparison_labels)
            axes[1,1].set_ylabel('Improvement %')
            axes[1,1].set_title('Improvement Box Plot')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Learning trend comparison
        if umi_data:
            umi_trends = [d['trend_slope'] for d in umi_data]
            axes[1,2].hist(umi_trends, alpha=0.7, label='UMI', bins=15)
        if tdl_data:
            tdl_trends = [d['trend_slope'] for d in tdl_data]
            axes[1,2].hist(tdl_trends, alpha=0.7, label='TDL', bins=15)
        
        axes[1,2].set_xlabel('Learning Trend Slope')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Learning Trend Distribution')
        axes[1,2].legend()
        axes[1,2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparative_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, results):
        """Save detailed results to CSV and JSON"""
        print("Saving detailed results...")
        
        # Convert to DataFrame and save CSV
        df_data = []
        for task_name, stats in results.items():
            df_data.append({
                'task_name': task_name,
                'first_occurrence': stats['first_occurrence'],
                'last_occurrence': stats['last_occurrence'],
                'total_occurrences': stats['total_occurrences'],
                'step2_mean': stats['step2_mean'],
                'step2_std': stats['step2_std'],
                'step2_min': stats['step2_min'],
                'step2_max': stats['step2_max'],
                'first_step2_loss': stats['first_step2_loss'],
                'last_step2_loss': stats['last_step2_loss'],
                'improvement': stats['improvement'],
                'improvement_pct': stats['improvement_pct'],
                'trend_slope': stats['trend_slope'],
                'trend_direction': stats['trend_direction'],
                'inner_loop_improvement': stats['inner_loop_improvement'],
                'inner_loop_improvement_pct': stats['inner_loop_improvement_pct'],
                'step0_mean': stats['step0_mean'],
                'step1_mean': stats['step1_mean']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(self.output_dir, 'task_performance_analysis.csv'), index=False)
        
        # Save detailed JSON
        with open(os.path.join(self.output_dir, 'task_performance_detailed.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        with open(os.path.join(self.output_dir, 'analysis_summary_report.txt'), 'w') as f:
            f.write("MAML Task Performance Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tasks Analyzed: {len(results)}\n\n")
            
            # Overall statistics
            all_step2_means = [stats['step2_mean'] for stats in results.values()]
            all_improvements = [stats['improvement_pct'] for stats in results.values()]
            all_inner_improvements = [stats['inner_loop_improvement_pct'] for stats in results.values()]
            
            f.write("Overall Statistics:\n")
            f.write(f"  Average Step 2 Loss: {np.mean(all_step2_means):.4f} ± {np.std(all_step2_means):.4f}\n")
            f.write(f"  Average Improvement: {np.mean(all_improvements):.2f}% ± {np.std(all_improvements):.2f}%\n")
            f.write(f"  Average Inner Loop Improvement: {np.mean(all_inner_improvements):.2f}% ± {np.std(all_inner_improvements):.2f}%\n")
            f.write(f"  Tasks with Positive Improvement: {sum(1 for x in all_improvements if x > 0)}/{len(all_improvements)}\n\n")
            
            # Best and worst performing tasks
            best_improvement = max(results.items(), key=lambda x: x[1]['improvement_pct'])
            worst_improvement = min(results.items(), key=lambda x: x[1]['improvement_pct'])
            
            f.write("Top Performing Tasks:\n")
            f.write(f"  Best Improvement: {best_improvement[0]} ({best_improvement[1]['improvement_pct']:.2f}%)\n")
            f.write(f"  Worst Improvement: {worst_improvement[0]} ({worst_improvement[1]['improvement_pct']:.2f}%)\n\n")
            
            # Task categorization
            umi_tasks = [name for name in results if 'UMI' in name.upper()]
            tdl_tasks = [name for name in results if 'TDL' in name.upper()]
            
            if umi_tasks:
                umi_improvements = [results[name]['improvement_pct'] for name in umi_tasks]
                f.write(f"UMI Tasks ({len(umi_tasks)}):\n")
                f.write(f"  Average Improvement: {np.mean(umi_improvements):.2f}%\n")
            
            if tdl_tasks:
                tdl_improvements = [results[name]['improvement_pct'] for name in tdl_tasks]
                f.write(f"TDL Tasks ({len(tdl_tasks)}):\n")
                f.write(f"  Average Improvement: {np.mean(tdl_improvements):.2f}%\n")
        
        print(f"Results saved to {self.output_dir}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting MAML Task Performance Analysis...")
        
        # Load data
        self.load_tracking_data()
        
        if not self.task_data:
            print("No task data found! Please check the tracking directory.")
            return
        
        # Analyze performance
        results = self.analyze_task_performance()
        
        if not results:
            print("No results generated! Check the data structure.")
            return
        
        # Create visualizations
        self.create_task_summary_plots(results)
        self.create_individual_task_plots(results)
        self.create_comparative_analysis(results)
        
        # Save results
        self.save_detailed_results(results)
        
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MAML Task Performance Analysis')
    parser.add_argument('--tracking_dir', type=str, required=True,
                       help='Directory containing tracking data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = TaskPerformanceAnalyzer(args.tracking_dir, args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()


