#!/usr/bin/env python3
"""
Simple MAML Task Occurrence Analysis
Tracks only first and last occurrence values for each task.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class SimpleOccurrenceAnalyzer:
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
        
    def analyze_occurrences(self):
        """Analyze first and last occurrence for each task"""
        print("Analyzing first and last occurrences...")
        
        results = {}
        
        for task_name, task_info in self.task_data.items():
            step_0_losses = np.array(task_info['step_0_losses'])
            step_1_losses = np.array(task_info['step_1_losses'])
            step_2_losses = np.array(task_info['step_2_losses'])
            epochs = np.array(task_info['epochs'])
            
            if len(step_2_losses) == 0:
                print(f"Warning: No data for task {task_name}")
                continue
            
            # First occurrence values
            first_epoch = int(np.min(epochs)) if len(epochs) > 0 else 0
            first_step_0 = step_0_losses[0] if len(step_0_losses) > 0 else 0
            first_step_1 = step_1_losses[0] if len(step_1_losses) > 0 else 0
            first_step_2 = step_2_losses[0] if len(step_2_losses) > 0 else 0
            
            # Last occurrence values
            last_epoch = int(np.max(epochs)) if len(epochs) > 0 else 0
            last_step_0 = step_0_losses[-1] if len(step_0_losses) > 0 else 0
            last_step_1 = step_1_losses[-1] if len(step_1_losses) > 0 else 0
            last_step_2 = step_2_losses[-1] if len(step_2_losses) > 0 else 0
            
            # Total occurrences
            total_occurrences = len(step_2_losses)
            
            results[task_name] = {
                'first_epoch': first_epoch,
                'first_step_0': first_step_0,
                'first_step_1': first_step_1,
                'first_step_2': first_step_2,
                'last_epoch': last_epoch,
                'last_step_0': last_step_0,
                'last_step_1': last_step_1,
                'last_step_2': last_step_2,
                'total_occurrences': total_occurrences
            }
        
        return results
    
    def create_occurrence_plots(self, results):
        """Create plots showing first and last occurrence values"""
        print("Creating occurrence plots...")
        
        # Convert results to DataFrame for easier plotting
        df_data = []
        for task_name, stats in results.items():
            df_data.append({
                'task_name': task_name,
                'first_epoch': stats['first_epoch'],
                'last_epoch': stats['last_epoch'],
                'first_step_0': stats['first_step_0'],
                'first_step_1': stats['first_step_1'],
                'first_step_2': stats['first_step_2'],
                'last_step_0': stats['last_step_0'],
                'last_step_1': stats['last_step_1'],
                'last_step_2': stats['last_step_2'],
                'total_occurrences': stats['total_occurrences']
            })
        
        df = pd.DataFrame(df_data)
        
        # Create comprehensive occurrence plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. First vs Last Epoch
        axes[0,0].scatter(df['first_epoch'], df['last_epoch'], 
                         s=df['total_occurrences']*20, alpha=0.7)
        axes[0,0].set_xlabel('First Occurrence (Epoch)')
        axes[0,0].set_ylabel('Last Occurrence (Epoch)')
        axes[0,0].set_title('First vs Last Occurrence Timeline')
        axes[0,0].plot([df['first_epoch'].min(), df['first_epoch'].max()],
                      [df['first_epoch'].min(), df['first_epoch'].max()], 
                      'r--', alpha=0.5, label='y=x')
        axes[0,0].legend()
        
        # 2. First Step 0 vs Last Step 0
        axes[0,1].scatter(df['first_step_0'], df['last_step_0'], 
                         s=df['total_occurrences']*20, alpha=0.7)
        axes[0,1].set_xlabel('First Step 0 Loss')
        axes[0,1].set_ylabel('Last Step 0 Loss')
        axes[0,1].set_title('Step 0: First vs Last Occurrence')
        axes[0,1].plot([df['first_step_0'].min(), df['first_step_0'].max()],
                      [df['first_step_0'].min(), df['first_step_0'].max()], 
                      'r--', alpha=0.5)
        
        # 3. First Step 2 vs Last Step 2
        axes[0,2].scatter(df['first_step_2'], df['last_step_2'], 
                         s=df['total_occurrences']*20, alpha=0.7)
        axes[0,2].set_xlabel('First Step 2 Loss')
        axes[0,2].set_ylabel('Last Step 2 Loss')
        axes[0,2].set_title('Step 2: First vs Last Occurrence')
        axes[0,2].plot([df['first_step_2'].min(), df['first_step_2'].max()],
                      [df['first_step_2'].min(), df['first_step_2'].max()], 
                      'r--', alpha=0.5)
        
        # 4. Occurrence Timeline
        task_names = df['task_name'].tolist()
        first_epochs = df['first_epoch'].tolist()
        last_epochs = df['last_epoch'].tolist()
        
        y_pos = np.arange(len(task_names))
        axes[1,0].barh(y_pos, first_epochs, alpha=0.7, label='First Occurrence')
        axes[1,0].barh(y_pos, last_epochs, alpha=0.7, label='Last Occurrence')
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in task_names])
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_title('Task Occurrence Timeline')
        axes[1,0].legend()
        
        # 5. Step 0 First vs Last comparison
        x_pos = np.arange(len(task_names))
        width = 0.35
        axes[1,1].bar(x_pos - width/2, df['first_step_0'], width, label='First', alpha=0.7)
        axes[1,1].bar(x_pos + width/2, df['last_step_0'], width, label='Last', alpha=0.7)
        axes[1,1].set_xlabel('Tasks')
        axes[1,1].set_ylabel('Step 0 Loss')
        axes[1,1].set_title('Step 0: First vs Last Occurrence')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in task_names], rotation=45)
        axes[1,1].legend()
        
        # 6. Step 2 First vs Last comparison
        axes[1,2].bar(x_pos - width/2, df['first_step_2'], width, label='First', alpha=0.7)
        axes[1,2].bar(x_pos + width/2, df['last_step_2'], width, label='Last', alpha=0.7)
        axes[1,2].set_xlabel('Tasks')
        axes[1,2].set_ylabel('Step 2 Loss')
        axes[1,2].set_title('Step 2: First vs Last Occurrence')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in task_names], rotation=45)
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'occurrence_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_task_plots(self, results):
        """Create individual plots for each task showing first and last occurrence"""
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
            axes[0,0].plot(epochs, step_2_losses, 'o-', alpha=0.7, label='All occurrences')
            axes[0,0].axvline(x=results[task_name]['first_epoch'], color='green', linestyle='--', 
                             label=f'First: Epoch {results[task_name]["first_epoch"]}')
            axes[0,0].axvline(x=results[task_name]['last_epoch'], color='red', linestyle='--', 
                             label=f'Last: Epoch {results[task_name]["last_epoch"]}')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Step 2 Loss')
            axes[0,0].set_title(f'{task_name}\nStep 2 Performance Over Time')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # All steps at first occurrence
            steps = ['Step 0', 'Step 1', 'Step 2']
            first_values = [results[task_name]['first_step_0'], 
                           results[task_name]['first_step_1'], 
                           results[task_name]['first_step_2']]
            last_values = [results[task_name]['last_step_0'], 
                          results[task_name]['last_step_1'], 
                          results[task_name]['last_step_2']]
            
            x_pos = np.arange(len(steps))
            width = 0.35
            axes[0,1].bar(x_pos - width/2, first_values, width, label='First Occurrence', alpha=0.7)
            axes[0,1].bar(x_pos + width/2, last_values, width, label='Last Occurrence', alpha=0.7)
            axes[0,1].set_xlabel('Inner Steps')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].set_title('First vs Last Occurrence Comparison')
            axes[0,1].set_xticks(x_pos)
            axes[0,1].set_xticklabels(steps)
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Step 0 and Step 2 comparison
            axes[1,0].scatter([results[task_name]['first_step_0']], [results[task_name]['first_step_2']], 
                             s=100, color='green', label='First Occurrence', alpha=0.7)
            axes[1,0].scatter([results[task_name]['last_step_0']], [results[task_name]['last_step_2']], 
                             s=100, color='red', label='Last Occurrence', alpha=0.7)
            axes[1,0].set_xlabel('Step 0 Loss')
            axes[1,0].set_ylabel('Step 2 Loss')
            axes[1,0].set_title('Step 0 vs Step 2: First vs Last')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Task statistics
            stats = results[task_name]
            stats_text = f"""
            Task: {task_name}
            
            First Occurrence:
            Epoch: {stats['first_epoch']}
            Step 0: {stats['first_step_0']:.4f}
            Step 1: {stats['first_step_1']:.4f}
            Step 2: {stats['first_step_2']:.4f}
            
            Last Occurrence:
            Epoch: {stats['last_epoch']}
            Step 0: {stats['last_step_0']:.4f}
            Step 1: {stats['last_step_1']:.4f}
            Step 2: {stats['last_step_2']:.4f}
            
            Total Occurrences: {stats['total_occurrences']}
            """
            
            axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
            axes[1,1].set_title('Task Occurrence Summary')
            
            plt.tight_layout()
            
            # Clean task name for filename
            clean_name = task_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '_')
            plt.savefig(os.path.join(task_plots_dir, f'{clean_name}_occurrence_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_occurrence_results(self, results):
        """Save occurrence results to CSV and JSON"""
        print("Saving occurrence results...")
        
        # Convert to DataFrame and save CSV
        df_data = []
        for task_name, stats in results.items():
            df_data.append({
                'task_name': task_name,
                'first_epoch': stats['first_epoch'],
                'first_step_0': stats['first_step_0'],
                'first_step_1': stats['first_step_1'],
                'first_step_2': stats['first_step_2'],
                'last_epoch': stats['last_epoch'],
                'last_step_0': stats['last_step_0'],
                'last_step_1': stats['last_step_1'],
                'last_step_2': stats['last_step_2'],
                'total_occurrences': stats['total_occurrences']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(self.output_dir, 'occurrence_analysis.csv'), index=False)
        
        # Save detailed JSON
        with open(os.path.join(self.output_dir, 'occurrence_analysis_detailed.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        with open(os.path.join(self.output_dir, 'occurrence_summary_report.txt'), 'w') as f:
            f.write("MAML Task Occurrence Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tasks Analyzed: {len(results)}\n\n")
            
            f.write("Task Occurrence Summary:\n")
            f.write("-" * 30 + "\n")
            for task_name, stats in results.items():
                f.write(f"\nTask: {task_name}\n")
                f.write(f"  First Occurrence: Epoch {stats['first_epoch']}\n")
                f.write(f"    Step 0: {stats['first_step_0']:.4f}\n")
                f.write(f"    Step 1: {stats['first_step_1']:.4f}\n")
                f.write(f"    Step 2: {stats['first_step_2']:.4f}\n")
                f.write(f"  Last Occurrence: Epoch {stats['last_epoch']}\n")
                f.write(f"    Step 0: {stats['last_step_0']:.4f}\n")
                f.write(f"    Step 1: {stats['last_step_1']:.4f}\n")
                f.write(f"    Step 2: {stats['last_step_2']:.4f}\n")
                f.write(f"  Total Occurrences: {stats['total_occurrences']}\n")
        
        print(f"Results saved to {self.output_dir}")
    
    def run_analysis(self):
        """Run the complete occurrence analysis pipeline"""
        print("Starting MAML Task Occurrence Analysis...")
        
        # Load data
        self.load_tracking_data()
        
        if not self.task_data:
            print("No task data found! Please check the tracking directory.")
            return
        
        # Analyze occurrences
        results = self.analyze_occurrences()
        
        if not results:
            print("No results generated! Check the data structure.")
            return
        
        # Create visualizations
        self.create_occurrence_plots(results)
        self.create_individual_task_plots(results)
        
        # Save results
        self.save_occurrence_results(results)
        
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MAML Task Occurrence Analysis')
    parser.add_argument('--tracking_dir', type=str, required=True,
                       help='Directory containing tracking data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = SimpleOccurrenceAnalyzer(args.tracking_dir, args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

