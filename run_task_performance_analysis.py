#!/usr/bin/env python3
"""
Run comprehensive task performance analysis for both UMI and TDL MAML training data.
"""

import os
import subprocess
import sys
from datetime import datetime

from paths import (
    default_combined_task_performance_out,
    default_inner_loop_tracking_tdl,
    default_inner_loop_tracking_umi,
    default_tdl_task_performance_out,
    default_umi_task_performance_out,
    repo_root,
)

def run_analysis_for_dataset(dataset_name, tracking_dir, output_dir):
    """Run task performance analysis for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    if not os.path.exists(tracking_dir):
        print(f"❌ Tracking directory not found: {tracking_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    cmd = [
        'python', 'task_performance_analysis.py',
        '--tracking_dir', tracking_dir,
        '--output_dir', output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root()))
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} analysis completed successfully!")
            print(f"📊 Results saved to: {output_dir}")
            return True
        else:
            print(f"❌ {dataset_name} analysis failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during {dataset_name} analysis: {e}")
        return False

def main():
    print("MAML Task Performance Analysis Pipeline")
    print("=" * 50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define datasets to analyze
    datasets = [
        {
            'name': 'UMI',
            'tracking_dir': default_inner_loop_tracking_umi(),
            'output_dir': default_umi_task_performance_out()
        },
        {
            'name': 'TDL', 
            'tracking_dir': default_inner_loop_tracking_tdl(),
            'output_dir': default_tdl_task_performance_out()
        }
    ]
    
    # Check which datasets have tracking data
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset['tracking_dir']):
            available_datasets.append(dataset)
            print(f"✅ Found {dataset['name']} tracking data: {dataset['tracking_dir']}")
        else:
            print(f"❌ No {dataset['name']} tracking data found: {dataset['tracking_dir']}")
    
    if not available_datasets:
        print("\n❌ No tracking data found for any dataset!")
        print("Please ensure you have run MAML training with tracking enabled.")
        return
    
    # Run analysis for each available dataset
    successful_analyses = []
    failed_analyses = []
    
    for dataset in available_datasets:
        success = run_analysis_for_dataset(
            dataset['name'], 
            dataset['tracking_dir'], 
            dataset['output_dir']
        )
        
        if success:
            successful_analyses.append(dataset['name'])
        else:
            failed_analyses.append(dataset['name'])
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    
    if successful_analyses:
        print(f"\n📊 Analysis results available in:")
        for dataset in available_datasets:
            if dataset['name'] in successful_analyses:
                print(f"  {dataset['name']}: {dataset['output_dir']}")
    
    # Create combined analysis if multiple datasets were successful
    if len(successful_analyses) > 1:
        print(f"\n🔄 Creating combined analysis...")
        create_combined_analysis(available_datasets, successful_analyses)

def create_combined_analysis(datasets, successful_analyses):
    """Create a combined analysis comparing UMI and TDL results"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        
        combined_dir = default_combined_task_performance_out()
        os.makedirs(combined_dir, exist_ok=True)
        
        # Load results from each dataset
        all_results = {}
        for dataset in datasets:
            if dataset['name'] in successful_analyses:
                csv_file = os.path.join(dataset['output_dir'], 'task_performance_analysis.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df['dataset'] = dataset['name']
                    all_results[dataset['name']] = df
        
        if len(all_results) < 2:
            print("Not enough datasets for combined analysis")
            return
        
        # Combine all results
        combined_df = pd.concat(all_results.values(), ignore_index=True)
        
        # Create combined comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Step 2 performance comparison
        for dataset_name, df in all_results.items():
            axes[0,0].hist(df['step2_mean'], alpha=0.6, label=f'{dataset_name}', bins=15)
        axes[0,0].set_xlabel('Mean Step 2 Loss')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Step 2 Performance Distribution')
        axes[0,0].legend()
        
        # Improvement comparison
        for dataset_name, df in all_results.items():
            axes[0,1].hist(df['improvement_pct'], alpha=0.6, label=f'{dataset_name}', bins=15)
        axes[0,1].set_xlabel('Improvement %')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Improvement Distribution')
        axes[0,1].legend()
        axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Box plot comparison
        step2_data = [df['step2_mean'].values for df in all_results.values()]
        step2_labels = list(all_results.keys())
        axes[1,0].boxplot(step2_data, labels=step2_labels)
        axes[1,0].set_ylabel('Step 2 Loss')
        axes[1,0].set_title('Step 2 Performance Box Plot')
        
        # Improvement box plot
        improvement_data = [df['improvement_pct'].values for df in all_results.values()]
        axes[1,1].boxplot(improvement_data, labels=step2_labels)
        axes[1,1].set_ylabel('Improvement %')
        axes[1,1].set_title('Improvement Box Plot')
        axes[1,1].axvline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, 'combined_dataset_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save combined results
        combined_df.to_csv(os.path.join(combined_dir, 'combined_task_performance.csv'), index=False)
        
        # Create summary statistics
        summary_stats = {}
        for dataset_name, df in all_results.items():
            summary_stats[dataset_name] = {
                'num_tasks': len(df),
                'avg_step2_loss': df['step2_mean'].mean(),
                'std_step2_loss': df['step2_mean'].std(),
                'avg_improvement': df['improvement_pct'].mean(),
                'std_improvement': df['improvement_pct'].std(),
                'positive_improvement_count': (df['improvement_pct'] > 0).sum()
            }
        
        # Save summary
        with open(os.path.join(combined_dir, 'combined_summary_report.txt'), 'w') as f:
            f.write("Combined MAML Task Performance Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for dataset_name, stats in summary_stats.items():
                f.write(f"{dataset_name} Dataset:\n")
                f.write(f"  Number of Tasks: {stats['num_tasks']}\n")
                f.write(f"  Average Step 2 Loss: {stats['avg_step2_loss']:.4f} ± {stats['std_step2_loss']:.4f}\n")
                f.write(f"  Average Improvement: {stats['avg_improvement']:.2f}% ± {stats['std_improvement']:.2f}%\n")
                f.write(f"  Tasks with Positive Improvement: {stats['positive_improvement_count']}/{stats['num_tasks']}\n\n")
        
        print(f"✅ Combined analysis created: {combined_dir}")
        
    except Exception as e:
        print(f"❌ Error creating combined analysis: {e}")

if __name__ == "__main__":
    main()


