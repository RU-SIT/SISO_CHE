#!/usr/bin/env python3
"""
Complete Analysis Pipeline for Advisor's Requirements

This script runs the complete pipeline to track inner loop losses during
MAML training and create box plots for each channel, as requested by the advisor.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def run_maml_training_with_tracking(args):
    """Run MAML training with loss tracking."""
    print("="*60)
    print("STEP 1: RUNNING MAML TRAINING WITH LOSS TRACKING")
    print("="*60)
    
    cmd = [
        'python', 'MAML_trainer_with_tracking.py',
        '--root', args.root,
        '--device', args.device,
        '--save_init', args.save_init,
        '--tracking_dir', args.tracking_dir,
        '--epoch', str(args.epoch),
        '--n_way', str(args.n_way),
        '--k_spt', str(args.k_spt),
        '--k_qry', str(args.k_qry),
        '--batchsz', str(args.batchsz),
        '--meta_lr', str(args.meta_lr),
        '--update_lr', str(args.update_lr),
        '--update_step', str(args.update_step)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Training for {args.epoch} epochs...")
    print(f"Tracking data will be saved to: {args.tracking_dir}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ MAML training with tracking completed successfully!")
        print(f"Training output saved to: {args.save_init}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during MAML training: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: MAML_trainer_with_tracking.py not found")
        return False

def run_box_plot_analysis(args):
    """Run box plot analysis on tracked data."""
    print("\n" + "="*60)
    print("STEP 2: RUNNING BOX PLOT ANALYSIS")
    print("="*60)
    
    cmd = [
        'python', 'box_plot_analysis.py',
        '--tracking_dir', args.tracking_dir,
        '--output_dir', args.output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Analyzing tracking data from: {args.tracking_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Box plot analysis completed successfully!")
        print(f"Analysis results saved to: {args.output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during box plot analysis: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: box_plot_analysis.py not found")
        return False

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        'MAML_trainer_with_tracking.py',
        'box_plot_analysis.py',
        'meta.py',
        'learner.py',
        'Data_Nshot.py',
        'utils.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    
    print("✓ All required files found")
    return True

def create_summary_report(args):
    """Create a summary report of the analysis."""
    print("\n" + "="*60)
    print("CREATING SUMMARY REPORT")
    print("="*60)
    
    report_file = Path(args.output_dir) / 'analysis_summary_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("MAML INNER LOOP LOSS ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {os.popen('date').read().strip()}\n")
        f.write(f"Training Epochs: {args.epoch}\n")
        f.write(f"Number of Ways: {args.n_way}\n")
        f.write(f"Support Set Size: {args.k_spt}\n")
        f.write(f"Query Set Size: {args.k_qry}\n")
        f.write(f"Batch Size: {args.batchsz}\n")
        f.write(f"Meta Learning Rate: {args.meta_lr}\n")
        f.write(f"Task Learning Rate: {args.update_lr}\n")
        f.write(f"Update Steps: {args.update_step}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("-" * 20 + "\n")
        
        # List generated files
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            for file in sorted(output_dir.glob("*")):
                f.write(f"- {file.name}\n")
        
        f.write("\nANALYSIS DESCRIPTION:\n")
        f.write("-" * 20 + "\n")
        f.write("This analysis tracks inner loop losses for each channel during MAML training.\n")
        f.write("For each channel that appears in a training task, we record:\n")
        f.write("- Step 0: Initial loss (before any adaptation)\n")
        f.write("- Step 1: Loss after 1 inner loop update\n")
        f.write("- Step 2: Loss after 2 inner loop updates\n\n")
        
        f.write("The analysis generates:\n")
        f.write("1. Individual channel box plots showing loss distributions\n")
        f.write("2. Combined box plots for all channels\n")
        f.write("3. Learning progression plots over training epochs\n")
        f.write("4. Statistical summaries with improvement metrics\n\n")
        
        f.write("INTERPRETATION GUIDE:\n")
        f.write("-" * 20 + "\n")
        f.write("✓ Good Learning Indicators:\n")
        f.write("  - Decreasing losses from Step 0 → Step 1 → Step 2\n")
        f.write("  - Consistent improvement across different channel appearances\n")
        f.write("  - Low variance in loss distributions\n\n")
        
        f.write("❌ Poor Learning Indicators:\n")
        f.write("  - Flat or increasing loss curves\n")
        f.write("  - High variance in loss distributions\n")
        f.write("  - Inconsistent learning across channel appearances\n\n")
    
    print(f"✓ Summary report created: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Complete MAML inner loop loss analysis pipeline')
    
    # Training parameters
    parser.add_argument('--root', type=str, 
                       default="Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak",
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_init', type=str, default="SISO_UMi_init/std_scaler_interpolated_noleak",
                       help='Directory to save model checkpoints')
    parser.add_argument('--tracking_dir', type=str, default="inner_loop_tracking_data_umi",
                       help='Directory to save tracking data')
    parser.add_argument('--output_dir', type=str, default="inner_analysis_results_umi",
                       help='Directory to save analysis results')
    
    # MAML parameters
    parser.add_argument('--epoch', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--n_way', type=int, default=5, help='Number of ways')
    parser.add_argument('--k_spt', type=int, default=5, help='Support set size')
    parser.add_argument('--k_qry', type=int, default=5, help='Query set size')
    parser.add_argument('--batchsz', type=int, default=8, help='Batch size')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Meta learning rate')
    parser.add_argument('--update_lr', type=float, default=1e-3, help='Task learning rate')
    parser.add_argument('--update_step', type=int, default=2, help='Inner loop update steps')
    
    # Analysis options
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training and only run analysis (if tracking data exists)')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip analysis and only run training')
    
    args = parser.parse_args()
    
    print("MAML INNER LOOP LOSS ANALYSIS PIPELINE")
    print("=" * 50)
    print("This pipeline will:")
    print("1. Run MAML training with loss tracking")
    print("2. Analyze tracked data and create box plots")
    print("3. Generate statistical summaries")
    print("4. Create comprehensive reports")
    print()
    
    # Check requirements
    if not check_requirements():
        print("Please ensure all required files are present.")
        return
    
    # Create output directories
    os.makedirs(args.tracking_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    # Step 1: Run MAML training with tracking
    if not args.skip_training:
        if not run_maml_training_with_tracking(args):
            success = False
    else:
        print("Skipping training step (--skip_training flag set)")
    
    # Step 2: Run box plot analysis
    if not args.skip_analysis and success:
        if not run_box_plot_analysis(args):
            success = False
    else:
        print("Skipping analysis step (--skip_analysis flag set or training failed)")
    
    # Step 3: Create summary report
    if success:
        create_summary_report(args)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"All results saved to: {args.output_dir}")
        print("\nGenerated files:")
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            for file in sorted(output_dir.glob("*")):
                print(f"  - {file.name}")
        
        print("\nNext steps:")
        print("1. Review the box plots to verify learning is happening")
        print("2. Check statistical summaries for improvement metrics")
        print("3. Share results with your advisor")
        
    else:
        print("\n" + "="*60)
        print("PIPELINE FAILED!")
        print("="*60)
        print("Please check the error messages above and try again.")

if __name__ == '__main__':
    main()
