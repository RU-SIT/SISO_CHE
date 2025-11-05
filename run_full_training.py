#!/usr/bin/env python3
"""
Run full MAML training with tracking for multiple epochs.
"""

import subprocess
import sys
import os

def run_full_training():
    """Run full MAML training with tracking."""
    
    print("="*60)
    print("RUNNING FULL MAML TRAINING WITH TRACKING")
    print("="*60)
    
    # Command to run MAML training with tracking
    cmd = [
        'python', 'MAML_trainer_with_tracking.py',
        '--root', 'Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak',
        '--device', 'cuda:0',
        '--save_init', 'SISO_UMi_init/std_scaler_interpolated_noleak',
        '--tracking_dir', 'inner_loop_tracking_data_umi',
        '--epoch', '100',  # Start with 100 epochs for testing
        '--n_way', '5',
        '--k_spt', '5', 
        '--k_qry', '5',
        '--batchsz', '8',
        '--meta_lr', '1e-4',
        '--update_lr', '1e-3',
        '--update_step', '2'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("This will train for 100 epochs and track inner loop losses...")
    print("Press Ctrl+C to stop if needed.")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        print("Partial training data may be available")
        return False

def run_analysis():
    """Run box plot analysis on the tracked data."""
    
    print("\n" + "="*60)
    print("RUNNING BOX PLOT ANALYSIS")
    print("="*60)
    
    cmd = [
        'python', 'box_plot_analysis.py',
        '--tracking_dir', 'inner_loop_tracking_data_umi',
        '--output_dir', 'inner_analysis_results_umi'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed with error: {e}")
        return False

if __name__ == '__main__':
    print("MAML TRAINING AND ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Step 1: Run training
    training_success = run_full_training()
    
    if training_success:
        # Step 2: Run analysis
        analysis_success = run_analysis()
        
        if analysis_success:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check the following directories for results:")
            print("- inner_loop_tracking_data_umi/ (tracking data)")
            print("- inner_analysis_results_umi/ (box plots and analysis)")
        else:
            print("\n⚠ Training completed but analysis failed")
    else:
        print("\n⚠ Training failed - check error messages above")
