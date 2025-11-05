#!/usr/bin/env python3
"""
Run MAML training with tracking for TDL channel data.
This script will generate the inner_loop_tracking_data_tdl directory.
"""

import subprocess
import sys
import os

def run_tdl_training():
    """Run MAML training with tracking for TDL data."""
    
    print("="*60)
    print("RUNNING MAML TRAINING WITH TRACKING FOR TDL DATA")
    print("="*60)
    
    # Command to run MAML training with tracking for TDL data
    cmd = [
        'python', 'MAML_trainer_with_tracking.py',
        '--root', 'Sionna_datasets/ps2_p612/speed5/SISO_pspacing_4/speed5/nointerp',
        '--device', 'cuda:0',
        '--save_init', 'TDL_init/std_scaler_experiments',
        '--tracking_dir', 'inner_loop_tracking_data_tdl',
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
    print("This will train for 100 epochs and track inner loop losses for TDL data...")
    print("Press Ctrl+C to stop if needed.")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("✓ TDL Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ TDL Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ TDL Training interrupted by user")
        print("Partial training data may be available")
        return False

def run_tdl_analysis():
    """Run box plot analysis on the TDL tracked data."""
    
    print("\n" + "="*60)
    print("RUNNING BOX PLOT ANALYSIS FOR TDL DATA")
    print("="*60)
    
    cmd = [
        'python', 'box_plot_analysis.py',
        '--tracking_dir', 'inner_loop_tracking_data_tdl',
        '--output_dir', 'inner_analysis_results_tdl'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ TDL Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ TDL Analysis failed with error: {e}")
        return False

if __name__ == '__main__':
    print("MAML TRAINING AND ANALYSIS PIPELINE FOR TDL DATA")
    print("=" * 50)
    
    # Step 1: Run training
    training_success = run_tdl_training()
    
    if training_success:
        # Step 2: Run analysis
        analysis_success = run_tdl_analysis()
        
        if analysis_success:
            print("\n" + "="*60)
            print("TDL PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check the following directories for results:")
            print("- inner_loop_tracking_data_tdl/ (tracking data)")
            print("- inner_analysis_results_tdl/ (box plots and analysis)")
        else:
            print("\n⚠ TDL Training completed but analysis failed")
    else:
        print("\n⚠ TDL Training failed - check error messages above")



