#!/usr/bin/env python3
"""
Run box plot analysis on existing tracking data.
"""

import subprocess
import os

def run_analysis_on_existing_data():
    """Run analysis on the existing tracking data."""
    
    print("="*60)
    print("RUNNING BOX PLOT ANALYSIS ON EXISTING DATA")
    print("="*60)
    
    # Check if tracking data exists
    tracking_dir = "inner_loop_tracking_data"
    if not os.path.exists(tracking_dir):
        print(f"✗ Tracking directory {tracking_dir} not found!")
        return False
    
    # List files in tracking directory
    files = os.listdir(tracking_dir)
    print(f"Found tracking files: {files}")
    
    # Run the analysis
    cmd = [
        'python', 'box_plot_analysis.py',
        '--tracking_dir', tracking_dir,
        '--output_dir', 'analysis_results_existing_data'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Analysis completed successfully!")
        print("Check 'analysis_results_existing_data/' for results")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed with error: {e}")
        return False

if __name__ == '__main__':
    run_analysis_on_existing_data()

