#!/usr/bin/env python3
"""
Fixed runner script for iMAML Inner Step Analysis

This script automatically detects the dataset size and adjusts parameters accordingly.
"""

import os
import sys
import argparse
import numpy as np
from imaml_inner_step_analyzer import run_imaml_analysis

def detect_dataset_parameters(data_dir):
    """Detect dataset parameters automatically."""
    print(f"🔍 Detecting dataset parameters from: {data_dir}")
    
    try:
        # Load data dictionary to check available files
        data_dict_path = os.path.join(data_dir, 'channel_data_dict.npy')
        data_dict = np.load(data_dict_path, allow_pickle=True).item()
        file_names = list(data_dict.keys())
        
        print(f"  - Found {len(file_names)} files: {file_names}")
        
        # Check data shape to determine samples per file
        first_file = file_names[0]
        first_data = data_dict[first_file]
        samples_per_file = first_data.shape[0]
        
        print(f"  - Samples per file: {samples_per_file}")
        
        # Determine optimal parameters
        total_files = len(file_names)
        train_files = min(4, total_files - 1)  # Leave at least 1 for testing
        test_files = total_files - train_files
        
        # Determine max N_way based on available train files
        max_n_way = train_files
        
        # Determine max K_shot based on available samples
        max_k_shot = min(20, samples_per_file // 4)  # Use at most 1/4 of samples per file
        
        print(f"  - Recommended parameters:")
        print(f"    - N_way: {max_n_way} (max: {max_n_way})")
        print(f"    - K_shot: {min(10, max_k_shot)} (max: {max_k_shot})")
        print(f"    - Train files: {train_files}")
        print(f"    - Test files: {test_files}")
        
        return {
            'n_way': max_n_way,
            'k_shot': min(10, max_k_shot),
            'train_files': train_files,
            'test_files': test_files,
            'total_files': total_files,
            'samples_per_file': samples_per_file
        }
        
    except Exception as e:
        print(f"❌ Error detecting dataset parameters: {e}")
        return None

def main():
    from pathlib import Path

    _WC_ROOT = Path(__file__).resolve().parents[2]
    if str(_WC_ROOT) not in sys.path:
        sys.path.insert(0, str(_WC_ROOT))
    from paths import default_dataset_tdl_siso_folder, default_dataset_umi_pspacing

    parser = argparse.ArgumentParser(description='Run iMAML Inner Step Analysis (Auto-detection)')
    
    # Data paths
    parser.add_argument('--umi_data_dir', type=str, 
                       default=default_dataset_umi_pspacing(),
                       help='Path to UMi dataset')
    parser.add_argument('--tdl_data_dir', type=str, 
                       default=default_dataset_tdl_siso_folder(),
                       help='Path to TDL dataset')
    
    # Analysis parameters
    parser.add_argument('--scenario', type=str, choices=['UMi', 'TDL', 'both'], 
                       default='UMi', help='Scenario to analyze')
    parser.add_argument('--save_dir', type=str, 
                       default="./imaml_analysis_results", 
                       help='Directory to save results')
    
    # Training parameters (can be overridden)
    parser.add_argument('--meta_steps', type=int, default=10, 
                       help='Number of meta training steps')
    parser.add_argument('--N_way', type=int, default=None, 
                       help='Number of ways for few-shot learning (auto-detected if not specified)')
    parser.add_argument('--K_shot', type=int, default=None, 
                       help='Number of shots (auto-detected if not specified)')
    parser.add_argument('--inner_lr', type=float, default=1e-3, 
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-4, 
                       help='Outer loop learning rate')
    parser.add_argument('--n_steps', type=int, default=2, 
                       help='Number of inner steps')
    parser.add_argument('--lam', type=float, default=2.0, 
                       help='Regularization parameter')
    
    # System parameters
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU for training')
    parser.add_argument('--no_gpu', dest='use_gpu', action='store_false',
                       help='Disable GPU usage')
    parser.add_argument('--auto_detect', action='store_true', default=True,
                       help='Auto-detect dataset parameters')
    
    args = parser.parse_args()
    
    print("="*80)
    print("iMAML INNER STEP ANALYSIS RUNNER (FIXED)")
    print("="*80)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    scenarios_to_run = []
    if args.scenario == 'both':
        scenarios_to_run = ['UMi', 'TDL']
    else:
        scenarios_to_run = [args.scenario]
    
    for scenario in scenarios_to_run:
        print(f"\n{'='*60}")
        print(f"ANALYZING {scenario.upper()} SCENARIO")
        print(f"{'='*60}")
        
        # Set data directory
        if scenario == 'UMi':
            data_dir = args.umi_data_dir
        else:  # TDL
            data_dir = args.tdl_data_dir
        
        # Auto-detect parameters if requested
        n_way = args.N_way
        k_shot = args.K_shot
        
        if args.auto_detect:
            print(f"🔍 Auto-detecting parameters for {scenario}...")
            params = detect_dataset_parameters(data_dir)
            
            if params is None:
                print(f"❌ Failed to detect parameters for {scenario}. Skipping...")
                continue
            
            # Use detected parameters if not specified
            if n_way is None:
                n_way = params['n_way']
            if k_shot is None:
                k_shot = params['k_shot']
            
            print(f"✅ Using parameters: N_way={n_way}, K_shot={k_shot}")
        else:
            # Use provided parameters or defaults
            n_way = n_way or 4
            k_shot = k_shot or 5
            print(f"📋 Using provided parameters: N_way={n_way}, K_shot={k_shot}")
        
        # Create scenario-specific save directory
        scenario_save_dir = os.path.join(args.save_dir, f"{scenario.lower()}_analysis")
        
        try:
            print(f"\n🚀 Starting analysis...")
            print(f"  - Data directory: {data_dir}")
            print(f"  - Save directory: {scenario_save_dir}")
            print(f"  - N_way: {n_way}, K_shot: {k_shot}")
            print(f"  - Meta steps: {args.meta_steps}")
            print(f"  - Lambda: {args.lam}")
            
            run_imaml_analysis(
                data_dir=data_dir,
                scenario_name=scenario,
                save_dir=scenario_save_dir,
                meta_steps=args.meta_steps,
                n_way=n_way,
                k_shot=k_shot,
                inner_lr=args.inner_lr,
                outer_lr=args.outer_lr,
                n_steps=args.n_steps,
                lam=args.lam,
                use_gpu=args.use_gpu
            )
            print(f"\n✅ {scenario} analysis completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error analyzing {scenario} scenario: {e}")
            print(f"Continuing with next scenario...")
            continue
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.save_dir}")
    
    # Print summary of generated files
    print(f"\nGenerated files:")
    for root, dirs, files in os.walk(args.save_dir):
        for file in files:
            if file.endswith(('.png', '.json', '.csv', '.txt')):
                rel_path = os.path.relpath(os.path.join(root, file), args.save_dir)
                print(f"  - {rel_path}")

if __name__ == '__main__':
    main()
