#!/usr/bin/env python3
"""
Simple runner script for iMAML Inner Step Analysis

This script provides an easy way to run the iMAML inner step analysis
for different scenarios with various configurations.
"""

import os
import sys
import argparse
from imaml_inner_step_analyzer import run_imaml_analysis

def main():
    from pathlib import Path

    _WC_ROOT = Path(__file__).resolve().parents[2]
    if str(_WC_ROOT) not in sys.path:
        sys.path.insert(0, str(_WC_ROOT))
    from paths import default_dataset_tdl_siso_folder, default_dataset_umi_pspacing

    parser = argparse.ArgumentParser(description='Run iMAML Inner Step Analysis')
    
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
    
    # Training parameters
    parser.add_argument('--meta_steps', type=int, default=10, 
                       help='Number of meta training steps')
    parser.add_argument('--N_way', type=int, default=4, 
                        help='Number of ways for few-shot learning')
    parser.add_argument('--K_shot', type=int, default=5, 
                       help='Number of shots')
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("iMAML INNER STEP ANALYSIS RUNNER")
    print("="*80)
    print(f"Scenario(s): {args.scenario}")
    print(f"Meta steps: {args.meta_steps}")
    print(f"N-way: {args.N_way}, K-shot: {args.K_shot}")
    print(f"Inner steps: {args.n_steps}")
    print(f"Lambda: {args.lam}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Save directory: {args.save_dir}")
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
        
        # Create scenario-specific save directory
        scenario_save_dir = os.path.join(args.save_dir, f"{scenario.lower()}_analysis")
        
        try:
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
            print(f"\n{scenario} analysis completed successfully!")
            
        except Exception as e:
            print(f"\nError analyzing {scenario} scenario: {e}")
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
