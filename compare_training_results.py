#!/usr/bin/env python3
"""
Training Comparison: Random vs Pre-trained Initialization
==========================================================

This script helps you visualize and compare training results between:
1. MAML with random initialization (baseline)
2. MAML with pre-trained ChannelNet initialization (transfer learning)

Use this to demonstrate the benefits of transfer learning!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from glob import glob


def load_training_losses(results_dir, pattern="*training_loss_curve*.png"):
    """
    Load training loss data from saved plots or CSV files.
    
    Args:
        results_dir: Directory containing training results
        pattern: File pattern to match
        
    Returns:
        Dictionary with loss histories
    """
    # Try to find CSV or text files with loss data
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    
    losses = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'loss' in df.columns or 'training_loss' in df.columns:
                loss_col = 'loss' if 'loss' in df.columns else 'training_loss'
                losses[csv_file] = df[loss_col].values
        except:
            pass
    
    return losses


def parse_loss_from_log(log_file):
    """
    Extract training losses from log file output.
    
    Args:
        log_file: Path to training log file
        
    Returns:
        List of loss values
    """
    losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for lines like: "step: 0, training loss: 0.1234"
            if 'training loss:' in line:
                try:
                    loss_str = line.split('training loss:')[1].split(',')[0].strip()
                    losses.append(float(loss_str))
                except:
                    pass
    
    return losses


def create_comparison_plot(random_losses, pretrained_losses, output_path='comparison.png'):
    """
    Create a comparison plot showing both training curves.
    
    Args:
        random_losses: Loss values from random initialization
        pretrained_losses: Loss values from pre-trained initialization
        output_path: Where to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Full training curves
    plt.subplot(1, 2, 1)
    
    steps_random = list(range(len(random_losses)))
    steps_pretrained = list(range(len(pretrained_losses)))
    
    plt.plot(steps_random, random_losses, label='Random Initialization', 
             color='red', alpha=0.7, linewidth=2)
    plt.plot(steps_pretrained, pretrained_losses, label='Pre-trained Initialization', 
             color='blue', alpha=0.7, linewidth=2)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: First 500 steps (zoomed)
    plt.subplot(1, 2, 2)
    
    zoom_steps = 500
    plt.plot(steps_random[:zoom_steps], random_losses[:zoom_steps], 
             label='Random Init', color='red', alpha=0.7, linewidth=2)
    plt.plot(steps_pretrained[:zoom_steps], pretrained_losses[:zoom_steps], 
             label='Pre-trained Init', color='blue', alpha=0.7, linewidth=2)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'First {zoom_steps} Steps (Zoomed)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def calculate_metrics(random_losses, pretrained_losses):
    """
    Calculate comparison metrics between two training runs.
    
    Args:
        random_losses: Loss values from random initialization
        pretrained_losses: Loss values from pre-trained initialization
        
    Returns:
        Dictionary of comparison metrics
    """
    metrics = {}
    
    # Initial loss comparison
    metrics['initial_loss_random'] = random_losses[0]
    metrics['initial_loss_pretrained'] = pretrained_losses[0]
    metrics['initial_loss_improvement'] = (
        (random_losses[0] - pretrained_losses[0]) / random_losses[0] * 100
    )
    
    # Final loss comparison
    metrics['final_loss_random'] = random_losses[-1]
    metrics['final_loss_pretrained'] = pretrained_losses[-1]
    metrics['final_loss_improvement'] = (
        (random_losses[-1] - pretrained_losses[-1]) / random_losses[-1] * 100
    )
    
    # Convergence speed: steps to reach certain threshold
    threshold = random_losses[-1] * 1.1  # 110% of random's final loss
    
    steps_to_threshold_random = len(random_losses)
    for i, loss in enumerate(random_losses):
        if loss <= threshold:
            steps_to_threshold_random = i
            break
    
    steps_to_threshold_pretrained = len(pretrained_losses)
    for i, loss in enumerate(pretrained_losses):
        if loss <= threshold:
            steps_to_threshold_pretrained = i
            break
    
    metrics['steps_to_convergence_random'] = steps_to_threshold_random
    metrics['steps_to_convergence_pretrained'] = steps_to_threshold_pretrained
    
    if steps_to_threshold_random > 0:
        metrics['convergence_speedup'] = (
            steps_to_threshold_random / steps_to_threshold_pretrained
        )
    else:
        metrics['convergence_speedup'] = 1.0
    
    return metrics


def print_comparison_report(metrics):
    """
    Print a formatted comparison report.
    
    Args:
        metrics: Dictionary of comparison metrics
    """
    print("\n" + "="*70)
    print("TRANSFER LEARNING COMPARISON REPORT")
    print("="*70)
    
    print("\n📊 INITIAL PERFORMANCE (First Epoch)")
    print("-" * 70)
    print(f"Random Initialization:      {metrics['initial_loss_random']:.6f}")
    print(f"Pre-trained Initialization: {metrics['initial_loss_pretrained']:.6f}")
    print(f"Improvement:                {metrics['initial_loss_improvement']:.2f}% better")
    
    if metrics['initial_loss_improvement'] > 50:
        print("✓ Excellent! Pre-training provides much better starting point!")
    elif metrics['initial_loss_improvement'] > 20:
        print("✓ Good! Pre-training helps significantly.")
    else:
        print("⚠ Modest improvement. Check if architectures match well.")
    
    print("\n📈 FINAL PERFORMANCE (Last Epoch)")
    print("-" * 70)
    print(f"Random Initialization:      {metrics['final_loss_random']:.6f}")
    print(f"Pre-trained Initialization: {metrics['final_loss_pretrained']:.6f}")
    print(f"Improvement:                {metrics['final_loss_improvement']:.2f}% better")
    
    if metrics['final_loss_improvement'] > 15:
        print("✓ Excellent! Pre-training leads to better final model!")
    elif metrics['final_loss_improvement'] > 5:
        print("✓ Good! Pre-training improves final performance.")
    else:
        print("⚠ Small improvement. Both methods converge to similar solutions.")
    
    print("\n⚡ CONVERGENCE SPEED")
    print("-" * 70)
    print(f"Random Initialization:      {metrics['steps_to_convergence_random']} steps")
    print(f"Pre-trained Initialization: {metrics['steps_to_convergence_pretrained']} steps")
    print(f"Speedup:                    {metrics['convergence_speedup']:.2f}x faster")
    
    if metrics['convergence_speedup'] > 2:
        print("✓ Excellent! Pre-training reduces training time significantly!")
    elif metrics['convergence_speedup'] > 1.3:
        print("✓ Good! Pre-training accelerates convergence.")
    else:
        print("⚠ Similar convergence speed. Pre-training benefit is mainly in final performance.")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    overall_score = (
        (metrics['initial_loss_improvement'] > 20) +
        (metrics['final_loss_improvement'] > 5) +
        (metrics['convergence_speedup'] > 1.5)
    )
    
    if overall_score >= 2:
        print("✓✓ Transfer learning provides significant benefits!")
        print("Recommendation: USE pre-trained initialization for production.")
    elif overall_score == 1:
        print("✓ Transfer learning provides moderate benefits.")
        print("Recommendation: Consider pre-trained initialization for time savings.")
    else:
        print("⚠ Transfer learning provides limited benefits.")
        print("Recommendation: Verify that ChannelNet and MAML architectures match well.")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare training results: Random vs Pre-trained initialization'
    )
    parser.add_argument('--random_log', type=str,
                        help='Log file from random initialization training')
    parser.add_argument('--pretrained_log', type=str,
                        help='Log file from pre-trained initialization training')
    parser.add_argument('--random_losses', type=str,
                        help='CSV file with random init losses (alternative to log)')
    parser.add_argument('--pretrained_losses', type=str,
                        help='CSV file with pre-trained init losses (alternative to log)')
    parser.add_argument('--output', type=str, default='training_comparison.png',
                        help='Output path for comparison plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Training Comparison Tool")
    print("="*70)
    
    # Load losses from logs or CSVs
    random_losses = []
    pretrained_losses = []
    
    if args.random_log and os.path.exists(args.random_log):
        print(f"\nLoading random init data from: {args.random_log}")
        random_losses = parse_loss_from_log(args.random_log)
        print(f"  Found {len(random_losses)} data points")
    
    if args.pretrained_log and os.path.exists(args.pretrained_log):
        print(f"\nLoading pre-trained init data from: {args.pretrained_log}")
        pretrained_losses = parse_loss_from_log(args.pretrained_log)
        print(f"  Found {len(pretrained_losses)} data points")
    
    # Check if we have data
    if len(random_losses) == 0 or len(pretrained_losses) == 0:
        print("\n⚠ Error: Could not load training data.")
        print("\nUsage example:")
        print("  python compare_training_results.py \\")
        print("      --random_log training_log_random.txt \\")
        print("      --pretrained_log training_log_pretrained.txt")
        print("\nOr create example data:")
        print("  python compare_training_results.py --demo")
        return
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(random_losses, pretrained_losses, args.output)
    
    # Calculate and print metrics
    print("\nCalculating comparison metrics...")
    metrics = calculate_metrics(random_losses, pretrained_losses)
    print_comparison_report(metrics)


def demo_mode():
    """Create demo comparison with synthetic data."""
    print("\n🎮 DEMO MODE: Generating synthetic comparison\n")
    
    # Synthetic data showing typical transfer learning benefits
    steps = 1000
    random_losses = 0.5 * np.exp(-np.linspace(0, 3, steps)) + 0.05 + np.random.normal(0, 0.01, steps)
    pretrained_losses = 0.1 * np.exp(-np.linspace(0, 4, steps)) + 0.03 + np.random.normal(0, 0.005, steps)
    
    # Ensure decreasing
    random_losses = np.maximum.accumulate(random_losses[::-1])[::-1]
    pretrained_losses = np.maximum.accumulate(pretrained_losses[::-1])[::-1]
    
    create_comparison_plot(random_losses, pretrained_losses, 'demo_comparison.png')
    
    metrics = calculate_metrics(random_losses, pretrained_losses)
    print_comparison_report(metrics)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        main()

