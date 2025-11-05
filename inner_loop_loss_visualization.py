#!/usr/bin/env python3
"""
Inner Loop Loss Visualization for MAML Framework

This script loads a trained MAML model and visualizes the loss curve during
the inner loop adaptation steps to verify that learning is happening.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

# Import your existing modules
from Data_Nshot import ChannelEstimationNShot
from meta import Meta
from utils import Utils


class InnerLoopTracker:
    """Tracks and visualizes inner loop losses for MAML."""
    
    def __init__(self, maml_model, device):
        self.maml = maml_model
        self.device = device
        self.inner_losses = []
        
    def track_inner_loop(self, x_qry, y_qry, x_spt, y_spt):
        """
        Track inner loop losses for a single meta-learning task.
        
        Args:
            x_qry: Query set data [batchsz, setsz, c_, h, w]
            y_qry: Query set labels [batchsz, setsz, c_, h, w] 
            x_spt: Support set data [batchsz, setsz, c_, h, w]
            y_spt: Support set labels [batchsz, setsz, c_, h, w]
        """
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_qry.size(1)
        
        # Initialize loss tracking
        task_losses = []
        
        # Process each task in the batch
        for i in range(self.maml.n_way):
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            x_spt_i = x_spt[i].view(querysz, c_, h, w)
            y_spt_i = y_spt[i].view(querysz, c_, h, w)
            
            # Track losses for this task
            task_step_losses = []
            
            # Step 0: Initial loss (before any adaptation)
            with torch.no_grad():
                logits_0 = self.maml.net(x_spt_i, vars=None, bn_training=True)
                logits_0 = logits_0.view(querysz, c_, h, w)
                loss_0 = F.mse_loss(logits_0, y_spt_i)
                task_step_losses.append(loss_0.item())
                print(f"Task {i}, Step 0 (initial): Loss = {loss_0.item():.6f}")
            
            # Inner loop adaptation
            fast_weights = None
            for k in range(self.maml.update_step):
                if k == 0:
                    # First adaptation step
                    logits = self.maml.net(x_qry_i, vars=None, bn_training=True)
                    loss = F.mse_loss(logits, y_qry_i)
                    grad = torch.autograd.grad(loss, self.maml.net.parameters(), retain_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.maml.update_lr * p[0], 
                                          zip(grad, self.maml.net.parameters())))
                else:
                    # Subsequent adaptation steps
                    logits = self.maml.net(x_qry_i, fast_weights, bn_training=True)
                    loss = F.mse_loss(logits, y_qry_i)
                    grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.maml.update_lr * p[0], 
                                          zip(grad, fast_weights)))
                
                # Evaluate on support set with current fast weights
                with torch.no_grad():
                    logits_s = self.maml.net(x_spt_i, fast_weights, bn_training=True)
                    logits_s = logits_s.view(querysz, c_, h, w)
                    loss_s = F.mse_loss(logits_s, y_spt_i)
                    task_step_losses.append(loss_s.item())
                    print(f"Task {i}, Step {k+1}: Loss = {loss_s.item():.6f}")
            
            task_losses.append(task_step_losses)
        
        # Average losses across all tasks
        avg_losses = np.mean(task_losses, axis=0)
        self.inner_losses.append(avg_losses)
        
        return avg_losses, task_losses


def load_maml_checkpoint(checkpoint_path, args, config):
    """Load MAML model from checkpoint."""
    print(f"Loading MAML checkpoint from: {checkpoint_path}")
    
    # Create MAML model
    maml = Meta(args, config)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        maml.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demonstration")
    
    return maml


def visualize_inner_loop_losses(args):
    """Main function to visualize inner loop losses."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define the model configuration (same as in MAML_trainer.py)
    config = [
        ('conv2d', [64, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [64]),
        ('conv2d', [256, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [512, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [512]),
        ('conv2d', [256, 512, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [32, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [32]),
        ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [args.batchsz]),
        ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    ]
    
    # Load MAML model
    maml = load_maml_checkpoint(args.checkpoint_path, args, config)
    maml.to(device)
    maml.eval()  # Set to evaluation mode
    
    # Create data loader
    db_train = ChannelEstimationNShot(
        args.root,
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    
    # Create inner loop tracker
    tracker = InnerLoopTracker(maml, device)
    
    print(f"\nAnalyzing inner loop losses for {args.num_tasks} meta-learning tasks...")
    print(f"Update steps per task: {args.update_step}")
    print(f"Update learning rate: {args.update_lr}")
    
    # Track losses for multiple tasks
    all_task_losses = []
    all_avg_losses = []
    
    for task_idx in range(args.num_tasks):
        print(f"\n--- Task {task_idx + 1} ---")
        
        # Get a batch of data
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
         xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        
        # Convert to tensors and move to device
        x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
        y_qry_scld = torch.from_numpy(y_qry_scld).to(device)
        x_spt_scld = torch.from_numpy(x_spt_scld).to(device)
        y_spt_scld = torch.from_numpy(y_spt_scld).to(device)
        
        # Track inner loop losses
        avg_losses, task_losses = tracker.track_inner_loop(
            x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld
        )
        
        all_avg_losses.append(avg_losses)
        all_task_losses.extend(task_losses)
    
    # Create visualizations
    create_loss_visualizations(all_avg_losses, all_task_losses, args)
    
    print(f"\nVisualization complete! Check the output directory: {args.output_dir}")


def create_loss_visualizations(all_avg_losses, all_task_losses, args):
    """Create and save loss visualizations."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Average loss curve across all tasks
    plt.figure(figsize=(12, 8))
    
    # Plot individual task curves
    for i, task_losses in enumerate(all_task_losses):
        steps = range(len(task_losses))
        plt.plot(steps, task_losses, alpha=0.3, color='lightblue', linewidth=1)
    
    # Plot average curve
    if all_avg_losses:
        avg_across_tasks = np.mean(all_avg_losses, axis=0)
        steps = range(len(avg_across_tasks))
        plt.plot(steps, avg_across_tasks, color='red', linewidth=3, 
                label=f'Average across {len(all_avg_losses)} tasks')
        
        # Add improvement annotation
        if len(avg_across_tasks) > 1:
            improvement = avg_across_tasks[0] - avg_across_tasks[-1]
            improvement_pct = (improvement / avg_across_tasks[0]) * 100
            plt.annotate(f'Improvement: {improvement:.4f} ({improvement_pct:.1f}%)',
                        xy=(len(avg_across_tasks)-1, avg_across_tasks[-1]),
                        xytext=(len(avg_across_tasks)//2, avg_across_tasks[0]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=12, ha='center')
    
    plt.title(f'MAML Inner Loop Loss Curves\n'
              f'Update Steps: {args.update_step}, Update LR: {args.update_lr}',
              fontsize=16)
    plt.xlabel('Inner Loop Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(args.output_dir, 
                              f'inner_loop_loss_curves_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed analysis plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: All individual task curves
    plt.subplot(2, 2, 1)
    for i, task_losses in enumerate(all_task_losses):
        steps = range(len(task_losses))
        plt.plot(steps, task_losses, alpha=0.7, linewidth=1.5, label=f'Task {i+1}')
    plt.title('Individual Task Loss Curves')
    plt.xlabel('Inner Loop Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Average with error bars
    plt.subplot(2, 2, 2)
    if all_avg_losses:
        avg_losses = np.mean(all_avg_losses, axis=0)
        std_losses = np.std(all_avg_losses, axis=0)
        steps = range(len(avg_losses))
        
        plt.errorbar(steps, avg_losses, yerr=std_losses, 
                   color='red', linewidth=2, capsize=5, capthick=2)
        plt.title('Average Loss with Standard Deviation')
        plt.xlabel('Inner Loop Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    # Subplot 3: Loss reduction analysis
    plt.subplot(2, 2, 3)
    if all_avg_losses:
        initial_losses = [losses[0] for losses in all_avg_losses]
        final_losses = [losses[-1] for losses in all_avg_losses]
        improvements = [init - final for init, final in zip(initial_losses, final_losses)]
        improvement_pcts = [(imp / init) * 100 for imp, init in zip(improvements, initial_losses)]
        
        plt.bar(range(len(improvement_pcts)), improvement_pcts, alpha=0.7, color='green')
        plt.title('Loss Reduction Percentage per Task')
        plt.xlabel('Task Index')
        plt.ylabel('Improvement (%)')
        plt.grid(True, alpha=0.3)
    
    # Subplot 4: Learning rate effect analysis
    plt.subplot(2, 2, 4)
    if all_avg_losses:
        # Calculate learning curves (loss reduction per step)
        learning_rates = []
        for task_losses in all_avg_losses:
            if len(task_losses) > 1:
                lr_curve = []
                for i in range(1, len(task_losses)):
                    lr = (task_losses[i-1] - task_losses[i]) / task_losses[i-1]
                    lr_curve.append(lr)
                learning_rates.append(lr_curve)
        
        if learning_rates:
            avg_lr_curve = np.mean(learning_rates, axis=0)
            steps = range(1, len(avg_lr_curve) + 1)
            plt.plot(steps, avg_lr_curve, 'o-', color='purple', linewidth=2, markersize=6)
            plt.title('Effective Learning Rate per Step')
            plt.xlabel('Inner Loop Step')
            plt.ylabel('Relative Loss Reduction')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed analysis
    detailed_output_path = os.path.join(args.output_dir, 
                                       f'detailed_inner_loop_analysis_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png')
    plt.savefig(detailed_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("INNER LOOP LEARNING ANALYSIS SUMMARY")
    print("="*60)
    
    if all_avg_losses:
        all_initial_losses = [losses[0] for losses in all_avg_losses]
        all_final_losses = [losses[-1] for losses in all_avg_losses]
        
        print(f"Number of tasks analyzed: {len(all_avg_losses)}")
        print(f"Average initial loss: {np.mean(all_initial_losses):.6f} ± {np.std(all_initial_losses):.6f}")
        print(f"Average final loss: {np.mean(all_final_losses):.6f} ± {np.std(all_final_losses):.6f}")
        
        improvements = [init - final for init, final in zip(all_initial_losses, all_final_losses)]
        improvement_pcts = [(imp / init) * 100 for imp, init in zip(improvements, all_initial_losses)]
        
        print(f"Average improvement: {np.mean(improvements):.6f} ± {np.std(improvements):.6f}")
        print(f"Average improvement %: {np.mean(improvement_pcts):.2f}% ± {np.std(improvement_pcts):.2f}%")
        
        # Check if learning is happening
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        print(f"Tasks with positive improvement: {positive_improvements}/{len(improvements)}")
        
        if np.mean(improvement_pcts) > 1.0:
            print("✓ LEARNING IS HAPPENING: Significant loss reduction observed")
        else:
            print("⚠ LEARNING UNCERTAIN: Minimal loss reduction observed")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize MAML inner loop losses')
    
    # Model and data arguments
    parser.add_argument('--root', type=str, 
                       default="Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak",
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to MAML checkpoint file')
    
    # MAML parameters
    parser.add_argument('--n_way', type=int, default=5, help='Number of ways')
    parser.add_argument('--k_spt', type=int, default=5, help='Support set size')
    parser.add_argument('--k_qry', type=int, default=5, help='Query set size')
    parser.add_argument('--batchsz', type=int, default=8, help='Batch size')
    parser.add_argument('--update_step', type=int, default=2, help='Inner loop update steps')
    parser.add_argument('--update_lr', type=float, default=1e-3, help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Meta learning rate')
    
    # Visualization parameters
    parser.add_argument('--num_tasks', type=int, default=5, 
                       help='Number of meta-learning tasks to analyze')
    parser.add_argument('--output_dir', type=str, default='inner_loop_analysis',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the visualization
    visualize_inner_loop_losses(args)


if __name__ == '__main__':
    main()
