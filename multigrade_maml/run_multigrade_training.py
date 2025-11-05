#!/usr/bin/env python3
"""
Complete Multigrade MAML Training Script
Easy to use with different configurations
"""

import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigrade_maml_stair import MultigradeMAMLStair
from Data_Nshot import ChannelEstimationNShot
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Multigrade MAML Training')
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs')
    parser.add_argument('--grades', type=int, default=3, help='Number of grades')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--root', type=str, 
                       default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak",
                       help='Path to data root directory')
    parser.add_argument('--n_way', type=int, default=4, help='Number of classes per task')
    parser.add_argument('--batchsz', type=int, default=2, help='Batch size')
    parser.add_argument('--k_spt', type=int, default=5, help='Number of support samples per task')
    parser.add_argument('--k_qry', type=int, default=5, help='Number of query samples per task')
    parser.add_argument('--update_step', type=int, default=2, help='Number of inner loop update steps')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Meta learning rate')
    parser.add_argument('--update_lr', type=float, default=1e-3, help='Task learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test (30 epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 30
        print(" Quick Test Mode (30 epochs)")
    else:
        print(" Full Training Mode")
    
    print("=" * 60)
    print("MULTIGRADE MAML TRAINING")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Grades: {args.grades}")
    print(f"Epochs per grade: {args.epochs // args.grades}")
    print(f"Device: {args.device}")
    print(f"Data root: {args.root}")
    print(f"N-way: {args.n_way}")
    print(f"Batch size: {args.batchsz}")
    print(f"K-shot (support): {args.k_spt}")
    print(f"K-query: {args.k_qry}")
    print(f"Update steps: {args.update_step}")
    print(f"Meta LR: {args.meta_lr}")
    print(f"Task LR: {args.update_lr}")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    
    # Simple args class
    class TrainingArgs:
        def __init__(self):
            self.update_lr = args.update_lr
            self.meta_lr = args.meta_lr
            self.n_way = args.n_way
            self.k_spt = args.k_spt
            self.k_qry = args.k_qry
            self.batchsz = args.batchsz
            self.update_step = args.update_step
            self.num_grades = args.grades
    
    training_args = TrainingArgs()
    
    # Network configuration
    # config = [
    #     ('conv2d', [32, 2, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('bn', [32]),
    #     ('conv2d', [64, 32, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('bn', [64]),
    #     ('conv2d', [128, 64, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('bn', [128]),
    #     ('conv2d', [64, 128, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('bn', [64]),
    #     ('conv2d', [32, 64, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('bn', [32]),
    #     ('conv2d', [2, 32, 3, 3, 1, 1])
    # ]
    
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
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    maml = MultigradeMAMLStair(training_args, config, num_grades=args.grades).to(device)
    total_params = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    # Create data loader
    db_train = ChannelEstimationNShot(
        root=args.root,
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    print("✓ Data loader created")
    
    # Extract dataset name from root path or use data type
    # Try to get a meaningful dataset name from the path
    dataset_name = "Unknown"
    if hasattr(db_train, 'data_type'):
        dataset_name = db_train.data_type
    
    # Try to extract more specific name from path (e.g., SISO-UMi, TDL-A, etc.)
    root_parts = args.root.split('/')
    for part in root_parts:
        if 'UMi' in part or 'umi' in part.lower():
            dataset_name = 'UMi'
            # Try to get more specific info (e.g., speed5)
            for p in root_parts:
                if 'speed' in p.lower():
                    dataset_name = f'UMi_{p}'
                    break
            break
        elif 'TDL' in part or 'tdl' in part.lower():
            dataset_name = 'TDL'
            # Try to get more specific info
            for p in root_parts:
                if 'speed' in p.lower() or any(char.isdigit() for char in p):
                    dataset_name = f'TDL_{p}'
                    break
            break
    
    print(f"Dataset identified: {dataset_name}")
    
    # Training storage
    all_losses = []
    grade_losses = {i: [] for i in range(args.grades)}
    epochs_per_grade = args.epochs // args.grades
    
    # Create dataset-specific directories
    results_base_dir = os.path.join("multigrade_maml_results", dataset_name)
    
    # Checkpoint directory (dataset-specific)
    ckpt_dir = os.path.join(
        results_base_dir,
        f"checkpoints_nway_{args.n_way}_grades_{args.grades}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {ckpt_dir}")
    
    # Plot directory (dataset-specific)
    plot_dir = os.path.join(results_base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot directory: {plot_dir}")
    
    print(f"\nStarting training...")
    print("=" * 60)
    
    for step in range(args.epochs):
        # Determine current grade
        current_grade = min(step // epochs_per_grade, args.grades - 1)
        
        # Get batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
         xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        
        # Convert to tensors
        x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
        y_qry_scld = torch.from_numpy(y_qry_scld).to(device)
        x_spt_scld = torch.from_numpy(x_spt_scld).to(device)
        y_spt_scld = torch.from_numpy(y_spt_scld).to(device)
        
        # Forward pass - only train the current grade (others are frozen)
        losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, current_grade=current_grade)
        
        # Store losses
        for grade_idx in range(args.grades):
            current_loss = losses[grade_idx][-1]
            if hasattr(current_loss, 'item'):
                current_loss = current_loss.item()
            grade_losses[grade_idx].append(current_loss)
        
        # Main loss (last grade)
        main_loss = losses[args.grades - 1][-1]
        if hasattr(main_loss, 'item'):
            main_loss = main_loss.item()
        all_losses.append(main_loss)
        
        # Print progress
        # if step % 50 == 0 or step == args.epochs - 1:
        print(f"Step {step:3d}: [Training Grade {current_grade + 1}]")
        for grade_idx in range(args.grades):
            grade_loss = losses[grade_idx][-1]
            if hasattr(grade_loss, 'item'):
                grade_loss = grade_loss.item()
            status = " ← TRAINING" if grade_idx == current_grade else " (frozen)"
            print(f"  Grade {grade_idx + 1}: {grade_loss:.6f}{status}")
        
        # Save checkpoint at the last epoch of each grade's training period
        # Check if this is the last epoch of the current grade
        grade_start_epoch = current_grade * epochs_per_grade
        grade_end_epoch = min((current_grade + 1) * epochs_per_grade - 1, args.epochs - 1)
        
        is_last_epoch_of_grade = (step == grade_end_epoch)
        is_final_epoch = (step == args.epochs - 1)
        
        if is_last_epoch_of_grade or is_final_epoch:
            # Get the current loss for this grade
            current_grade_loss = losses[current_grade][-1]
            if hasattr(current_grade_loss, 'item'):
                current_grade_loss = current_grade_loss.item()
            
            # Create checkpoint name
            ckpt_name = (
                f"MultigradeStairMAML_Grade{current_grade + 1}_"
                f"Step{step}_Loss{current_grade_loss:.6f}_"
                f"MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            
            # Save checkpoint
            torch.save({
                'step': step,
                'grade': current_grade,
                'state_dict': maml.state_dict(),
                'loss': current_grade_loss,
                'meta_lr': args.meta_lr,
                'update_lr': args.update_lr,
                'epochs_per_grade': epochs_per_grade,
                'total_epochs': args.epochs,
                'batchsz': args.batchsz,
                'n_way': args.n_way,
                'grades': args.grades,
            }, ckpt_path)
            
            print(f"  ✓ Checkpoint saved: Grade {current_grade + 1}, Step {step}, Loss: {current_grade_loss:.6f}")
            print(f"    Path: {ckpt_path}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: All losses
    plt.subplot(2, 2, 1)
    for grade_idx in range(args.grades):
        plt.plot(grade_losses[grade_idx], label=f'Grade {grade_idx + 1}', alpha=0.7)
    plt.plot(all_losses, label='Main Loss', linestyle='--', linewidth=2, color='red')
    plt.title(f'Multigrade Training Losses ({args.grades} Grades)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Grade-specific periods
    plt.subplot(2, 2, 2)
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for grade_idx in range(args.grades):
        start_epoch = grade_idx * epochs_per_grade
        end_epoch = min((grade_idx + 1) * epochs_per_grade, len(grade_losses[grade_idx]))
        epochs = range(start_epoch, end_epoch)
        losses = grade_losses[grade_idx][start_epoch:end_epoch]
        plt.plot(epochs, losses, label=f'Grade {grade_idx + 1}', 
                color=colors[grade_idx % len(colors)], linewidth=2)
    plt.title('Grade-Specific Training Periods')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss reduction per grade
    plt.subplot(2, 2, 3)
    grade_reductions = []
    for grade_idx in range(args.grades):
        start_epoch = grade_idx * epochs_per_grade
        end_epoch = min((grade_idx + 1) * epochs_per_grade, len(grade_losses[grade_idx]))
        if start_epoch < len(grade_losses[grade_idx]) and end_epoch <= len(grade_losses[grade_idx]):
            start_loss = grade_losses[grade_idx][start_epoch]
            end_loss = grade_losses[grade_idx][end_epoch - 1]
            reduction = start_loss - end_loss
            grade_reductions.append(reduction)
        else:
            grade_reductions.append(0)
    
    bars = plt.bar(range(1, args.grades + 1), grade_reductions, 
                   color=colors[:args.grades])
    plt.title('Loss Reduction per Grade')
    plt.xlabel('Grade')
    plt.ylabel('Loss Reduction')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 4: Overall progress
    plt.subplot(2, 2, 4)
    plt.plot(all_losses, label='Overall Progress', color='red', linewidth=2)
    plt.title('Overall Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot in dataset-specific directory
    plot_name = f'multigrade_{args.grades}grades_{args.epochs}epochs_training_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png'
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {plot_path}")
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final losses:")
    for grade_idx in range(args.grades):
        final_loss = grade_losses[grade_idx][-1]
        print(f"  Grade {grade_idx + 1}: {final_loss:.6f}")
    print(f"  Overall: {all_losses[-1]:.6f}")
    
    # Calculate improvement (only from the last grade's training period)
    # Find when the last grade started training
    last_grade_start = (args.grades - 1) * epochs_per_grade
    if last_grade_start < len(all_losses):
        initial_loss = grade_losses[args.grades - 1][last_grade_start]
        final_loss = all_losses[-1]
        if initial_loss > 0:
            improvement = initial_loss - final_loss
            print(f"  Last Grade Improvement: {improvement:.6f} ({improvement/initial_loss*100:.1f}%)")
    
    # List all saved checkpoints and plots
    print("\n" + "=" * 60)
    print(" SAVED FILES")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Results base directory: {results_base_dir}")
    print(f"\nCheckpoint directory: {ckpt_dir}")
    print(f"Plot directory: {plot_dir}")
    
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    checkpoint_files.sort()  # Sort by name
    
    if checkpoint_files:
        print(f"\nTotal checkpoints saved: {len(checkpoint_files)}")
        print("\nCheckpoints by grade:")
        for grade_idx in range(args.grades):
            grade_checkpoints = [f for f in checkpoint_files if f'Grade{grade_idx + 1}_' in f]
            if grade_checkpoints:
                print(f"\n  Grade {grade_idx + 1}:")
                for ckpt_file in grade_checkpoints:
                    ckpt_full_path = os.path.join(ckpt_dir, ckpt_file)
                    # Extract step and loss from filename
                    parts = ckpt_file.split('_')
                    step = [p for p in parts if p.startswith('Step')][0].replace('Step', '')
                    loss = [p for p in parts if p.startswith('Loss')][0].replace('Loss', '').replace('.pth.tar', '')
                    print(f"    - Step {step}, Loss: {loss}")
                    print(f"      File: {ckpt_file}")
    else:
        print("No checkpoints found in directory.")
    
    # List saved plots
    plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    if plot_files:
        print(f"\nSaved plots:")
        for plot_file in sorted(plot_files):
            plot_full_path = os.path.join(plot_dir, plot_file)
            print(f"  - {plot_file}")
            print(f"    Path: {plot_full_path}")
    else:
        print("\nNo plots found in directory.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

