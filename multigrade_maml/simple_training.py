#!/usr/bin/env python3
"""
Simple Multigrade MAML Training - Working Version
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigrade_maml_stair import MultigradeMAMLStair
from Data_Nshot import ChannelEstimationNShot

def main():
    print("🚀 Simple Multigrade MAML Training")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    # Training parameters
    EPOCHS = 30
    NUM_GRADES = 3
    EPOCHS_PER_GRADE = EPOCHS // NUM_GRADES
    
    print(f"Total epochs: {EPOCHS}")
    print(f"Epochs per grade: {EPOCHS_PER_GRADE}")
    print(f"Number of grades: {NUM_GRADES}")
    
    # Simple args
    class Args:
        def __init__(self):
            self.update_lr = 1e-3
            self.meta_lr = 1e-4
            self.n_way = 3  # Use 3-way for simplicity
            self.k_spt = 5
            self.k_qry = 5
            self.batchsz = 2
            self.update_step = 2
            self.num_grades = NUM_GRADES
    
    args = Args()
    
    # Simple config (smaller network for faster training)
    config = [
        ('conv2d', [16, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [16]),
        ('conv2d', [32, 16, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [32]),
        ('conv2d', [2, 32, 3, 3, 1, 1])
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    maml = MultigradeMAMLStair(args, config, num_grades=args.num_grades).to(device)
    total_params = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    # Create data loader
    db_train = ChannelEstimationNShot(
        "/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak",
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    print("✓ Data loader created")
    
    # Training storage
    all_losses = []
    grade_losses = {i: [] for i in range(args.num_grades)}
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 50)
    
    for step in range(EPOCHS):
        # Determine current grade
        current_grade = min(step // EPOCHS_PER_GRADE, args.num_grades - 1)
        
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
        for grade_idx in range(args.num_grades):
            current_loss = losses[grade_idx][-1]
            if hasattr(current_loss, 'item'):
                current_loss = current_loss.item()
            grade_losses[grade_idx].append(current_loss)
        
        # Main loss (last grade)
        main_loss = losses[args.num_grades - 1][-1]
        if hasattr(main_loss, 'item'):
            main_loss = main_loss.item()
        all_losses.append(main_loss)
        
        # Print progress
        if step % 5 == 0 or step == EPOCHS - 1:
            print(f"Step {step:2d}: Grade {current_grade + 1}, Main Loss: {main_loss:.6f}")
            for grade_idx in range(args.num_grades):
                grade_loss = losses[grade_idx][-1]
                if hasattr(grade_loss, 'item'):
                    grade_loss = grade_loss.item()
                print(f"  Grade {grade_idx + 1}: {grade_loss:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: All losses
    plt.subplot(2, 2, 1)
    for grade_idx in range(args.num_grades):
        plt.plot(grade_losses[grade_idx], label=f'Grade {grade_idx + 1}', alpha=0.7)
    plt.plot(all_losses, label='Main Loss', linestyle='--', linewidth=2, color='red')
    plt.title('Multigrade Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Grade-specific periods
    plt.subplot(2, 2, 2)
    colors = ['blue', 'green', 'orange']
    for grade_idx in range(args.num_grades):
        start_epoch = grade_idx * EPOCHS_PER_GRADE
        end_epoch = min((grade_idx + 1) * EPOCHS_PER_GRADE, len(grade_losses[grade_idx]))
        epochs = range(start_epoch, end_epoch)
        losses = grade_losses[grade_idx][start_epoch:end_epoch]
        plt.plot(epochs, losses, label=f'Grade {grade_idx + 1}', color=colors[grade_idx], linewidth=2)
    plt.title('Grade-Specific Training Periods')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss reduction per grade
    plt.subplot(2, 2, 3)
    grade_reductions = []
    for grade_idx in range(args.num_grades):
        start_epoch = grade_idx * EPOCHS_PER_GRADE
        end_epoch = min((grade_idx + 1) * EPOCHS_PER_GRADE, len(grade_losses[grade_idx]))
        if start_epoch < len(grade_losses[grade_idx]) and end_epoch <= len(grade_losses[grade_idx]):
            start_loss = grade_losses[grade_idx][start_epoch]
            end_loss = grade_losses[grade_idx][end_epoch - 1]
            reduction = start_loss - end_loss
            grade_reductions.append(reduction)
        else:
            grade_reductions.append(0)
    
    bars = plt.bar(range(1, args.num_grades + 1), grade_reductions, color=colors)
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
    plt.savefig('multigrade_training_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: multigrade_training_results.png")
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final losses:")
    for grade_idx in range(args.num_grades):
        final_loss = grade_losses[grade_idx][-1]
        print(f"  Grade {grade_idx + 1}: {final_loss:.6f}")
    print(f"  Overall: {all_losses[-1]:.6f}")
    
    # Calculate improvement
    initial_loss = all_losses[0]
    final_loss = all_losses[-1]
    improvement = initial_loss - final_loss
    print(f"  Improvement: {improvement:.6f} ({improvement/initial_loss*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()

