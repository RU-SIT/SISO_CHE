#!/usr/bin/env python3
"""
iMAML Trainer with Inner Loop Loss Tracking

This script modifies the original iMAML trainer to track inner loop losses
for each channel across all training epochs, similar to the MAML implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import utils as utils
import random
import time as timer
import pickle
import argparse
import pathlib
import os
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
from learner_model import Learner
from learner_model import make_fc_network, make_conv_network
from utils import DataLog
import sys
import json
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to access Wireless_communication modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from paths import default_dataset_umi_siso_folder, default_imaml_save_dir
from Data_Nshot import ChannelEstimationNShot

# Fix the relative import issue by adding the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

class LossTracker:
    """Tracks inner loop losses for each channel during iMAML training."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.tracking_data = []
        self.channel_stats = {}
        os.makedirs(save_dir, exist_ok=True)
        
    def track_losses(self, epoch, channel_names, losses_s, task_idx=0):
        """
        Track inner loop losses for a specific task (first and last steps only for iMAML).
        
        Args:
            epoch: Current training epoch
            channel_names: List of channel names for this task
            losses_s: List of loss dictionaries for each task
            task_idx: Task index within the batch
        """
        for i, channel_name in enumerate(channel_names):
            if i < len(losses_s):
                # Extract losses for this channel (losses_s is a list of dictionaries)
                task_loss = losses_s[i]
                
                # For iMAML, we only track first and last steps
                first_loss = float(task_loss.get('first_step', 0.0))
                last_loss = float(task_loss.get('last_step', 0.0))
                
                # Store tracking data (only first and last steps)
                tracking_entry = {
                    'epoch': epoch,
                    'task_idx': task_idx,
                    'channel_name': channel_name,
                    'first_step_loss': first_loss,
                    'last_step_loss': last_loss,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.tracking_data.append(tracking_entry)
                
                # Update channel statistics
                if channel_name not in self.channel_stats:
                    self.channel_stats[channel_name] = {
                        'total_appearances': 0,
                        'first_step_losses': [],
                        'last_step_losses': [],
                        'epochs': []
                    }
                
                self.channel_stats[channel_name]['total_appearances'] += 1
                self.channel_stats[channel_name]['first_step_losses'].append(first_loss)
                self.channel_stats[channel_name]['last_step_losses'].append(last_loss)
                self.channel_stats[channel_name]['epochs'].append(epoch)
    
    def save_tracking_data(self, epoch):
        """Save tracking data to files."""
        # Save detailed tracking data
        tracking_file = os.path.join(self.save_dir, f'tracking_data_epoch_{epoch}.json')
        with open(tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.tracking_data:
            df = pd.DataFrame(self.tracking_data)
            csv_file = os.path.join(self.save_dir, f'tracking_data_epoch_{epoch}.csv')
            df.to_csv(csv_file, index=False)
        
        # Save channel statistics
        stats_file = os.path.join(self.save_dir, f'channel_stats_epoch_{epoch}.json')
        with open(stats_file, 'w') as f:
            json.dump(self.channel_stats, f, indent=2)
    
    def get_summary_stats(self):
        """Get summary statistics for all channels (first and last steps only for iMAML)."""
        summary = {}
        for channel_name, stats in self.channel_stats.items():
            summary[channel_name] = {
                'total_appearances': stats['total_appearances'],
                'first_step_mean': np.mean(stats['first_step_losses']),
                'first_step_std': np.std(stats['first_step_losses']),
                'last_step_mean': np.mean(stats['last_step_losses']),
                'last_step_std': np.std(stats['last_step_losses']),
                'improvement_first_to_last': np.mean(stats['first_step_losses']) - np.mean(stats['last_step_losses'])
            }
        return summary

def main():
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    logger = DataLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='iMAML on Wireless Channel Estimation with Loss Tracking')
    parser.add_argument('--data_dir', type=str, default=default_dataset_umi_siso_folder(), help='location of the dataset')
    parser.add_argument('--N_way', type=int, default=2, help='number of classes for few-shot learning tasks')
    parser.add_argument('--K_shot', type=int, default=10, help='number of instances for few-shot learning tasks')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help='inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-4, help='outer loop learning rate')
    parser.add_argument('--n_steps', type=int, default=2, help='number of steps in inner loop')
    parser.add_argument('--meta_steps', type=int, default=4000, help='number of meta steps')
    parser.add_argument('--task_mb_size', type=int, default=8)
    parser.add_argument('--lam', type=float, default=2.0, help='regularization in inner steps')
    parser.add_argument('--cg_steps', type=int, default=1)  # Reduced to avoid CG solver issues
    parser.add_argument('--cg_damping', type=float, default=1.0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default=default_imaml_save_dir())
    parser.add_argument('--tracking_dir', type=str, default="inner_loop_tracking_data_imaml")
    parser.add_argument('--load_agent', type=str, default=None)
    parser.add_argument('--lam_lr', type=float, default=0.0, help='learning rate for lambda')
    parser.add_argument('--lam_min', type=float, default=0.0, help='minimum lambda value')
    args = parser.parse_args()
    logger.log_exp_args(args)

    # Create loss tracker
    loss_tracker = LossTracker(args.tracking_dir)

    print("="*60)
    print("iMAML TRAINING WITH INNER LOOP LOSS TRACKING")
    print("="*60)
    print(f"Tracking directory: {args.tracking_dir}")
    print(f"Meta steps: {args.meta_steps}")
    print(f"Number of ways: {args.N_way}")
    print(f"K-shot: {args.K_shot}")
    print(f"Lambda: {args.lam}")

    # Load Data
    print("\nGenerating tasks ...... ")
    dataset = ChannelEstimationNShot(
        root=args.data_dir,
        batchsz=args.task_mb_size,
        n_way=args.N_way,
        k_shot=args.K_shot,
        k_query=args.K_shot  # Query set size same as support set
    )

    # Initialize Model
    if args.load_agent is None:
        learner_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=args.task_mb_size)
        fast_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=args.task_mb_size)

        meta_learner = Learner(model=learner_net,
                               loss_function=torch.nn.MSELoss(),
                               inner_lr=args.inner_lr,
                               outer_lr=args.outer_lr,
                               GPU=args.use_gpu)
        fast_learner = Learner(model=fast_net,
                               loss_function=torch.nn.MSELoss(),
                               inner_lr=args.inner_lr,
                               outer_lr=args.outer_lr,
                               GPU=args.use_gpu)
    else:
        meta_learner = pickle.load(open(args.load_agent, 'rb'))
        meta_learner.set_params(meta_learner.get_params())
        fast_learner = pickle.load(open(args.load_agent, 'rb'))
        fast_learner.set_params(fast_learner.get_params())
        
    # Count total trainable parameters
    tmp = filter(lambda x: x.requires_grad, meta_learner.model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(f"\nModel: {meta_learner}")
    print(f"Total trainable parameters: {num}")

    init_params = meta_learner.get_params()
    device = 'cuda:0' if args.use_gpu else 'cpu'
    if args.use_gpu:
        torch.cuda.set_device(0)  # Explicitly use GPU 0
        torch.cuda.empty_cache()  # Clear GPU cache
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    lam = torch.tensor(args.lam, device=device)

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Train
    print("\nTraining model with loss tracking...")
    print(f"Starting training for {args.meta_steps} meta steps...")
    print(f"Parameters: N_way={args.N_way}, K_shot={args.K_shot}, n_steps={args.n_steps}")
    print(f"CG steps: {args.cg_steps}, Lambda: {args.lam}")
    losses = np.zeros((args.meta_steps, 4))
    
    for outstep in tqdm(range(args.meta_steps)):
        print(f"\n=== META STEP {outstep}/{args.meta_steps} ===")
        print(f"Time: {timer.strftime('%H:%M:%S')}")
        
        # Get next batch
        print("  Loading next batch...")
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
            xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
            (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
            qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = dataset.next(mode='train')
        
        print(f"  Batch loaded. Support channels: {spt_name}")
        
        w_k = meta_learner.get_params()
        meta_grad = torch.zeros_like(w_k)  # Initialize as tensor instead of float
        lam_grad = 0.0
        
        # Track losses for each task
        task_losses = []
    
        for i in range(args.N_way):
            print(f"  Processing task {i+1}/{args.N_way}...")
            fast_learner.set_params(w_k.clone())
            task = {
                'x_train': torch.tensor(x_qry_scld[i], device=device),
                'y_train': torch.tensor(y_qry_scld[i], device=device),
                'x_val': torch.tensor(x_spt_scld[i], device=device),
                'y_val': torch.tensor(y_spt_scld[i], device=device)
            }
            
            # Learn task and get training losses
            print(f"    Learning task with {args.n_steps} inner steps...")
            tl = fast_learner.learn_task(task, num_steps=args.n_steps)
            vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            print(f"    Initial validation loss: {vl_before:.6f}")
        
            print(f"    Computing regularization loss...")
            fast_learner.inner_opt.zero_grad()
            regu_loss = fast_learner.regularization_loss(w_k, lam)
            regu_loss.backward()
            fast_learner.inner_opt.step()
            
            vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            print(f"    Final validation loss: {vl_after:.6f}")
            print(f"    Loss improvement: {vl_before - vl_after:.6f}")
            
            # Store losses for this task (first and last steps only for iMAML)
            task_loss = {
                'first_step': vl_before,  # Initial validation loss
                'last_step': vl_after   # Final validation loss after inner loop
            }
            task_losses.append(task_loss)
            
            print(f"    Computing meta-gradient...")
            valid_loss = fast_learner.get_loss(task['x_val'], task['y_val'])
            valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
            flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
            
            if torch.isnan(flat_grad).any():
                print(f"    WARNING: NaN detected in flat_grad at task {i} in meta step {outstep}")
            
            if args.cg_steps <= 1:
                print(f"    Using direct gradient (CG steps <= 1)")
                task_outer_grad = flat_grad
            else:
                print(f"    Starting CG solve with {args.cg_steps} steps...")
                try:
                    task_matrix_evaluator = fast_learner.matrix_evaluator(task, lam, args.cg_damping)
                    # Add timeout for CG solve to prevent infinite loops
                    start_time = timer.time()
                    task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)
                    solve_time = timer.time() - start_time
                    print(f"    CG solve completed in {solve_time:.2f}s")
                except Exception as e:
                    print(f"    ERROR: CG solve failed with error: {e}")
                    print(f"    Falling back to direct gradient")
                    task_outer_grad = flat_grad
            
            if torch.isnan(task_outer_grad).any():
                print(f"    WARNING: NaN detected in task_outer_grad at task {i} in meta step {outstep}")
                
            print(f"    Accumulating meta-gradient...")
            meta_grad += (task_outer_grad / args.task_mb_size)
            losses[outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after]) / args.task_mb_size)
                  
            if args.lam_lr <= 0.0:
                task_lam_grad = 0.0
            else:
                print("    Warning: lambda learning is not tested for this version of code")
                train_loss = fast_learner.get_loss(task['x_train'], task['y_train'])
                train_grad = torch.autograd.grad(train_loss, fast_learner.model.parameters())
                train_grad = torch.cat([g.contiguous().view(-1) for g in train_grad])
                inner_prod = train_grad.dot(task_outer_grad)
                task_lam_grad = inner_prod / (lam**2 + 0.1)
            lam_grad += (task_lam_grad / args.task_mb_size)
            print(f"    Task {i+1} completed successfully")
        
        print(f"  All tasks completed. Updating meta-learner...")
        # Track losses for this meta step
        loss_tracker.track_losses(outstep, spt_name, task_losses)
        
        print(f"  Applying meta-update...")
        meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
        lam_delta = - args.lam_lr * lam_grad  
        lam = torch.clamp(lam + lam_delta, args.lam_min, 5000.0)
        param_norm = torch.norm(meta_learner.get_params())
        print(f"  Meta Step {outstep}: parameter norm = {param_norm.item():.6f}")
        
        # Clear GPU cache periodically to prevent memory issues
        if args.use_gpu and outstep % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  GPU cache cleared")
        
        if outstep % 100 == 0:
            print(f"Meta Step {outstep}: parameter norm = {param_norm.item()}")
            
            # Print tracking summary
            if outstep % 500 == 0:
                summary = loss_tracker.get_summary_stats()
                print(f"\n--- iMAML Tracking Summary at Meta Step {outstep} ---")
                for channel, stats in summary.items():
                    print(f"{channel}: {stats['total_appearances']} appearances, "
                          f"First Step: {stats['first_step_mean']:.4f}, "
                          f"Last Step: {stats['last_step_mean']:.4f}, "
                          f"Improvement: {stats['improvement_first_to_last']:.4f}")
        
        if (outstep % 50 == 0) or (outstep == args.meta_steps - 1):
            print(f"  Saving checkpoint at step {outstep}...")
            train_pre, test_pre, train_post, test_post = losses[outstep]
            print(f"  Meta Step {outstep}: Train pre = {train_pre:.4f}, Test pre = {test_pre:.4f}, " 
                  f"Train post = {train_post:.4f}, Test post = {test_post:.4f}")
            
            meta_model_chpoint = os.path.join(args.save_dir, f"meta_model_nway_{args.N_way}")
            os.makedirs(meta_model_chpoint, exist_ok=True)
            print(f"  Checkpoint directory: {meta_model_chpoint}")
            # Save checkpoint manually since utils import might have issues
            checkpoint = {
                'step': outstep, 
                'state_dict': meta_learner.model.state_dict(),
                'meta_learner_state': meta_learner.get_params()
            }
            torch.save(checkpoint, os.path.join(meta_model_chpoint, f"wireless_IMAML_{args.K_shot}_shot_checkpoint_step_{outstep}_LAM{args.lam}.pth.tar"))

            # Save tracking data
            loss_tracker.save_tracking_data(outstep)
            
            steps = np.arange(outstep + 1)
            train_post_curve = losses[:outstep+1, 2]
            test_post_curve = losses[:outstep+1, 3]

            plt.figure(figsize=(10, 6))
            plt.plot(steps, train_post_curve, label='Train post')
            plt.plot(steps, test_post_curve, label='Test post')
            plt.title(f'iMAML Training Loss Curve for {args.K_shot}-shot with Tracking')
            plt.xlabel('Meta Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(meta_model_chpoint, f'iMAML_{args.K_shot}shot_LAM{args.lam}_training_loss_curve_with_tracking.png'))
            plt.close()
            print(f"  Checkpoint and plots saved successfully")
        
        print(f"=== META STEP {outstep} COMPLETED ===\n")

    # Final save of all tracking data
    loss_tracker.save_tracking_data(args.meta_steps - 1)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TRACKING SUMMARY")
    print("="*60)
    final_summary = loss_tracker.get_summary_stats()
    for channel, stats in final_summary.items():
        print(f"\n{channel}:")
        print(f"  Total appearances: {stats['total_appearances']}")
        print(f"  First Step - Mean: {stats['first_step_mean']:.6f}, Std: {stats['first_step_std']:.6f}")
        print(f"  Last Step - Mean: {stats['last_step_mean']:.6f}, Std: {stats['last_step_std']:.6f}")
        print(f"  Improvement First→Last: {stats['improvement_first_to_last']:.6f}")

    print(f"\niMAML training complete! Tracking data saved to: {args.tracking_dir}")

if __name__ == '__main__':
    main()
