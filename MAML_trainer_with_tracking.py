#!/usr/bin/env python3
"""
MAML Trainer with Inner Loop Loss Tracking

This script modifies the original MAML trainer to track inner loop losses
for each channel across all training epochs, as requested by the advisor.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from datetime import datetime
from Data_Nshot import ChannelEstimationNShot
from metrics import Metric, mmse_channel_estimation, extract_snr
from utils import Utils
import argparse
from meta import Meta
import matplotlib.pyplot as plt
import pdb

class LossTracker:
    """Tracks inner loop losses for each channel during MAML training."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.tracking_data = []
        self.channel_stats = {}
        os.makedirs(save_dir, exist_ok=True)
        
    def track_losses(self, epoch, channel_names, losses_s, task_idx=0):
        """
        Track inner loop losses for a specific task.
        
        Args:
            epoch: Current training epoch
            channel_names: List of channel names for this task
            losses_s: List of losses for each inner loop step
            task_idx: Task index within the batch
        """
        for i, channel_name in enumerate(channel_names):
            if i < len(losses_s):
                # Extract losses for this channel
                step_losses = losses_s[i] if isinstance(losses_s[i], list) else [losses_s[i]]
                
                # Ensure we have 3 steps (pad with last value if needed)
                while len(step_losses) < 3:
                    step_losses.append(step_losses[-1] if step_losses else 0.0)
                
                # Store tracking data
                tracking_entry = {
                    'epoch': epoch,
                    'task_idx': task_idx,
                    'channel_name': channel_name,
                    'step_0_loss': step_losses[0] if len(step_losses) > 0 else 0.0,
                    'step_1_loss': step_losses[1] if len(step_losses) > 1 else step_losses[0],
                    'step_2_loss': step_losses[2] if len(step_losses) > 2 else step_losses[-1],
                    'timestamp': datetime.now().isoformat()
                }
                
                self.tracking_data.append(tracking_entry)
                
                # Update channel statistics
                if channel_name not in self.channel_stats:
                    self.channel_stats[channel_name] = {
                        'total_appearances': 0,
                        'step_0_losses': [],
                        'step_1_losses': [],
                        'step_2_losses': [],
                        'epochs': []
                    }
                
                self.channel_stats[channel_name]['total_appearances'] += 1
                self.channel_stats[channel_name]['step_0_losses'].append(step_losses[0])
                self.channel_stats[channel_name]['step_1_losses'].append(step_losses[1])
                self.channel_stats[channel_name]['step_2_losses'].append(step_losses[2])
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
        """Get summary statistics for all channels."""
        summary = {}
        for channel_name, stats in self.channel_stats.items():
            summary[channel_name] = {
                'total_appearances': stats['total_appearances'],
                'step_0_mean': np.mean(stats['step_0_losses']),
                'step_0_std': np.std(stats['step_0_losses']),
                'step_1_mean': np.mean(stats['step_1_losses']),
                'step_1_std': np.std(stats['step_1_losses']),
                'step_2_mean': np.mean(stats['step_2_losses']),
                'step_2_std': np.std(stats['step_2_losses']),
                'improvement_0_to_1': np.mean(stats['step_0_losses']) - np.mean(stats['step_1_losses']),
                'improvement_1_to_2': np.mean(stats['step_1_losses']) - np.mean(stats['step_2_losses']),
                'improvement_0_to_2': np.mean(stats['step_0_losses']) - np.mean(stats['step_2_losses'])
            }
        return summary


class MetaWithTracking(Meta):
    """Extended Meta class with inner loop loss tracking."""
    
    def __init__(self, args, config, loss_tracker=None):
        super().__init__(args, config)
        self.loss_tracker = loss_tracker
        self.channel_names = []
        
    def set_channel_names(self, channel_names):
        """Set channel names for current batch."""
        self.channel_names = channel_names
        
    def forward(self, x_qry, y_qry, x_spt, y_spt):
        """Modified forward pass with loss tracking."""
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_qry.size(1)

        losses_s = [0 for _ in range(self.update_step + 1)]
        channel_losses = []  # Store losses for each channel

        for i in range(self.n_way):
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            
            # Get channel name for this task
            channel_name = self.channel_names[i] if i < len(self.channel_names) else f"channel_{i}"
            
            # Track losses for this specific channel
            channel_step_losses = []
            
            # Step 0: Initial loss
            logits = self.net(x_qry_i, vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_qry_i)
            channel_step_losses.append(loss.item())
            
            # First adaptation step
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  

            with torch.no_grad():
                x_spt_i = x_spt[i].view(querysz, c_, h, w)
                y_spt_i = y_spt[i].view(querysz, c_, h, w)
                logits_s = self.net(x_spt_i, self.net.parameters(), bn_training=True)
                logits_s = logits_s.view(querysz, c_, h, w) 
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[0] += loss_s

            with torch.no_grad():
                logits_s = self.net(x_spt_i, fast_weights, bn_training=True)  
                logits_s = logits_s.view(querysz, c_, h, w) 
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[1] += loss_s
                channel_step_losses.append(loss_s.item())

            # Additional adaptation steps
            for k in range(1, self.update_step):
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                logits = logits.view(setsz, c_, h, w)  
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_s = self.net(x_qry_i, fast_weights, bn_training=True)
                logits_s = logits_s.view(querysz, c_, h, w)  
                loss_s = F.mse_loss(logits_s, y_qry_i)
                losses_s[k + 1] += loss_s
                channel_step_losses.append(loss_s.item())

            # Store losses for this channel
            channel_losses.append(channel_step_losses)

        loss_q = losses_s[-1] / batchsz

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return losses_s, channel_losses


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    metric = Metric()

    # Define the model configuration
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
    
    # Create loss tracker
    loss_tracker = LossTracker(args.tracking_dir)
    
    # Create MAML model with tracking
    maml = MetaWithTracking(args, config, loss_tracker)
    maml.to(device)

    # count parameters
    total_params = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(maml)
    print(f"Total trainable parameters: {total_params}")

    # data loader
    db_train = ChannelEstimationNShot(
        args.root,
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    
    all_losses = []
    train_losses = []
    
    print(f"Starting MAML training with loss tracking...")
    print(f"Tracking directory: {args.tracking_dir}")
    
    for step in range(args.epoch):
        # fetch one meta‐batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
        xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        
        # Set channel names for tracking
        maml.set_channel_names(spt_name)
        
        # to tensors + device
        x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
        y_qry_scld = torch.from_numpy(y_qry_scld).to(device)
        x_spt_scld = torch.from_numpy(x_spt_scld).to(device)
        y_spt_scld = torch.from_numpy(y_spt_scld).to(device)
        xs_fixed_scld = torch.from_numpy(xs_fixed_scld).to(device)
        ys_fixed_scld = torch.from_numpy(ys_fixed_scld).to(device)

        x_qry = torch.from_numpy(x_qry).to(device)
        y_qry = torch.from_numpy(y_qry).to(device)
        x_spt = torch.from_numpy(x_spt).to(device)
        y_spt = torch.from_numpy(y_spt).to(device)
        xs_fixed = torch.from_numpy(xs_fixed).to(device)
        ys_fixed = torch.from_numpy(ys_fixed).to(device)
        
        # forward / meta‐update with tracking
        losses, channel_losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld)
        current_loss = losses[-1].item()
        all_losses.append(current_loss)
        
        # Track losses for each channel
        loss_tracker.track_losses(step, spt_name, channel_losses)
        
        if step % 100 == 0 or step == args.epoch - 1:
            train_losses.append(current_loss)
            print(f'step: {step}, training loss: {current_loss}')
            
            # Print tracking summary
            if step % 500 == 0:
                summary = loss_tracker.get_summary_stats()
                print(f"\n--- Tracking Summary at Epoch {step} ---")
                for channel, stats in summary.items():
                    print(f"{channel}: {stats['total_appearances']} appearances, "
                          f"Step 0: {stats['step_0_mean']:.4f}, "
                          f"Step 1: {stats['step_1_mean']:.4f}, "
                          f"Step 2: {stats['step_2_mean']:.4f}")
            
        if step % 1000 == 0 or step == args.epoch - 1:
            # Save model checkpoint
            ckpt_dir = os.path.join(
                args.save_init,
                f"meta_model_nway_{args.n_way}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{step}"
                + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            Utils.save_checkpoint(
                {'step': step, 'state_dict': maml.state_dict()},
                ckpt_path
            )
            
            # Save tracking data
            loss_tracker.save_tracking_data(step)

    # Final save of all tracking data
    loss_tracker.save_tracking_data(args.epoch - 1)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TRACKING SUMMARY")
    print("="*60)
    final_summary = loss_tracker.get_summary_stats()
    for channel, stats in final_summary.items():
        print(f"\n{channel}:")
        print(f"  Total appearances: {stats['total_appearances']}")
        print(f"  Step 0 - Mean: {stats['step_0_mean']:.6f}, Std: {stats['step_0_std']:.6f}")
        print(f"  Step 1 - Mean: {stats['step_1_mean']:.6f}, Std: {stats['step_1_std']:.6f}")
        print(f"  Step 2 - Mean: {stats['step_2_mean']:.6f}, Std: {stats['step_2_std']:.6f}")
        print(f"  Improvement 0→1: {stats['improvement_0_to_1']:.6f}")
        print(f"  Improvement 1→2: {stats['improvement_1_to_2']:.6f}")
        print(f"  Improvement 0→2: {stats['improvement_0_to_2']:.6f}")

    # plot training curve for all epochs
    plt.figure(figsize=(10, 6))
    steps_recorded = list(range(args.epoch))
    plt.plot(steps_recorded, all_losses, linestyle='-')
    plt.title(
        f'Training Loss ({args.k_qry}-shot MAML with Tracking)\n'
        f'Meta LR={args.meta_lr}, Task LR={args.update_lr}',
        fontsize=16
    )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    out_fig = os.path.join(
        args.save_init,
        f'{args.k_qry}shot_training_loss_curve_MetaLr{args.meta_lr}_TaskLr{args.update_lr}_with_tracking.png'
    )
    plt.savefig(out_fig)
    
    print(f"\nTraining complete! Tracking data saved to: {args.tracking_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     type=str,   default="Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    parser.add_argument('--device',   type=str,   default='cuda:0')
    parser.add_argument('--save_init',type=str,   default="SISO_UMi_init/std_scaler_interpolated_noleak")
    parser.add_argument('--tracking_dir', type=str, default="inner_loop_tracking_data_umi")
    parser.add_argument('--epoch',    type=int,   default=1000)
    parser.add_argument('--n_way',    type=int,   default=5)
    parser.add_argument('--k_spt',    type=int,   default=5)
    parser.add_argument('--k_qry',    type=int,   default=5)
    parser.add_argument('--batchsz',  type=int,   default=8)
    parser.add_argument('--meta_lr',  type=float, default=1e-4)
    parser.add_argument('--update_lr',type=float, default=1e-3)
    parser.add_argument('--update_step', type=int, default=3)
    args = parser.parse_args()
    main(args)


# python box_plot_analysis.py --tracking_dir inner_loop_tracking_data_tdl --output_dir inner_analysis_results_tdl

# # Step 3: Combined steps box plots
# python combined_steps_box_plots.py

# # Step 4: Learning progression box plots
# python learning_progression_box_plots.py