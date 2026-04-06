#!/usr/bin/env python3
"""
MAML + PCGrad Trainer for Channel Estimation

This script combines MAML meta-learning with PCGrad (Project Conflicting Gradients)
for multi-task channel estimation. Each channel type or SNR level is treated as a
separate task, and PCGrad resolves conflicting gradients between tasks.
"""

import copy
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
import random


class PCGrad():
    """
    PCGrad optimizer wrapper for resolving conflicting gradients in multi-task learning.
    Adapted for regression tasks with MSE loss.
    """
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''clear the gradient of the parameters'''
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''update the parameters with the gradient'''
        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters with conflict resolution

        input:
        - objectives: a list of loss tensors (one per task)
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        """Project conflicting gradients to resolve task conflicts"""
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:  # Conflicting gradients
                    # Project g_i onto the normal plane of g_j
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''set the modified gradients to the network'''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class LossTracker:
    """Tracks per-task losses for analysis"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.tracking_data = []
        self.task_stats = {}
        os.makedirs(save_dir, exist_ok=True)
        
    def track_losses(self, epoch, task_names, task_losses, meta_loss):
        """
        Track per-task losses after gradient surgery.
        
        Args:
            epoch: Current training epoch
            task_names: List of task/channel names
            task_losses: List of loss values for each task
            meta_loss: Final meta loss after PCGrad
        """
        for task_name, task_loss in zip(task_names, task_losses):
            tracking_entry = {
                'epoch': epoch,
                'task_name': task_name,
                'task_loss': float(task_loss),
                'meta_loss': float(meta_loss),
                'timestamp': datetime.now().isoformat()
            }
            
            self.tracking_data.append(tracking_entry)
            
            # Update task statistics
            if task_name not in self.task_stats:
                self.task_stats[task_name] = {
                    'total_appearances': 0,
                    'losses': [],
                    'epochs': []
                }
            
            self.task_stats[task_name]['total_appearances'] += 1
            self.task_stats[task_name]['losses'].append(float(task_loss))
            self.task_stats[task_name]['epochs'].append(epoch)
    
    def save_tracking_data(self, epoch):
        """Save tracking data to files."""
        # Save detailed tracking data
        tracking_file = os.path.join(self.save_dir, f'pcgrad_tracking_epoch_{epoch}.json')
        with open(tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.tracking_data:
            df = pd.DataFrame(self.tracking_data)
            csv_file = os.path.join(self.save_dir, f'pcgrad_tracking_epoch_{epoch}.csv')
            df.to_csv(csv_file, index=False)
        
        # Save task statistics
        stats_file = os.path.join(self.save_dir, f'task_stats_epoch_{epoch}.json')
        with open(stats_file, 'w') as f:
            json.dump(self.task_stats, f, indent=2)
    
    def get_summary_stats(self):
        """Get summary statistics for all tasks."""
        summary = {}
        for task_name, stats in self.task_stats.items():
            if stats['losses']:
                summary[task_name] = {
                    'total_appearances': stats['total_appearances'],
                    'mean_loss': np.mean(stats['losses']),
                    'std_loss': np.std(stats['losses']),
                    'min_loss': np.min(stats['losses']),
                    'max_loss': np.max(stats['losses']),
                }
        return summary


class MetaWithPCGrad(Meta):
    """Extended Meta class with PCGrad for multi-task learning."""
    
    def __init__(self, args, config, loss_tracker=None):
        super().__init__(args, config)
        
        # Wrap optimizer with PCGrad
        self.meta_optim = PCGrad(self.meta_optim, reduction='mean')
        self.loss_tracker = loss_tracker
        self.task_names = []
        
    def set_task_names(self, channel_names):
        """
        Set task names from channel names.
        Each task corresponds to one channel/SNR group.
        
        channel_names is a flat list with k_spt entries per task.
        We group them and take one name per task.
        """
        processed_names = []
        
        if isinstance(channel_names, (list, tuple, np.ndarray)):
            k_spt = self.k_spt
            num_tasks = len(channel_names) // k_spt if k_spt > 0 else len(channel_names)
            
            for i in range(num_tasks):
                idx = i * k_spt
                if idx < len(channel_names):
                    entry = channel_names[idx]
                    if isinstance(entry, (list, tuple, np.ndarray)):
                        if len(entry) > 0:
                            processed_names.append(str(entry[0]))
                        else:
                            processed_names.append("unknown_task")
                    else:
                        processed_names.append(str(entry))
                else:
                    processed_names.append("unknown_task")
        else:
            processed_names.append(str(channel_names))
        
        self.task_names = processed_names
        
    def forward(self, x_qry, y_qry, x_spt, y_spt):
        """
        Modified forward pass with PCGrad for multi-task learning.
        
        Each task in the batch is treated as a separate task for gradient surgery.
        """
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_spt.size(1)

        # Store per-task losses for PCGrad
        task_losses = []
        task_predictions = []

        for i in range(batchsz):
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            x_spt_i = x_spt[i].view(querysz, c_, h, w)
            y_spt_i = y_spt[i].view(querysz, c_, h, w)

            # Initial prediction on query set
            logits = self.net(x_qry_i, vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_qry_i)

            # Compute gradients for inner loop update
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                    zip(grad, self.net.parameters())))

            # Perform additional inner loop updates
            for k in range(1, self.update_step):
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                        zip(grad, fast_weights)))

            # Compute loss on support set after adaptation (meta-loss for this task)
            logits_s = self.net(x_spt_i, fast_weights, bn_training=False)
            loss_s = F.mse_loss(logits_s, y_spt_i)
            
            task_losses.append(loss_s)

        # Apply PCGrad: resolve conflicting gradients between tasks
        self.meta_optim.zero_grad()
        
        # PCGrad performs gradient surgery and sets the resolved gradients
        self.meta_optim.pc_backward(task_losses)
        
        # Apply gradient clipping if specified
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        # Update meta-parameters with resolved gradients
        self.meta_optim.step()

        # Calculate average loss for logging
        avg_loss = torch.stack(task_losses).mean()
        
        # Track losses for analysis
        if self.loss_tracker is not None:
            task_loss_values = [loss.item() for loss in task_losses]
            self.loss_tracker.track_losses(
                epoch=0,  # Will be set by caller
                task_names=self.task_names,
                task_losses=task_loss_values,
                meta_loss=avg_loss.item()
            )

        return avg_loss, task_losses


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)

    print(args)
    metric = Metric()

    # Define the model configuration (same as original MAML)
    config = [
        ('conv2d', [32, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [32]),
        ('conv2d', [128, 32, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [128]),
        ('conv2d', [256, 128, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [128, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [128]),
        ('conv2d', [32, 128, 3, 3, 1, 1]),
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
    
    # Create MAML model with PCGrad
    maml = MetaWithPCGrad(args, config, loss_tracker)
    maml.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(maml)
    print(f"Total trainable parameters: {total_params}")
    print("\n" + "="*80)
    print("USING PCGRAD FOR MULTI-TASK CHANNEL ESTIMATION")
    print("="*80)
    print("Each channel type/SNR group is treated as a separate task.")
    print("PCGrad resolves conflicting gradients between tasks.")
    print("="*80 + "\n")

    # Data loader
    db_train = ChannelEstimationNShot(
        args.root,
        batchsz=args.batchsz,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry
    )
    
    all_losses = []
    train_losses = []
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_step = 0
    early_stopped = False
    
    ckpt_dir = os.path.join(
        args.save_init,
        f"meta_pcgrad_model_nway_{args.n_way}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Starting MAML+PCGrad training...")
    print(f"Tracking directory: {args.tracking_dir}")
    
    for step in range(args.epoch):
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
         xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        
        # Set task names for tracking
        maml.set_task_names(spt_name)
        
        # Move data to device
        x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
        y_qry_scld = torch.from_numpy(y_qry_scld).to(device)
        x_spt_scld = torch.from_numpy(x_spt_scld).to(device)
        y_spt_scld = torch.from_numpy(y_spt_scld).to(device)
        xs_fixed_scld = torch.from_numpy(xs_fixed_scld).to(device)
        ys_fixed_scld = torch.from_numpy(ys_fixed_scld).to(device)

        # Forward pass with PCGrad
        avg_loss, task_losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld)
        
        # Update loss tracker with current epoch
        if loss_tracker is not None:
            loss_tracker.tracking_data[-len(task_losses):] = [
                {**entry, 'epoch': step} 
                for entry in loss_tracker.tracking_data[-len(task_losses):]
            ]
        
        current_loss = avg_loss.item()
        all_losses.append(current_loss)
        
        # Step learning rate scheduler
        maml.meta_scheduler.step(current_loss)
        
        # Early stopping logic
        if current_loss < best_loss - args.early_stop_min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(maml.state_dict())
            best_step = step
            
            if args.early_stop_save_best:
                best_ckpt_path = os.path.join(
                    ckpt_dir,
                    f"MAML_PCGrad_{args.k_qry}_shot_{args.k_spt}_query_BEST_step_{step}"
                    + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
                )
                Utils.save_checkpoint(
                    {'step': step, 'state_dict': best_model_state, 'loss': best_loss},
                    best_ckpt_path
                )
        else:
            epochs_no_improve += 1
            if args.enable_early_stopping and step % 100 == 0:
                print(f'No improvement for {epochs_no_improve} step(s). Best loss: {best_loss:.6f}')
        
        train_losses.append(current_loss)
        
        # Logging
        if step % 10 == 0:
            task_loss_str = ", ".join([f"{loss.item():.4f}" for loss in task_losses[:3]])
            print(f'Step {step}: avg_loss={current_loss:.6f}, best={best_loss:.6f}')
            print(f'  Tasks: {maml.task_names[:3]} → Losses: [{task_loss_str}...]')
        
        # Periodic summary
        if step % 500 == 0:
            summary = loss_tracker.get_summary_stats()
            print(f"\n--- PCGrad Task Summary at Step {step} ---")
            for task_name, stats in sorted(summary.items())[:5]:  # Show top 5
                print(f"{task_name}: {stats['total_appearances']} appearances, "
                      f"Mean: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")
            print()
        
        # Save checkpoints
        if step % 100 == 0 or step == args.epoch - 1:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"MAML_PCGrad_{args.k_qry}_shot_{args.k_spt}_query_step_{step}"
                + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            Utils.save_checkpoint(
                {'step': step, 'state_dict': maml.state_dict()},
                ckpt_path
            )
            loss_tracker.save_tracking_data(step)
        
        # Early stopping check
        if args.enable_early_stopping and epochs_no_improve >= args.early_stop_patience:
            print(f'Early stopping at step {step + 1}. Best loss: {best_loss:.6f} at step {best_step}')
            if args.early_stop_restore_best and best_model_state is not None:
                print('Restoring best model weights...')
                maml.load_state_dict(best_model_state)
            early_stopped = True
            loss_tracker.save_tracking_data(step)
            break

    # Save final results
    final_step = len(all_losses) - 1 if all_losses else 0
    loss_tracker.save_tracking_data(final_step)
    
    print('\n' + '='*80)
    print('MAML+PCGrad Training Summary:')
    if all_losses:
        print(f'  Total steps: {len(all_losses)}')
        print(f'  Final loss: {all_losses[-1]:.6f}')
    else:
        print('  Total steps: 0')
        print('  Final loss: N/A')
    print(f'  Best loss: {best_loss:.6f}' + (f' (at step {best_step})' if best_loss < float("inf") else ''))
    if args.enable_early_stopping:
        print(f'  Early stopping: {"Yes (triggered)" if early_stopped else "No (completed all steps)"}')
    else:
        print('  Early stopping: Disabled')
    print('='*80 + '\n')
    
    # Final task summary
    print("="*60)
    print("FINAL TASK-WISE PERFORMANCE (PCGrad)")
    print("="*60)
    final_summary = loss_tracker.get_summary_stats()
    for task_name, stats in sorted(final_summary.items()):
        print(f"\n{task_name}:")
        print(f"  Total appearances: {stats['total_appearances']}")
        print(f"  Mean loss: {stats['mean_loss']:.6f} ± {stats['std_loss']:.6f}")
        print(f"  Min loss: {stats['min_loss']:.6f}")
        print(f"  Max loss: {stats['max_loss']:.6f}")

    # Plot training curve
    if all_losses:
        plt.figure(figsize=(12, 6))
        
        # Main loss curve
        plt.subplot(1, 2, 1)
        steps_recorded = list(range(len(all_losses)))
        plt.plot(steps_recorded, all_losses, linestyle='-', alpha=0.7)
        plt.title(
            f'MAML+PCGrad Training Loss\n({args.k_qry}-shot)\n'
            f'Meta LR={args.meta_lr}, Task LR={args.update_lr}',
            fontsize=14
        )
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Per-task statistics
        plt.subplot(1, 2, 2)
        task_names_plot = list(final_summary.keys())[:10]  # Top 10 tasks
        mean_losses = [final_summary[t]['mean_loss'] for t in task_names_plot]
        std_losses = [final_summary[t]['std_loss'] for t in task_names_plot]
        
        plt.barh(range(len(task_names_plot)), mean_losses, xerr=std_losses, alpha=0.7)
        plt.yticks(range(len(task_names_plot)), task_names_plot, fontsize=8)
        plt.xlabel('Mean Loss', fontsize=12)
        plt.title('Per-Task Performance', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        out_fig = os.path.join(
            args.save_init,
            f'{args.k_qry}shot_pcgrad_training_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png'
        )
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f'\nTraining curves saved to: {out_fig}')
    
    print(f"\nTraining complete! All tracking data saved to: {args.tracking_dir}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML+PCGrad for Channel Estimation')
    
    # Data arguments
    parser.add_argument('--root', type=str, 
                        default="Sionna_datasets/ps2_p612/speed5/SISO-TDL/interpolated_noleak",
                        help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, or cpu)')
    parser.add_argument('--save_init', type=str, 
                        default="SISO_TDL_init/pcgrad_std_scaler_interpolated_noleak",
                        help='Directory to save checkpoints')
    parser.add_argument('--tracking_dir', type=str, 
                        default="pcgrad_tracking_data_tdl",
                        help='Directory to save loss tracking data')
    
    # Training arguments
    parser.add_argument('--epoch', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of tasks per batch')
    parser.add_argument('--k_spt', type=int, default=5,
                        help='Number of support samples (k-shot)')
    parser.add_argument('--k_qry', type=int, default=5,
                        help='Number of query samples')
    parser.add_argument('--batchsz', type=int, default=8,
                        help='Meta batch size (number of tasks)')
    
    # Optimizer arguments
    parser.add_argument('--meta_lr', type=float, default=5e-4,
                        help='Meta learning rate')
    parser.add_argument('--update_lr', type=float, default=1e-4,
                        help='Inner loop learning rate')
    parser.add_argument('--update_step', type=int, default=3,
                        help='Number of inner loop updates')
    
    # Scheduler arguments
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which learning rate will be reduced')
    parser.add_argument('--scheduler_patience', type=int, default=8,
                        help='Number of steps with no improvement after which LR will be reduced')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                        help='Lower bound on learning rate')
    
    # Gradient clipping
    parser.add_argument('--max_grad_norm', type=float, default=0.75,
                        help='Maximum gradient norm for clipping (set <=0 to disable)')
    
    # Early stopping arguments
    parser.add_argument('--enable_early_stopping', action='store_true', default=False,
                        help='Enable early stopping during training')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='Number of steps with no improvement after which training will be stopped')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-3,
                        help='Minimum change in loss to qualify as an improvement')
    parser.add_argument('--early_stop_restore_best', action='store_true', default=True,
                        help='Restore best model weights when early stopping triggers')
    parser.add_argument('--early_stop_save_best', action='store_true', default=True,
                        help='Save best model checkpoint when improvement occurs')
    
    args = parser.parse_args()
    main(args)

