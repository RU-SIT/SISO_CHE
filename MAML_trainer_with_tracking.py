#!/usr/bin/env python3
"""
MAML Trainer with Inner Loop Loss Tracking

This script modifies the original MAML trainer to track inner loop losses
for each channel across all training epochs, as requested by the advisor.

NEW: Supports initialization with pre-trained ChannelNet weights (converted from Keras/TensorFlow)
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
                    'step_0_loss': float(step_losses[0]) if len(step_losses) > 0 else 0.0,
                    'step_1_loss': float(step_losses[1]) if len(step_losses) > 1 else float(step_losses[0]),
                    'step_2_loss': float(step_losses[2]) if len(step_losses) > 2 else float(step_losses[-1]),
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
                self.channel_stats[channel_name]['step_0_losses'].append(float(step_losses[0]))
                self.channel_stats[channel_name]['step_1_losses'].append(float(step_losses[1]))
                self.channel_stats[channel_name]['step_2_losses'].append(float(step_losses[2]))
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
        """
        Set channel names for current batch.
        
        channel_names is a flat list with k_spt entries per task.
        We need to group them by task and take one name per task.
        For example: ['TDL-A.mat', 'TDL-A.mat', ...(k_spt times), 'SNR_0dB_GROUP', 'SNR_0dB_GROUP', ...(k_spt times)]
        Should become: ['TDL-A.mat', 'SNR_0dB_GROUP', ...]
        """
        processed_names = []
        
        if isinstance(channel_names, (list, tuple, np.ndarray)):
            # Group by k_spt to get one name per task
            k_spt = self.k_spt
            num_tasks = len(channel_names) // k_spt if k_spt > 0 else len(channel_names)
            
            for i in range(num_tasks):
                # Take the first name from each k_spt group
                idx = i * k_spt
                if idx < len(channel_names):
                    entry = channel_names[idx]
                    if isinstance(entry, (list, tuple, np.ndarray)):
                        if len(entry) > 0:
                            processed_names.append(str(entry[0]))
                        else:
                            processed_names.append("unknown_channel")
                    else:
                        processed_names.append(str(entry))
                else:
                    processed_names.append("unknown_channel")
        else:
            processed_names.append(str(channel_names))
        
        self.channel_names = processed_names
        
    def forward(self, x_qry, y_qry, x_spt, y_spt):
        """Modified forward pass with loss tracking aligned to updated Meta class."""
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_spt.size(1)

        losses_s = [0 for _ in range(self.update_step + 1)]
        channel_losses = []

        for i in range(batchsz):
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            x_spt_i = x_spt[i].view(querysz, c_, h, w)
            y_spt_i = y_spt[i].view(querysz, c_, h, w)

            channel_step_losses = []

            logits = self.net(x_qry_i, vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_qry_i)

            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                    zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_s = self.net(x_spt_i, self.net.parameters(), bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[0] += loss_s
                channel_step_losses.append(loss_s.item())

            with torch.no_grad():
                logits_s = self.net(x_spt_i, fast_weights, bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[1] += loss_s
                channel_step_losses.append(loss_s.item())

            for k in range(1, self.update_step):
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                        zip(grad, fast_weights)))

                logits_s = self.net(x_spt_i, fast_weights, bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[k + 1] += loss_s
                channel_step_losses.append(loss_s.item())

            channel_losses.append(channel_step_losses)

        loss_q = losses_s[-1] / batchsz

        self.meta_optim.zero_grad()
        loss_q.backward()

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.meta_optim.step()

        return losses_s, channel_losses


def load_pretrained_weights(maml_model, checkpoint_path, strict=False):
    """
    Load pre-trained weights from ChannelNet (converted from Keras) into MAML model.
    
    This function enables transfer learning by initializing MAML with weights
    learned from a different training procedure (e.g., supervised ChannelNet training).
    
    Args:
        maml_model: Your MAML/Meta model instance
        checkpoint_path: Path to the converted PyTorch checkpoint (.pth.tar file)
        strict: If False, allows partial loading when architectures don't match exactly
        
    Returns:
        maml_model with loaded pre-trained weights
        
    Technical Details:
        - strict=False is recommended for transfer learning between different architectures
        - The function handles missing or extra keys gracefully
        - Useful when your MAML model has similar (but not identical) structure to ChannelNet
    """
    print(f"\n{'='*70}")
    print(f"Loading pre-trained weights from: {checkpoint_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Warning: Checkpoint file not found: {checkpoint_path}")
        print("Proceeding with random initialization...")
        return maml_model
    
    # Load the checkpoint file
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dictionary (handles different checkpoint formats)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            if 'source' in checkpoint:
                print(f"Source: {checkpoint['source']}")
            if 'keras_source' in checkpoint:
                print(f"Original Keras model: {checkpoint['keras_source']}")
        else:
            state_dict = checkpoint
        
        print(f"Checkpoint contains {len(state_dict)} parameters")
        
        # Get current model state for comparison
        model_state = maml_model.state_dict()
        print(f"MAML model has {len(model_state)} parameters")
        
        # Load weights with partial matching allowed
        missing_keys, unexpected_keys = maml_model.load_state_dict(
            state_dict, 
            strict=strict
        )
        
        # Report loading status
        loaded_count = len(state_dict) - len(unexpected_keys)
        print(f"\n✓ Successfully loaded {loaded_count}/{len(model_state)} parameters")
        
        if missing_keys:
            print(f"\n⚠ {len(missing_keys)} parameters in MAML model not found in checkpoint:")
            print("  (These will use random initialization)")
            for key in missing_keys[:3]:
                print(f"  - {key}")
            if len(missing_keys) > 3:
                print(f"  ... and {len(missing_keys) - 3} more")
        
        if unexpected_keys:
            print(f"\n⚠ {len(unexpected_keys)} parameters in checkpoint not used:")
            for key in unexpected_keys[:3]:
                print(f"  - {key}")
            if len(unexpected_keys) > 3:
                print(f"  ... and {len(unexpected_keys) - 3} more")
        
        if not missing_keys and not unexpected_keys:
            print("\n✓ Perfect match! All weights loaded successfully!")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n⚠ Error loading checkpoint: {e}")
        print("Proceeding with random initialization...\n")
    
    return maml_model


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    metric = Metric()

    # Define the model configuration
    # config = [
    #     ('conv2d', [64, 2, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [64]),
    #     ('conv2d', [256, 64, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [256]),
    #     ('conv2d', [512, 256, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [512]),
    #     ('conv2d', [256, 512, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [256]),
    #     ('conv2d', [32, 256, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [32]),
    #     ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
    #     ('tanh', [True]),
    #     ('avg_pool2d', [3, 1, 1]),
    #     ('bn', [args.batchsz]),
    #     ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    # ]
    config = [
        # SRCNN layers commented out - using DNCNN only
        # # SRCNN Layer 1: 9x9 conv, in=2, out=64
        # ('conv2d', [64, 2, 9, 9, 1, 4]),  # padding=4 to preserve spatial dims
        # ('tanh', [True]),
        # 
        # # SRCNN Layer 2: 1x1 conv, in=64, out=32
        # ('conv2d', [32, 64, 1, 1, 1, 0]),  # padding=0 for 1x1 kernel
        # ('tanh', [True]),
        # 
        # # SRCNN Layer 3: 5x5 conv, in=32, out=2
        # ('conv2d', [2, 32, 5, 5, 1, 2]),  # padding=2 to preserve spatial dims
        # ('tanh', [True]),
        
        # DNCNN Layer 1: 3x3 conv, in=2, out=64
        ('conv2d', [64, 2, 3, 3, 1, 1]),  # padding=1 to preserve spatial dims
        ('tanh', [True]),
        ('bn', [64]),
        
        # DNCNN Layers 2-19 (conv5-22): 18 layers of 3x3 conv, in=64, out=64 with BN
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        # DNCNN Layer 20 (conv23): 3x3 conv, in=64, out=2 (final output)
        ('conv2d', [2, 64, 3, 3, 1, 1])
    ]
   
   
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create loss tracker
    loss_tracker = LossTracker(args.tracking_dir)
    
    # Create MAML model with tracking
    maml = MetaWithTracking(args, config, loss_tracker)
    
    # Load pre-trained weights if provided (BEFORE moving to device)
    if args.pretrained_weights is not None:
        print("\n🔄 Initializing with pre-trained ChannelNet weights...")
        maml = load_pretrained_weights(maml, args.pretrained_weights, strict=False)
        print("✓ Model initialized with pre-trained weights!\n")
    else:
        print("\n⚠ No pre-trained weights provided. Using random initialization.\n")
    
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
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_step = 0
    early_stopped = False
    
    ckpt_dir = os.path.join(
        args.save_init,
        f"meta_model_nway_{args.n_way}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Starting MAML training with loss tracking...")
    print(f"Tracking directory: {args.tracking_dir}")
    
    for step in range(args.epoch):
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
         xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        
        maml.set_channel_names(spt_name)
        
        x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
        y_qry_scld = torch.from_numpy(y_qry_scld).to(device)
        x_spt_scld = torch.from_numpy(x_spt_scld).to(device)
        y_spt_scld = torch.from_numpy(y_spt_scld).to(device)
        xs_fixed_scld = torch.from_numpy(xs_fixed_scld).to(device)
        ys_fixed_scld = torch.from_numpy(ys_fixed_scld).to(device)

        losses, channel_losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld)
        current_loss = losses[-1].item()
        all_losses.append(current_loss)
        
        maml.meta_scheduler.step(current_loss)
        
        loss_tracker.track_losses(step, maml.channel_names, channel_losses)
        
        if current_loss < best_loss - args.early_stop_min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(maml.state_dict())
            best_step = step
            
            if args.early_stop_save_best:
                best_ckpt_path = os.path.join(
                    ckpt_dir,
                    f"MAML_{args.k_qry}_shot_{args.k_spt}_query_BEST_checkpoint_step_{step}"
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
        print(f'step: {step}, training loss: {current_loss:.6f}, best loss: {best_loss:.6f}')
        
        # Debug: Show channel names being tracked (first 10 steps only)
        if step < 10:
            print(f"  → Tracking channels: {maml.channel_names}")
        
        if step % 500 == 0:
            summary = loss_tracker.get_summary_stats()
            print(f"\n--- Tracking Summary at Epoch {step} ---")
            for channel, stats in summary.items():
                print(f"{channel}: {stats['total_appearances']} appearances, "
                      f"Step 0: {stats['step_0_mean']:.4f}, "
                      f"Step 1: {stats['step_1_mean']:.4f}, "
                      f"Step 2: {stats['step_2_mean']:.4f}")
        
        if step % 20 == 0 or step == args.epoch - 1:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{step}"
                + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            Utils.save_checkpoint(
                {'step': step, 'state_dict': maml.state_dict()},
                ckpt_path
            )
            loss_tracker.save_tracking_data(step)
        
        if args.enable_early_stopping and epochs_no_improve >= args.early_stop_patience:
            print(f'Early stopping at step {step + 1}. Best loss: {best_loss:.6f} at step {best_step}')
            if args.early_stop_restore_best and best_model_state is not None:
                print('Restoring best model weights...')
                maml.load_state_dict(best_model_state)
            early_stopped = True
            loss_tracker.save_tracking_data(step)
            break

    final_step = len(all_losses) - 1 if all_losses else 0
    loss_tracker.save_tracking_data(final_step)
    
    print('\n' + '='*80)
    print('Training Summary:')
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
    
    print("="*60)
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
    if all_losses:
        plt.figure(figsize=(10, 6))
        steps_recorded = list(range(len(all_losses)))
        plt.plot(steps_recorded, all_losses, linestyle='-')
        plt.title(
            f'Training Loss ({args.k_qry}-shot MAML with Tracking)\n'
            f'Meta LR={args.meta_lr}, Task LR={args.update_lr}',
            fontsize=16
        )
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        out_fig = os.path.join(
            args.save_init,
            f'{args.k_qry}shot_training_loss_curve_MetaLr{args.meta_lr}_TaskLr{args.update_lr}_with_tracking.png'
        )
        plt.savefig(out_fig)
        plt.close()
        print(f'Training curve saved to: {out_fig}')
    
    print(f"\nTraining complete! Tracking data saved to: {args.tracking_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     type=str,   default="Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    parser.add_argument('--device',   type=str,   default='cuda:0')
    parser.add_argument('--save_init',type=str,   default="SISO_UMi_init/std_scaler_interpolated_noleak")
    parser.add_argument('--tracking_dir', type=str, default="inner_loop_tracking_data_umi")
    parser.add_argument('--epoch',    type=int,   default=5000)
    parser.add_argument('--n_way',    type=int,   default=5)
    parser.add_argument('--k_spt',    type=int,   default=5)
    parser.add_argument('--k_qry',    type=int,   default=5)
    parser.add_argument('--batchsz',  type=int,   default=8)
    parser.add_argument('--meta_lr',  type=float, default=5e-4)
    parser.add_argument('--update_lr',type=float, default=1e-4)
    parser.add_argument('--update_step', type=int, default=3)
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which learning rate will be reduced')
    parser.add_argument('--scheduler_patience', type=int, default=8,
                        help='Number of steps with no improvement after which LR will be reduced')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                        help='Lower bound on learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.75,
                        help='Maximum gradient norm for clipping (set <=0 to disable)')
    parser.add_argument('--enable_early_stopping', action='store_true', default=False,
                        help='Enable early stopping during training')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Number of steps with no improvement after which training will be stopped')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-3,
                        help='Minimum change in loss to qualify as an improvement')
    parser.add_argument('--early_stop_restore_best', action='store_true', default=True,
                        help='Restore best model weights when early stopping triggers')
    parser.add_argument('--early_stop_save_best', action='store_true', default=True,
                        help='Save best model checkpoint when improvement occurs')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pre-trained weights (converted from Keras ChannelNet). '
                             'If provided, initializes MAML with these weights instead of random initialization.')
    args = parser.parse_args()
    main(args)


# python box_plot_analysis.py --tracking_dir inner_loop_tracking_data_tdl --output_dir inner_analysis_results_tdl

# # Step 3: Combined steps box plots
# python combined_steps_box_plots.py

# # Step 4: Learning progression box plots
# python learning_progression_box_plots.py