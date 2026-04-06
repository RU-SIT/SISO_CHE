
# torch.cuda.empty_cache()
# print(torch.cuda.device_count())
import os
import torch
# torch.distributed.init_process_group(backend='gloo')
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import copy
from Data_Nshot import ChannelEstimationNShot
from metrics import Metric, mmse_channel_estimation, extract_snr
from utils import Utils
import argparse
from meta import Meta
import matplotlib.pyplot as plt
import pdb

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    metric = Metric()

    # Define the model configuration to match ChannelNet architecture (SRCNN + DNCNN merged)
    # SRCNN part: 3 layers (conv1: 9x9, conv2: 1x1, conv3: 5x5)
    # DNCNN part: 20 layers (conv4: 3x3, conv5-22: 18x 3x3 with BN, conv23: 3x3)
    # Total params: SRCNN (8,129) + DNCNN (668,225) ≈ 676,354 parameters
    config = [
        # SRCNN Layer 1: 9x9 conv, in=2, out=64
        ('conv2d', [64, 2, 9, 9, 1, 4]),  # padding=4 to preserve spatial dims
        ('tanh', [True]),
        
        # SRCNN Layer 2: 1x1 conv, in=64, out=32
        ('conv2d', [32, 64, 1, 1, 1, 0]),  # padding=0 for 1x1 kernel
        ('tanh', [True]),
        
        # SRCNN Layer 3: 5x5 conv, in=32, out=2
        ('conv2d', [2, 32, 5, 5, 1, 2]),  # padding=2 to preserve spatial dims
        ('tanh', [True]),
        
        # DNCNN Layer 1 (conv4): 3x3 conv, in=2, out=64
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
    # pdb.set_trace()
    # if multiple GPUs, wrap in DataParallel
    maml = Meta(args, config)
    # pdb.set_trace()
    # if torch.cuda.device_count() > 1:
    #     device_ids = list(range(torch.cuda.device_count()))
    #     print(f"[*] DataParallel on GPUs {device_ids}")
    #     maml = torch.nn.DataParallel(
    #         maml,
    #         device_ids=device_ids,
    #         output_device=device_ids[0]
    #     )
        
    # move model to GPU(s)
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
    
    # Early stopping setup
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_step = 0
    
    all_losses = []
    train_losses = []
    ckpt_dir = os.path.join(
        args.save_init,
        f"meta_model_nway_{args.n_way}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for step in range(args.epoch):
        # fetch one meta‐batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
        xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()
        # pdb.set_trace()
        #data is in the format of (batchsz, setsz, c_, h, w) , the reason we are using this format is because the model is designed to take input in this format(pytorch conv2d layer expects input in this format).
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
        # pdb.set_trace()
        
        # forward / meta‐update
        losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld)
        current_loss = losses[-1].item()
        all_losses.append(current_loss)
        
        # Update learning rate scheduler
        maml.meta_scheduler.step(current_loss)
        
        # Early stopping: check for improvement
        if current_loss < best_loss - args.early_stop_min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(maml.state_dict())
            best_step = step
            
            # Save best model checkpoint
            if args.early_stop_save_best:
                best_ckpt_path = os.path.join(
                    ckpt_dir,
                    f"MAML_{args.k_qry}_shot_{args.k_spt}_query_BEST_checkpoint_step_{step}"
                    + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}_loss{best_loss:.6f}.pth.tar"
                )
                Utils.save_checkpoint(
                    {'step': step, 'state_dict': best_model_state, 'loss': best_loss},
                    best_ckpt_path
                )
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and step % 100 == 0:
                print(f'No improvement for {epochs_no_improve} step(s). Best loss: {best_loss:.6f}')
        
        # Early stopping: break if patience exceeded
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f'Early stopping at step {step + 1}. Best loss: {best_loss:.6f} at step {best_step}')
            if args.early_stop_restore_best and best_model_state is not None:
                print('Restoring best model weights...')
                maml.load_state_dict(best_model_state)
            break
        
        if step % 100 == 0 or step == args.epoch - 1:
            train_losses.append(current_loss)
            print(f'step: {step}, training loss: {current_loss:.6f}, best loss: {best_loss:.6f}')
            
        if step % 1000 == 0 or step == args.epoch - 1:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{step}"
                + f"_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.pth.tar"
            )
            Utils.save_checkpoint(
                {'step': step, 'state_dict': maml.state_dict()},
                ckpt_path
            )

    # Print final summary
    print('\n' + '='*80)
    print('Training Summary:')
    print(f'  Total steps: {len(all_losses)}')
    print(f'  Final loss: {all_losses[-1]:.6f}')
    print(f'  Best loss: {best_loss:.6f} (at step {best_step})')
    early_stopped = args.early_stop_patience > 0 and len(all_losses) < args.epoch
    if early_stopped:
        print(f'  Early stopping: Yes (stopped at step {len(all_losses)} after {epochs_no_improve} steps without improvement)')
    else:
        print(f'  Early stopping: No (completed all {args.epoch} steps)')
    print('='*80 + '\n')
    
    # plot training curve for all epochs
    plt.figure(figsize=(10, 6))
    # Create an array of step indices corresponding to each loss value
    steps_recorded = list(range(len(all_losses)))
    plt.plot(steps_recorded, all_losses, linestyle='-', label='Training Loss')
    
    # Mark best loss point
    if best_step < len(all_losses):
        plt.plot(best_step, best_loss, 'ro', markersize=10, label=f'Best Loss: {best_loss:.6f}')
        plt.axvline(x=best_step, color='r', linestyle='--', alpha=0.5, label=f'Best Step: {best_step}')
    
    plt.title(
        f'Training Loss ({args.k_qry}-shot MAML)\n'
        f'Meta LR={args.meta_lr}, Task LR={args.update_lr}',
        fontsize=16
    )
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_fig = os.path.join(
        args.save_init,
        f'{args.k_qry}shot_training_loss_curve_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png'
    )
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig)
    print(f'Training curve saved to: {out_fig}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     type=str,   default="Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    parser.add_argument('--device',   type=str,   default='cuda:1')
    parser.add_argument('--save_init',type=str,   default="SISO_UMi_init/Second_order_MAML")
    parser.add_argument('--epoch',    type=int,   default=5000)
    parser.add_argument('--n_way',    type=int,   default=4)
    parser.add_argument('--k_spt',    type=int,   default=5)
    parser.add_argument('--k_qry',    type=int,   default=5)
    parser.add_argument('--batchsz',  type=int,   default=8)
    parser.add_argument('--meta_lr',  type=float, default=5e-4)
    parser.add_argument('--update_lr',type=float, default=1e-3)
    parser.add_argument('--update_step', type=int, default=3)
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which learning rate will be reduced ')
    parser.add_argument('--scheduler_patience', type=int, default=8,
                        help='Number of epochs with no improvement after which LR will be reduced ')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                        help='Lower bound on learning rate ')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for clipping (default: 1.0, set to 0 or negative to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Number of steps with no improvement after which training will be stopped (default: 20, set to 0 to disable early stopping)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-5,
                        help='Minimum change in loss to qualify as an improvement (default: 1e-5)')
    parser.add_argument('--early_stop_restore_best', action='store_true', default=True,
                        help='Restore best model weights when early stopping triggers (default: True)')
    parser.add_argument('--early_stop_save_best', action='store_true', default=True,
                        help='Save best model checkpoint when improvement occurs (default: True)')
    args = parser.parse_args()
    main(args)

#     device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
#     maml = Meta(args, config)
#     maml = Meta(args, config).to(device)
#     if torch.cuda.device_count() > 1:
#         maml = torch.nn.DataParallel(maml)  # Distribute across multiple GPUs
        
#     maml.to(device)
#     # Calculate number of trainable parameters
#     tmp = filter(lambda x: x.requires_grad, maml.parameters())
#     num = sum(map(lambda x: np.prod(x.shape), tmp))
#     print(maml)
#     print(f"Total trainable parameters: {num}")

#     db_train = ChannelEstimationNShot(args.root,  
#                                       batchsz=args.batchsz,
#                                       n_way=args.n_way,
#                                       k_shot=args.k_spt,
#                                       k_query=args.k_qry)

#     # print([k for k in db_train.unique_samples_dict['train'].keys())
    
#     train_losses = []
#     for step in range(args.epoch):
#         # Get training batch
#         (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), qry_name, spt_name, fixed_name, qry_denom, spt_denom,  rx_signal, tx_signal = db_train.next()
#         # pdb.set_trace()
        
#         # Convert to tensors and move to device
#         x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld, ys_fixed_scld = \
#             torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
#             torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
#             torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(ys_fixed_scld).to(device)

#         x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed = \
#             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
#             torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
#             torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)
        
#         # Training phase
#         # pdb.set_trace()
#         # print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=False))
#         losses = maml(x_qry_scld, y_qry, x_spt_scld, y_spt)
#         # print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=False))
#         # pdb.set_trace()
#         current_loss = losses[-1].item()
        
#         if step % 100 == 0 or step == args.epoch-1:
#             train_losses.append(current_loss)
#             print(f'step: {step}, training loss: {current_loss}')

#             # Save training checkpoint
#             meta_model_chpoint = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}")

#             os.makedirs( meta_model_chpoint, exist_ok=True)
            
#             Utils.save_checkpoint({'step': step, 'state_dict': maml.state_dict() }, os.path.join(meta_model_chpoint ,f"MAML_{args.k_qry}_shot_{args.k_spt}_query_checkpoint_step_{step}_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.pth.tar"))

#             # Visualize predictions
#             # pdb.set_trace()
#             # train_pred = [maml.predict(xs_fixed_scld[:, i, :, :, :]).cpu().numpy() for i in range(0, xs_fixed_scld.shape[0], 9)]
#             # train_pred = np.stack(train_pred, axis=0)
#             # np.save(os.path.join(args.save_init, f'{args.k_qry}_shot_MAML_train_prediction_epoch{step}.npy'), train_pred)
#             # Utils.visualize(train_pred, ys_fixed.cpu().numpy(), num_samples=5, sub_title_1="scaled training prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"training_prediction_{step}.png", title=f"Prediction at step {step}")
            
        
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(0, len(train_losses) * 50, 50), train_losses, color='b')
#     plt.title(f'Training Loss Curve of {args.k_qry}_shot_MAML. Meta_Lr = {args.meta_lr}, Task_Lr{args.update_lr}', fontsize=16)
#     plt.xlabel('Steps', fontsize=14)
#     plt.ylabel('Loss', fontsize=14)
#     plt.savefig(os.path.join(args.save_init, f'{args.k_qry}training_loss_curve_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.png'))     
        
# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--root', type=str, help='path to processed_data dir', default="Sionna_datasets/ps2_p612/speed5/SISO-UMi")
#     argparser.add_argument('--device', type=str, help='device to run the process', default='cuda:0')
#     argparser.add_argument('--save_init', type=str, help='path to save directory', default="SISO_UMi_init")
#     argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
#     argparser.add_argument('--n_way', type=int, help='n way', default=2)
#     argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=15)
#     argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
#     argparser.add_argument('--batchsz', type=int, help='meta batch size', default=8)
#     argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
#     argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
#     argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
#     # argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=100)

#     args = argparser.parse_args()
#     main(args)
    
    
# conda activate pytorch_env  
# cd /path/to/Wireless_communication   # example: clone location 
# python MAML_trainer.py  