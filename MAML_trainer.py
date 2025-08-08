
# torch.cuda.empty_cache()
# print(torch.cuda.device_count())
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE']  = '1'
os.environ['NCCL_DEBUG']       = 'INFO'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# torch.distributed.init_process_group(backend='gloo')
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
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
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_losses = []
    for step in range(args.epoch):
        # fetch one meta‐batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
        xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = db_train.next()

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

        # forward / meta‐update
        losses = maml(x_qry_scld, y_qry, x_spt_scld, y_spt)
        current_loss = losses[-1].item()

        if step % 1000 == 0 or step == args.epoch - 1:
            train_losses.append(current_loss)
            print(f'step: {step}, training loss: {current_loss}')

            # prepare state_dict for saving (unwrap DataParallel)
            state_dict = maml.module.state_dict() if isinstance(maml, torch.nn.DataParallel) else maml.state_dict()
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
                {'step': step, 'state_dict': state_dict},
                ckpt_path
            )

    # plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(0, len(train_losses) * 100, 100),
        train_losses,
        linestyle='-'
    )
    plt.title(
        f'Training Loss ({args.k_qry}-shot MAML)\n'
        f'Meta LR={args.meta_lr}, Task LR={args.update_lr}',
        fontsize=16
    )
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    out_fig = os.path.join(
        args.save_init,
        f'{args.k_qry}shot_training_loss_curve_MetaLr{args.meta_lr}_TaskLr{args.update_lr}.png'
    )
    plt.savefig(out_fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     type=str,   default="new_data")
    parser.add_argument('--device',   type=str,   default='cuda:0')
    parser.add_argument('--save_init',type=str,   default="/home/CAMPUS/rghasemi/projects/MyPrivaterepo/new_data/Multi_GPU_for_more_shot")
    parser.add_argument('--epoch',    type=int,   default=4000)
    parser.add_argument('--n_way',    type=int,   default=5)
    parser.add_argument('--k_spt',    type=int,   default=15)
    parser.add_argument('--k_qry',    type=int,   default=15)
    parser.add_argument('--batchsz',  type=int,   default=8)
    parser.add_argument('--meta_lr',  type=float, default=1e-4)
    parser.add_argument('--update_lr',type=float, default=1e-3)
    parser.add_argument('--update_step', type=int, default=2)
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
# cd /home/CAMPUS/rghasemi/projects/MyPrivaterepo 
# python MAML_trainer.py  