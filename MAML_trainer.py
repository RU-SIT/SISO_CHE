import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
import numpy as np
from Data_Nshot import ChannelEstimationNShot
from metrics import Metric, mmse_channel_estimation, extract_snr
from utils import Utils
import argparse
from meta import Meta
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.interpolate import griddata
import pdb

def mse(true: torch.Tensor, pred: torch.Tensor) -> float:
    """Computes mean squared error"""
    if true.shape != pred.shape:
        # This can happen if the batch size is 1
        pred = pred.squeeze(0)
    return torch.mean(torch.square(torch.abs(true - pred)))

def evaluate_ls_baseline(eval_data_tensor, eval_labels_tensor, pilot_indices, grid_indices):
    """
    Computes the LS baseline MSE.
    """
    # Extract pilot data
    pilots = eval_data_tensor.cpu().numpy().reshape(-1, eval_data_tensor.shape[-1])
    
    # Perform linear interpolation
    interpolated = griddata(pilot_indices, pilots, grid_indices, method='linear')
    
    # Fill NaNs with nearest value
    interpolated = griddata(pilot_indices, pilots, grid_indices, method='nearest', fill_value=0)
    
    # Calculate MSE
    ls_mse = mse(eval_labels_tensor, torch.from_numpy(interpolated).float())
    return ls_mse

def evaluate_lmmse_baseline(h_ls, true_channels, snr):
    """
    Computes the LMMSE baseline MSE. This is a simplified version and might need adjustments.
    """
    h_ls_flat = h_ls.reshape(h_ls.shape[0], -1)
    true_channels_flat = true_channels.reshape(true_channels.shape[0], -1)
    
    # Estimate channel correlation
    Rhh = torch.mean(torch.einsum('bi,bj->bij', true_channels_flat, true_channels_flat.conj()), axis=0)
    
    # Estimate LS error correlation
    noise_var = 1 / (10**(snr / 10))
    Rnn = torch.eye(Rhh.shape[0]) * noise_var  # Simplified assumption
    
    # LMMSE estimate
    lmmse_filter = Rhh @ torch.inverse(Rhh + Rnn)
    h_lmmse_flat = torch.einsum('ij,bj->bi', lmmse_filter, h_ls_flat)
    
    h_lmmse = h_lmmse_flat.reshape(true_channels.shape)
    
    lmmse_mse = mse(true_channels, h_lmmse)
    return lmmse_mse

def evaluate_model(maml, db_train, args, device):
    """
    Evaluate the MAML model on the test set with CeBed-style evaluation.
    """
    print("Starting CeBed-style evaluation...")
    
    snr_range = np.arange(0, 30, 5)  # Example SNR range
    mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
    
    # Get test data
    (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
    (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
    qry_name, spt_name, fixed_name, qry_denom, spt_denom,  rx_signal, tx_signal = db_train.next('test')

    # Group test data by SNR
    snr_groups = defaultdict(list)
    for i, file_path in enumerate(fixed_name):
        snr = extract_snr(file_path)
        if snr is not None:
            snr_groups[snr].append(i)

    for snr in snr_range:
        if snr not in snr_groups:
            continue

        indices = snr_groups[snr]
        # These are numpy arrays from the dataloader
        eval_data_snr = xs_fixed_scld[0, indices] 
        eval_labels_snr = ys_fixed[0, indices]
        
        # Convert to tensors
        eval_data_tensor = torch.from_numpy(eval_data_snr).to(device)
        eval_labels_tensor = torch.from_numpy(eval_labels_snr).to(device)

        # Predict with the model
        predictions = maml.predict(eval_data_tensor)

        # Calculate MSE
        current_mse = mse(eval_labels_tensor, predictions.cpu())
        
        # Store results
        new_row = {"snr": snr, "mse": current_mse.item(), "method": "MAML", "seed": args.seed}
        mses = pd.concat([mses, pd.DataFrame([new_row])], ignore_index=True)

        # Baselines evaluation
        # Note: These are simplified implementations. For accurate results, pilot info and channel statistics are needed.
        num_subcarriers, num_symbols = 72, 14  # Example dimensions
        pilot_rows, pilot_cols = np.mgrid[0:num_subcarriers:4, 0:num_symbols:2]
        pilot_indices = np.vstack([pilot_rows.ravel(), pilot_cols.ravel()]).T
        grid_rows, grid_cols = np.mgrid[0:num_subcarriers, 0:num_symbols]
        grid_indices = np.vstack([grid_rows.ravel(), grid_cols.ravel()]).T
        
        # LS Baseline
        ls_mse = evaluate_ls_baseline(eval_data_tensor, eval_labels_tensor, pilot_indices, grid_indices)
        mses = pd.concat([mses, pd.DataFrame([{"snr": snr, "mse": ls_mse.item(), "method": "LS", "seed": args.seed}])], ignore_index=True)

        # LMMSE Baseline
        # This requires H_LS, which is the result of LS estimation.
        # We'll use the interpolated result from the LS baseline as a stand-in.
        h_ls_estimated = torch.from_numpy(griddata(pilot_indices, eval_data_tensor.cpu().numpy().reshape(-1, eval_data_tensor.shape[-1]), grid_indices, method='linear', fill_value=0)).float()
        lmmse_mse = evaluate_lmmse_baseline(h_ls_estimated, eval_labels_tensor, snr)
        mses = pd.concat([mses, pd.DataFrame([{"snr": snr, "mse": lmmse_mse.item(), "method": "LMMSE", "seed": args.seed}])], ignore_index=True)

        # ALMMSE is complex to implement without Sionna's environment. It's skipped here.

    print("Evaluation results:")
    print(mses)
    
    # Save results to CSV
    results_path = os.path.join(args.save_init, f"evaluation_results_nway_{args.n_way}.csv")
    mses.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    metric = Metric()

    # Define the model configuration
    config = [
        ('conv2d', [16, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [16]),
        ('conv2d', [8, 16, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [8]),
        ('conv2d', [args.batchsz, 8, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [args.batchsz]),
        ('conv2d', [16, args.batchsz, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [16]),
        ('conv2d', [16, 16, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [16]),
        ('conv2d', [args.batchsz, 16, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [args.batchsz]),
        ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    ]
    
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, config).to(device)

    # Calculate number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print(f"Total trainable parameters: {num}")

    db_train = ChannelEstimationNShot(args.root,  
                                      batchsz=args.batchsz,
                                      n_way=args.n_way,
                                      k_shot=args.k_spt,
                                      k_query=args.k_qry,
                                      seed=args.seed)

    # print([k for k in db_train.unique_samples_dict['train'].keys())
    
    train_losses = []
    for step in range(args.epoch):
        # Get training batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), qry_name, spt_name, fixed_name, qry_denom, spt_denom,  rx_signal, tx_signal = db_train.next()
        # pdb.set_trace()
        
        # Convert to tensors and move to device
        x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld, ys_fixed_scld = \
            torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
            torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
            torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(ys_fixed_scld).to(device)

        x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed = \
            torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
            torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
            torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)
        
        # Training phase
        losses = maml(x_qry_scld, y_qry, x_spt_scld, y_spt)
        # pdb.set_trace()
        current_loss = losses[-1].item()
        
        if step % 100 == 0 or step == args.epoch-1:
            train_losses.append(current_loss)
            print(f'step: {step}, training loss: {current_loss}')

            # Save training checkpoint
            meta_model_chpoint = os.path.join(args.save_init, f"meta_model_nway_{args.n_way}")

            os.makedirs( meta_model_chpoint, exist_ok=True)
            
            Utils.save_checkpoint({'step': step, 'state_dict': maml.state_dict() }, os.path.join(meta_model_chpoint ,f"MAML_{args.k_qry}_shot_checkpoint_step_{step}_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.pth.tar"))

            # Visualize predictions
            # pdb.set_trace()
            # train_pred = [maml.predict(xs_fixed_scld[:, i, :, :, :]).cpu().numpy() for i in range(0, xs_fixed_scld.shape[0], 9)]
            # train_pred = np.stack(train_pred, axis=0)
            # np.save(os.path.join(args.save_init, f'{args.k_qry}_shot_MAML_train_prediction_epoch{step}.npy'), train_pred)
            # Utils.visualize(train_pred, ys_fixed.cpu().numpy(), num_samples=5, sub_title_1="scaled training prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"training_prediction_{step}.png", title=f"Prediction at step {step}")
            
        if step == args.epoch-1:
                db_train.save_unique_samples(args.save_init)
                
                
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(train_losses) * 50, 50), train_losses, color='b')
    plt.title(f'Training Loss Curve of {args.k_qry}_shot_MAML. Meta_Lr = {args.meta_lr}, Task_Lr{args.update_lr}', fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.savefig(os.path.join(args.save_init, f'{args.k_qry}training_loss_curve_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.png'))
    
    evaluate_model(maml, db_train, args, device)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='path to processed_data dir', default="new_data")
    argparser.add_argument('--device', type=str, help='device to run the process', default='cuda:0')
    argparser.add_argument('--save_init', type=str, help='path to save directory', default="results")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3000)
    argparser.add_argument('--n_way', type=int, help='n way', default=7)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--batchsz', type=int, help='meta batch size', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--seed', type=int, help='random seed', default=222)
    # argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=100)

    args = argparser.parse_args()
    main(args)

    
    
# conda activate pytorch_env  
# cd /home/CAMPUS/rghasemi/projects/MyPrivaterepo 
# python MAML_trainer.py  