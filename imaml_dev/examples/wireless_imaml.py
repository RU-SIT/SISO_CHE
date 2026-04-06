import numpy as np
import torch
import torch.nn as nn
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
from datautils import Utils
import sys
from pathlib import Path

_WC_ROOT = Path(__file__).resolve().parents[2]
if str(_WC_ROOT) not in sys.path:
    sys.path.insert(0, str(_WC_ROOT))
from paths import default_dataset_umi_siso_folder, default_imaml_save_dir

from Data_Nshot import ChannelEstimationNShot


np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
logger = DataLog()

# ===================
# hyperparameters
# ===================

parser = argparse.ArgumentParser(description='Implicit MAML on Wireless Channel Estimation')
parser.add_argument('--data_dir', type=str, default=default_dataset_umi_siso_folder(), help='location of the dataset')
parser.add_argument('--N_way', type=int, default=2, help='number of classes for few-shot learning tasks')
parser.add_argument('--K_shot', type=int, default=10, help='number of instances for few-shot learning tasks')
parser.add_argument('--inner_lr', type=float, default=1e-3, help='inner loop learning rate')
parser.add_argument('--outer_lr', type=float, default=1e-4, help='outer loop learning rate')
parser.add_argument('--n_steps', type=int, default = 2, help='number of steps in inner loop') 
parser.add_argument('--meta_steps', type=int, default= 4000, help='number of meta steps')
parser.add_argument('--task_mb_size', type=int, default=8)
parser.add_argument('--lam', type=float, default=2.0, help='regularization in inner steps')
parser.add_argument('--cg_steps', type=int, default=5)
parser.add_argument('--cg_damping', type=float, default=1.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--num_tasks', type=int, default=5)  #we do not need this argument since it is used for task generation in Omniglot
parser.add_argument('--save_dir', type=str, default=default_imaml_save_dir())
parser.add_argument('--load_agent', type=str, default=None)
parser.add_argument('--lam_lr', type=float, default=0.05, help='learning rate for lambda')
parser.add_argument('--lam_min', type=float, default=0.001, help='minimum lambda value')
args = parser.parse_args()
logger.log_exp_args(args)

# ===================
# Load Data
# ===================

print("Generating tasks ...... ")
dataset = ChannelEstimationNShot(
    root=args.data_dir,
    batchsz=args.task_mb_size,
    n_way=args.N_way,
    k_shot=args.K_shot,
    k_query=args.K_shot  # Query set size same as support set
)

# ===================
# Initialize Model
# ===================

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
    
# Count total trainable parameters using the underlying model
tmp = filter(lambda x: x.requires_grad, meta_learner.model.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print(meta_learner)
print(f"Total trainable parameters: {num}")

init_params = meta_learner.get_params()
device = 'cuda' if args.use_gpu else 'cpu'
lam = torch.tensor(args.lam, device=device)

pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# ===================
# Train
# ===================
print("Training model ......")
losses = np.zeros((args.meta_steps, 4))
for outstep in tqdm(range(args.meta_steps)):
    (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld,
        xs_fixed_scld, ys_fixed_scld, eval_data_scld, eval_label_scld), \
        (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed, val_data, val_label), \
        qry_name, spt_name, fixed_name, qry_denom, spt_denom, rx_signal, tx_signal = dataset.next(mode='train')
    
    # pdb.set_trace()
    
    w_k = meta_learner.get_params()
    meta_grad = 0.0
    lam_grad = 0.0
    
    for i in range(args.N_way):
        fast_learner.set_params(w_k.clone())
        task = {
            'x_train': torch.tensor(x_qry_scld[i], device=device),
            'y_train': torch.tensor(y_qry_scld[i], device=device),
            'x_val': torch.tensor(x_spt_scld[i], device=device),
            'y_val': torch.tensor(y_spt_scld[i], device=device)
        }
        # pdb.set_trace()
        
        tl = fast_learner.learn_task(task, num_steps=args.n_steps)
        vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        
        fast_learner.inner_opt.zero_grad()
        regu_loss = fast_learner.regularization_loss(w_k, lam)
        regu_loss.backward()
        fast_learner.inner_opt.step()
        
        vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        
        valid_loss = fast_learner.get_loss(task['x_val'], task['y_val'])
        valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
        flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
        
        if torch.isnan(flat_grad).any():
            print(f"NaN detected in flat_grad at task {i} in meta step {outstep}")
        
        if args.cg_steps <= 1:
            task_outer_grad = flat_grad
        else:
            task_matrix_evaluator = fast_learner.matrix_evaluator(task, lam, args.cg_damping)
            task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)
        
        if torch.isnan(task_outer_grad).any():
            print(f"NaN detected in task_outer_grad at task {i} in meta step {outstep}")
            
        meta_grad += (task_outer_grad / args.task_mb_size)
        losses[outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after]) / args.task_mb_size)
              
        if args.lam_lr <= 0.0:
            task_lam_grad = 0.0
        else:
            print("Warning: lambda learning is not tested for this version of code")
            train_loss = fast_learner.get_loss(task['x_train'], task['y_train'])
            train_grad = torch.autograd.grad(train_loss, fast_learner.model.parameters())
            train_grad = torch.cat([g.contiguous().view(-1) for g in train_grad])
            inner_prod = train_grad.dot(task_outer_grad)
            task_lam_grad = inner_prod / (lam**2 + 0.1)
        lam_grad += (task_lam_grad / args.task_mb_size)
        
    meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
    lam_delta = - args.lam_lr * lam_grad  
    lam = torch.clamp(lam + lam_delta, args.lam_min, 5000.0)
    param_norm = torch.norm(meta_learner.get_params())
    print(f"Meta Step {outstep}: parameter norm = {param_norm.item()}")
    
    if (outstep % 1000 == 0) or (outstep == args.meta_steps - 1):
        train_pre, test_pre, train_post, test_post = losses[outstep]
        print(f"Meta Step {outstep}: Train pre = {train_pre:.4f}, Test pre = {test_pre:.4f}, " 
              f"Train post = {train_post:.4f}, Test post = {test_post:.4f}")
        
        meta_model_chpoint = os.path.join(args.save_dir, f"meta_model_nway_{args.N_way}")
        os.makedirs(meta_model_chpoint, exist_ok=True)
        Utils.save_checkpoint({'step': outstep, 'state_dict': meta_learner.model.state_dict() },
                    os.path.join(meta_model_chpoint, f"wireless_IMAML_{args.K_shot}_shot_checkpoint_step_{outstep}_LAM{args.lam}.pth.tar"))

        
        steps = np.arange(outstep + 1)
        train_post_curve = losses[:outstep+1, 2]
        test_post_curve = losses[:outstep+1, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_post_curve, label='Train post')
        plt.plot(steps, test_post_curve, label='Test post')
        plt.title(f'Training Loss Curve for {args.K_shot}-shot MAML')
        plt.xlabel('Meta Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(meta_model_chpoint, f'Wireless_iMAMAL{args.K_shot}shot__LAM{args.lam}_training_loss_curve.png'))
        plt.close()

