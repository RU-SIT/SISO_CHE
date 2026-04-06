import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner import Learner

class MultigradeMAMLStair(nn.Module):
    """
    Multigrade MAML with stair-like architecture where each grade trains specific layers
    """
    def __init__(self, args, config, num_grades=3):
        super(MultigradeMAMLStair, self).__init__()
        
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.batchsz = args.batchsz
        self.update_step = args.update_step
        self.num_grades = num_grades
        self.current_channel_names = []
        
        # Create a single Learner network (same as original MAML)
        self.net = Learner(config)
        
        # Divide the network into grades based on config layers
        self.grade_layers = self._divide_network_into_grades(config, num_grades)
        
        # Create optimizer for the entire network
        self.meta_optim = optim.AdamW(self.parameters(), lr=self.meta_lr, weight_decay=0.01)
    
    def _divide_network_into_grades(self, config, num_grades):
        """
        Divide the network configuration into grades
        Each grade gets approximately 1/num_grades of the layers
        """
        total_layers = len(config)
        layers_per_grade = total_layers // num_grades
        
        grade_layers = []
        for i in range(num_grades):
            start_idx = i * layers_per_grade
            if i == num_grades - 1:  # Last grade gets remaining layers
                end_idx = total_layers
            else:
                end_idx = (i + 1) * layers_per_grade
            
            grade_layers.append((start_idx, end_idx))
        
        return grade_layers
    
    def _get_grade_parameters(self, grade_idx):
        """
        Get parameters for a specific grade
        """
        start_idx, end_idx = self.grade_layers[grade_idx]
        
        # Get parameters for this grade's layers
        grade_params = []
        layer_idx = 0
        
        for i, (name, param) in enumerate(self.net.config):
            if name in ['conv2d', 'convt2d', 'linear', 'bn']:
                if start_idx <= i < end_idx:
                    # Weight and bias parameters
                    grade_params.append(self.net.vars[layer_idx])
                    grade_params.append(self.net.vars[layer_idx + 1])
                layer_idx += 2
        
        return grade_params
    
    def _update_vars_with_fast_weights(self, grade_idx, fast_weights):
        """
        Update the full network vars with fast weights for a specific grade
        """
        # Start with original vars
        updated_vars = list(self.net.vars)
        
        # Get the parameter indices for this grade
        start_idx, end_idx = self.grade_layers[grade_idx]
        fast_weight_idx = 0
        
        # Update the parameters for this grade
        layer_idx = 0
        for i, (name, param) in enumerate(self.net.config):
            if name in ['conv2d', 'convt2d', 'linear', 'bn']:
                if start_idx <= i < end_idx:
                    # Update weight and bias
                    updated_vars[layer_idx] = fast_weights[fast_weight_idx]
                    updated_vars[layer_idx + 1] = fast_weights[fast_weight_idx + 1]
                    fast_weight_idx += 2
                layer_idx += 2
        
        return updated_vars
    
    def _forward_grade(self, x, grade_idx, vars=None, bn_training=True):
        """
        Forward pass through a specific grade of the network
        """
        if vars is None:
            vars = self.net.vars
        
        start_idx, end_idx = self.grade_layers[grade_idx]
        
        idx = 0
        bn_idx = 0
        
        for i, (name, param) in enumerate(self.net.config):
            if i < start_idx:
                # Skip layers before this grade
                if name in ['conv2d', 'convt2d', 'linear']:
                    idx += 2
                elif name == 'bn':
                    idx += 2
                    bn_idx += 2
                continue
            elif i >= end_idx:
                # Stop at this grade
                break
            
            # Process this layer
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.net.vars_bn[bn_idx], self.net.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                x = x.reshape(x.size(0), -1)
            elif name == 'reshape':
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
        
        return x
    
    def set_channel_names(self, channel_names):
        processed_names = []
        if isinstance(channel_names, (list, tuple, np.ndarray)):
            for entry in channel_names:
                if isinstance(entry, (list, tuple, np.ndarray)):
                    if len(entry) > 0:
                        processed_names.append(str(entry[0]))
                    else:
                        processed_names.append("unknown_channel")
                else:
                    processed_names.append(str(entry))
        else:
            processed_names.append(str(channel_names))
        # Reduce duplicates by taking first occurrence of each block of k_spt entries
        if len(processed_names) >= self.n_way * self.k_spt:
            condensed = []
            for i in range(self.n_way):
                idx = i * self.k_spt
                if idx < len(processed_names):
                    condensed.append(processed_names[idx])
            self.current_channel_names = condensed
        else:
            self.current_channel_names = processed_names

    def forward(self, x_qry, y_qry, x_spt, y_spt, current_grade=None):
        """
        Multigrade MAML forward pass with stair-like architecture
        Each grade trains a portion of the network, but only the final grade's output is used for loss
        
        Args:
            current_grade: If specified, only train this grade (0-indexed). Previous grades are frozen.
        """
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_qry.size(1)
        
        # If current_grade is specified, only train that grade
        if current_grade is not None:
            grades_to_train = [current_grade]
        else:
            # Train all grades (for backward compatibility)
            grades_to_train = list(range(self.num_grades))
        
        # Store losses for each grade
        losses_s = {}
        grade_channel_losses = {grade_idx: [] for grade_idx in range(self.num_grades)}
        for grade_idx in range(self.num_grades):
            losses_s[grade_idx] = [0 for _ in range(self.update_step + 1)]
        
        # Process each task
        for i in range(self.n_way):
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            x_spt_i = x_spt[i].view(querysz, c_, h, w)
            y_spt_i = y_spt[i].view(querysz, c_, h, w)
            
            # Train only the specified grades
            channel_name = self.current_channel_names[i] if i < len(self.current_channel_names) else f"channel_{i}"

            for grade_idx in grades_to_train:
                channel_step_losses = []
                # Get parameters for this grade
                grade_params = self._get_grade_parameters(grade_idx)
                
                # 1. Run the i-th task and compute loss for k=0
                # For stair architecture, we use the full network but only update this grade's parameters
                logits = self.net(x_qry_i, vars=None, bn_training=True)
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, grade_params, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, grade_params)))
                
                # This is the loss before first update (on support set)
                with torch.no_grad():
                    logits_s = self.net(x_spt_i, vars=None, bn_training=False)
                    logits_s = logits_s.view(querysz, c_, h, w)
                    loss_s = F.mse_loss(logits_s, y_spt_i)
                    losses_s[grade_idx][0] += loss_s
                    channel_step_losses.append(loss_s.detach().item())
                
                # This is the loss after the first update (on support set)
                with torch.no_grad():
                    # Create updated vars with fast weights for this grade
                    updated_vars = self._update_vars_with_fast_weights(grade_idx, fast_weights)
                    logits_s = self.net(x_spt_i, vars=updated_vars, bn_training=False)
                    logits_s = logits_s.view(querysz, c_, h, w)
                    loss_s = F.mse_loss(logits_s, y_spt_i)
                    losses_s[grade_idx][1] += loss_s
                    channel_step_losses.append(loss_s.detach().item())
                
                # Inner loop updates
                for k in range(1, self.update_step):
                    # Create updated vars with fast weights for this grade
                    updated_vars = self._update_vars_with_fast_weights(grade_idx, fast_weights)
                    logits = self.net(x_qry_i, vars=updated_vars, bn_training=True)
                    logits = logits.view(setsz, c_, h, w)
                    loss = F.mse_loss(logits, y_qry_i)
                    grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                    
                    # Evaluate on support set
                    updated_vars = self._update_vars_with_fast_weights(grade_idx, fast_weights)
                    logits_s = self.net(x_spt_i, vars=updated_vars, bn_training=True)
                    logits_s = logits_s.view(querysz, c_, h, w)
                    loss_s = F.mse_loss(logits_s, y_spt_i)
                    losses_s[grade_idx][k + 1] += loss_s
                    channel_step_losses.append(loss_s.detach().item())

                # Ensure consistent length
                if channel_step_losses:
                    while len(channel_step_losses) < self.update_step + 1:
                        channel_step_losses.append(channel_step_losses[-1])
                    grade_channel_losses[grade_idx].append({
                        'channel_name': channel_name,
                        'step_losses': channel_step_losses
                    })
        
        # Meta-update using the final losses
        # Only update the grades that were trained
        total_loss = 0
        for grade_idx in grades_to_train:
            total_loss += losses_s[grade_idx][-1]
        
        loss_q = total_loss / batchsz
        
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        
        return losses_s, grade_channel_losses
    
    def finetuning(self, optimizer, data, label, epochs, batchsz, device, save_loss, 
                   patience=10, delta=1e-4, mode='train'):
        """
        Fine-tune the multigrade network on a single task
        """
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        self.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        epoch_losses = []
        
        if mode == "train":
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_data, batch_labels in self.batchify(data, label, batchsz):
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    
                    # Forward pass with full network
                    output = self.net(batch_data, vars=None, bn_training=True)
                    loss = F.mse_loss(output, batch_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                scheduler.step()
                
                avg_loss = epoch_loss / len(data)
                epoch_losses.append(avg_loss)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                
                if avg_loss < best_loss - delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    break
            
            np.save(save_loss, epoch_losses)
            return best_loss
    
    def evaluate(self, eval_data, eval_labels, batchsz, device, save_path=None):
        """
        Evaluate the multigrade model
        """
        self.eval()
        eval_loss = 0.0
        all_predictions = []
        
        with torch.no_grad():
            for batch_data, batch_labels in self.batchify(eval_data, eval_labels, batchsz):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                # Forward pass with full network
                output = self.net(batch_data, vars=None, bn_training=False)
                loss = F.mse_loss(output, batch_labels)
                eval_loss += loss.item()
                
                predictions = output.cpu().numpy()
                all_predictions.append(predictions)
        
        avg_eval_loss = eval_loss / len(eval_data)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        if save_path:
            np.save(save_path, all_predictions)
        
        return avg_eval_loss, all_predictions
    
    def batchify(self, data, labels, batch_size):
        """
        Create batches from data
        """
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield batch_data, batch_labels
    
    def predict(self, x):
        """
        Generate predictions using the full network
        """
        with torch.no_grad():
            output = self.net(x, vars=None, bn_training=False)
            return output
