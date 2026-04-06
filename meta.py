import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from learner import Learner
import pdb
from utils import Utils
import matplotlib.pyplot as plt

class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.batchsz =args.batchsz
        self.update_step = args.update_step
        # Gradient clipping max norm (default: 1.0, set to None to disable)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

        self.net = Learner(config)
        # AdamW is recommended for MAML with CNN architectures
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr, weight_decay=0.01)
        
        # Learning rate scheduler for meta optimizer
        scheduler_factor = getattr(args, 'scheduler_factor', 0.5)
        scheduler_patience = getattr(args, 'scheduler_patience', 8)
        scheduler_min_lr = getattr(args, 'scheduler_min_lr', 1e-7)
        self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optim, mode='min', factor=scheduler_factor, 
            patience=scheduler_patience, min_lr=scheduler_min_lr, verbose=True
        )

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_qry, y_qry, x_spt, y_spt):
        """
        FIXED VERSION of MAML forward pass.
        
        Naming convention in this code:
        - x_qry, y_qry: Data for adaptation (inner loop training)
        - x_spt, y_spt: Data for evaluation (held-out for meta-learning)
        
        Args:
            x_qry: Query/adaptation set [batch_size, n_way, k_shot, 2, 612, 14]
            y_qry: Query/adaptation labels [batch_size, n_way, k_shot, 2, 612, 14]
            x_spt: Support/evaluation set [batch_size, n_way, k_shot, 2, 612, 14]
            y_spt: Support/evaluation labels [batch_size, n_way, k_shot, 2, 612, 14]
        """
        
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_spt.size(1)  # Use support set size

        losses_s = [0 for _ in range(self.update_step + 1)]

        for i in range(batchsz):
            # Extract data for task i
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            x_spt_i = x_spt[i].view(querysz, c_, h, w)
            y_spt_i = y_spt[i].view(querysz, c_, h, w)
            
            
            # STEP 0: Initial forward pass
            
            logits = self.net(x_qry_i, vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_qry_i)
            
            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], 
                                   zip(grad, self.net.parameters())))
            
            # Evaluate on SUPPORT set before first update (for monitoring)
            with torch.no_grad():
                logits_s = self.net(x_spt_i, self.net.parameters(), bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[0] += loss_s

            # Evaluate on SUPPORT set after first update (for monitoring)
            with torch.no_grad():
                logits_s = self.net(x_spt_i, fast_weights, bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[1] += loss_s

            # INNER LOOP: Additional adaptation steps
            
            for k in range(1, self.update_step):
                # Adapt on QUERY set (x_qry_i, y_qry_i)
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_qry_i)
                
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], 
                                       zip(grad, fast_weights)))

                
                # This is the key fix - we must evaluate on held-out data
                logits_s = self.net(x_spt_i, fast_weights, bn_training=False)
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[k + 1] += loss_s

        
        # META-UPDATE: Update base parameters
        
        # Average loss over all tasks
        loss_q = losses_s[-1] / batchsz

        # Compute meta-gradient (differentiates through inner loop!)
        self.meta_optim.zero_grad()
        loss_q.backward()
        
        # Gradient clipping to prevent explosion
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        
        # Meta-update
        self.meta_optim.step()

        return losses_s
    
    
    def finetuning(self, optimizer, data, label, epochs, batchsz, device, save_loss ,patience=10, delta=1e-4, mode = 'train'):
        """
        Fine-tune the network on a single task.
        """
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        self.net.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        epoch_losses = []
        if mode =="train":
            # Fine-tuning loop
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_data, batch_labels in self.batchify(data, label, batchsz):
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    # batch_data, _, _ = Utils.trch_unit_scaling(batch_data, batch_labels)
                    
                    # Forward pass
                    logits = self.net(batch_data, vars=None, bn_training=True)
                    logits = logits.view(batch_data.size())
                    loss = F.mse_loss(logits, batch_labels)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                scheduler.step()

                # Log epoch details
                avg_loss = epoch_loss / len(data)
                epoch_losses.append(avg_loss)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                                        
                if avg_loss < best_loss - delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s).")

                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    break
                
            np.save(save_loss,epoch_losses)
            return best_loss
        
        
    def evaluate(self, eval_data, eval_labels, batchsz, device, save_path=None):
        """
        Evaluate the fine-tuned model on the evaluation set.
        """
        self.net.eval()
        eval_loss = 0.0
        all_predictions = []
        with torch.no_grad():
            for batch_data, batch_labels in self.batchify(eval_data, eval_labels, batchsz):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                # batch_data, _, _ = Utils.trch_unit_scaling(batch_data, batch_labels)

                logits = self.net(batch_data, vars=None, bn_training=False)
                logits = logits.view(batch_data.size())
                loss = F.mse_loss(logits, batch_labels)
                eval_loss += loss.item()
                
                predictions = logits.cpu().numpy()
                all_predictions.append(predictions)

        avg_eval_loss = eval_loss / len(eval_data)
        all_predictions = np.concatenate(all_predictions, axis=0)

        avg_eval_loss = eval_loss / len(eval_data)
        if save_path:
            np.save(save_path, all_predictions)

        return avg_eval_loss, all_predictions

        
    def batchify(self, data, labels, batch_size):
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield batch_data, batch_labels
        
    def predict(self, x):
        """
        Generate predictions for the given input using the current model weights.
        """
        with torch.no_grad():
            logits = self.net(x, vars=None, bn_training=False)
            return logits
        
        
def main():
    pass

if __name__ == '__main__':
    main()

