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
        # self.update_step_test = args.update_step_test

        self.net = Learner(config)
        # AdamW is recommended for MAML with CNN architectures
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr, weight_decay=0.01)
        
        # Learning rate scheduler for meta optimizer (similar to ChannelNet's ReduceLROnPlateau)
        # Default: factor=0.5, patience=8, min_lr=1e-7 (can be overridden via args)
        scheduler_factor = getattr(args, 'scheduler_factor', 0.5)
        scheduler_patience = getattr(args, 'scheduler_patience', 8)
        scheduler_min_lr = getattr(args, 'scheduler_min_lr', 1e-7)
        self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optim, mode='min', factor=scheduler_factor, 
            patience=scheduler_patience, min_lr=scheduler_min_lr, verbose=True
        )
        
        # Alternative optimizers:
        # self.meta_optim = optim.RMSprop(self.net.parameters(), lr=self.meta_lr, alpha=0.99, weight_decay=0.01)
        # self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr, momentum=0.9, weight_decay=0.01)
        # self.meta_optim = optim.SparseAdam(self.net.parameters(), lr=self.meta_lr) . #did not work because the gradiant is not sparse

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
        # x_qry = x_qry.contiguous()
        # x_spt = x_spt.contiguous()
        # y_qry = y_qry.contiguous()
        # y_spt = y_spt.contiguous()
        
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_qry.size(1)

        losses_s = [0 for _ in range(self.update_step + 1)]

        for i in range( self.n_way):
            # pdb.set_trace()
            # 1. run the i-th task and compute loss for k=0
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            logits = self.net(x_qry_i, vars=None, bn_training=True)                   
            # logits = logits.view(setsz, c_, h, w)
            loss = F.mse_loss(logits, y_qry_i)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  
            
            # this is the loss and accuracy before first update
            with torch.no_grad():
                x_spt_i = x_spt[i].view(querysz, c_, h, w)
                y_spt_i = y_spt[i].view(querysz, c_, h, w)
                logits_s = self.net(x_spt_i, self.net.parameters(), bn_training=True)
                logits_s = logits_s.view(querysz, c_, h, w) 
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[0] += loss_s

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_s = self.net(x_spt_i, fast_weights, bn_training=True)  
                logits_s = logits_s.view(querysz, c_, h, w) 
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[1] += loss_s

            for k in range(1, self.update_step):
                # print("inner step is {}".format(k))
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                logits = logits.view(setsz, c_, h, w)  
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_s = self.net(x_qry_i, fast_weights, bn_training=True)
                logits_s = logits_s.view(querysz, c_, h, w)  
                loss_s = F.mse_loss(logits_s, y_qry_i)
                losses_s[k + 1] += loss_s

        loss_q = losses_s[-1] / batchsz

        self.meta_optim.zero_grad()
        loss_q.backward() 
        # meta update
        self.meta_optim.step()
        
        # Update learning rate scheduler (step with loss value)
        # Note: This should be called after optimizer.step() in the training loop
        # The scheduler will be stepped from the training script with the loss value

        return losses_s
    
    
    
    
    def finetuning(self, optimizer, data, label, epochs, batchsz, device, save_loss ,patience=10, delta=1e-4, mode = 'train'):
        """
        Fine-tune the network on a single task.
        :param data: Input data [batchsz, c_, h, w]
        :param label: Target labels [batchsz, c_, h, w]
        :param epochs: Number of fine-tuning epochs
        :param batchsz: Batch size for fine-tuning
        :param device: Device to run the computations on (CPU/GPU)
        :return: Final loss after fine-tuning
        """
        # Create a deepcopy of the network for fine-tuning
        # net = deepcopy(self.net).to(device)
        
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
                    batch_data, _, _ = Utils.trch_unit_scaling(batch_data, batch_labels)
                    
                    # Forward pass
                    logits = self.net(batch_data)
                    logits = logits.view(batch_data.size())
                    loss = F.mse_loss(logits, batch_labels)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                scheduler.step()

                # Log epoch details
                avg_loss = epoch_loss / len(data)  # Average loss over batches
                epoch_losses.append(avg_loss)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                                        
                if avg_loss < best_loss - delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0  # Reset counter
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
                batch_data, _, _ = Utils.trch_unit_scaling(batch_data, batch_labels)

                logits = self.net(batch_data)
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
        :param x: Input data [batchsz, c_, h, w]
        :return: Predicted output [batchsz, c_, h, w]
        """
        with torch.no_grad():
            logits = self.net(x, vars=None, bn_training=False)
            return logits
        
        
def main():
    pass

if __name__ == '__main__':
    main()
