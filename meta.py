import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from learner import Learner
import pdb

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
        self.update_step_test = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay= 0.05)

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
        batchsz, setsz, c_, h, w = x_qry.size()
        querysz = x_qry.size(1)

        losses_s = [0 for _ in range(self.update_step + 1)]

        for i in range(batchsz):
            # pdb.set_trace()
            x_qry_i = x_qry[i].view(setsz, c_, h, w)
            y_qry_i = y_qry[i].view(setsz, c_, h, w)
            logits = self.net(x_qry_i, vars=None, bn_training=True)
            # logits = logits.view(setsz, c_, h, w)
            loss = F.mse_loss(logits, y_qry_i)
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

            for k in range(1, self.update_step):
                # print("inner step is {}".format(k))
                logits = self.net(x_qry_i, fast_weights, bn_training=True)
                logits = logits.view(setsz, c_, h, w)  
                loss = F.mse_loss(logits, y_qry_i)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_s = self.net(x_spt_i, fast_weights, bn_training=True)
                logits_s = logits_s.view(querysz, c_, h, w)  
                loss_s = F.mse_loss(logits_s, y_spt_i)
                losses_s[k + 1] += loss_s

        loss_q = losses_s[-1] / batchsz

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return losses_s
    
    def finetunning(self, x_qry, y_qry, x_spt, y_spt):
        # Ensure the inputs have the correct dimensions
        querysz = x_qry.size(0)
        
        # Create a deepcopy of the network for fine-tuning
        net = deepcopy(self.net)

        # Initial forward pass
        logits = net(x_qry)
        logits = logits.view(x_qry.size())  
        loss = F.mse_loss(logits, y_qry)

        # Compute gradients and update fast weights
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # Fine-tuning loop
        for k in range(1, self.update_step_test):
            logits = net(x_qry, fast_weights, bn_training=True)
            logits = logits.view(x_qry.size())  
            loss = F.mse_loss(logits, y_qry)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # Evaluate on support set
            logits_s = net(x_spt, fast_weights, bn_training=True)
            logits_s = logits_s.view(x_spt.size())  
            loss_q = F.mse_loss(logits_s, y_spt)

        del net  # Clean up

        return loss_q

    # def finetunning(self, x_qry, y_qry, x_spt, y_spt):
    #     # assert len(x_spt.shape) == 4

    #     querysz = x_qry.size(0)

    #     net = deepcopy(self.net)
    #     # pdb.set_trace()
    #     logits = net(x_qry)
    #     logits = logits.view(x_qry.size())  
    #     loss = F.mse_loss(logits, y_qry)
    #     grad = torch.autograd.grad(loss, net.parameters())
    #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

    #     with torch.no_grad():
    #         logits_s = net(x_spt, net.parameters(), bn_training=True)
    #         logits_s = logits_s.view(x_spt.size())  
    #         loss_s = F.mse_loss(logits_s, y_spt)

    #     with torch.no_grad():
    #         logits_s = net(x_spt, fast_weights, bn_training=True)
    #         logits_s = logits_s.view(x_spt.size())  
    #         loss_s = F.mse_loss(logits_s, y_spt)

    #     for k in range(1, self.update_step_test):
    #         logits = net(x_qry, fast_weights, bn_training=True)
    #         logits = logits.view(x_qry.size())  
    #         loss = F.mse_loss(logits, y_qry)
    #         grad = torch.autograd.grad(loss, fast_weights)
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

    #         logits_s = net(x_spt, fast_weights, bn_training=True)
    #         logits_s = logits_s.view(x_spt.size())  
    #         loss_q = F.mse_loss(logits_s, y_spt)

    #     del net

    #     return loss_q
    
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
