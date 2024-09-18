import torch
import numpy as np
from Data_Nshot import ChannelEstimationNShot
from metrics import Metric  # Assuming you have a Metric class for BER
from utils import save_checkpoint, visualize
import argparse
from meta import Meta
import matplotlib.pyplot as plt

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    metric = Metric()  # Assuming you have a Metric class for BER

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
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
                                      k_query=args.k_qry)

    train_losses = []
    test_losses_history = []
    best_model = None
    best_test_loss = float('inf')

    for step in range(args.epoch):
        # Get training batch
        (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld), (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed), qry_denom, spt_denom = db_train.next()
        
        # Convert to tensors and move to device
        x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld = \
            torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
            torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
            torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(xs_fixed_scld).to(device)

        x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed = \
            torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
            torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
            torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)
        # pdb.set_trace()
        # Training phase
        losses = maml(x_qry_scld, y_qry, x_spt_scld, y_spt)
        current_loss = losses[-1].item()

        if step % 50 == 0 or step == args.epoch-1:
            train_losses.append(current_loss)
            print(f'step: {step}, training loss: {current_loss}')

            # Save training checkpoint
            save_checkpoint({'step': step, 'state_dict': maml.state_dict()}, f"checkpoint_step_{step}.pth.tar")

            # Visualize predictions for a fixed set of samples
            train_pred = []
            for i in range(min(len(xs_fixed_scld), args.batchsz)):
                training_pred = maml.predict(xs_fixed_scld[i]).cpu().numpy()
                train_pred.append(training_pred)

            train_pred = np.stack(train_pred, axis=0)
            visualize(train_pred, ys_fixed.cpu().numpy(), num_samples=5, sub_title_1="scaled training prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"training_prediction_{step}.png", title=f"Prediction at step {step}")

        if step % 500 == 0 or step == args.epoch-1:
            # Fine-tuning and test phase
            test_losses = []
            predictions = []
            target_fixed_one = []

            # Get test batch
            (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld), (x_qry, y_qry, x_spt, y_spt ,xs_fixed,ys_fixed), qry_denom, spt_denom = db_train.next('test')
            
            # Convert to tensors and move to device
            x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld = \
                torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
                torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
                torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(ys_fixed_scld).to(device)
                
            x_qry, y_qry, x_spt, y_spt, xs_fixed ,ys_fixed = \
                torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
                torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device), \


            # pdb.set_trace()    
            x_qry_scld, y_qry, x_spt_scld, y_spt, xs_fixed_scld, ys_fixed  = x_qry_scld.squeeze(), y_qry.squeeze(), x_spt_scld.squeeze(), \
                                                                            y_spt.squeeze(), xs_fixed_scld.squeeze(), ys_fixed.squeeze()
            # Fine-tune and predict using the fine-tuned model
            test_loss = maml.finetunning(x_qry_scld, y_qry, x_spt_scld, y_spt)
            test_losses_history.append(test_loss.item())
            print(f'Test loss: {test_loss.item()}')
            save_checkpoint({'step': step, 'state_dict': maml.state_dict()}, f"fine_tuned_model_{step}.pth.tar")
            
            pred = maml.predict(xs_fixed_scld)
            predictions.append(pred.cpu().numpy())
            target_fixed_one.append(ys_fixed.cpu())

            # Visualize predictions after fine-tuning
            predictions = np.stack(predictions, axis=0)
            target_fixed_one = np.stack(target_fixed_one, axis=0)
            visualize(predictions, target_fixed_one, num_samples=5, sub_title_1="Unscaled test prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"test_prediction_{step}.png", title=f"Test Prediction at Step {step}")

    # Save final model checkpoint
    save_checkpoint({'step': args.epoch, 'state_dict': maml.state_dict()}, "final_model.pth.tar")

    # Plot the training and test losses
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle("Training and Test Losses", fontsize=16)

    # Plot training loss
    axes[0].plot(np.arange(0, len(train_losses) * 50, 50), train_losses, label='Training Loss', marker='o')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # Plot test loss
    axes[1].plot(np.arange(0, len(test_losses_history) * 500, 500), test_losses_history, label='Test Loss', marker='o', color='orange')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Test Loss Over Time')
    axes[1].legend()
    axes[1].grid(True)

    plt.savefig("losses_plot.png", dpi=300)
    
    # Calculate BER for SNR values 0, 5, and 10 dB using the last predicted channel
    snrs = [-5, 0, 5, 10, 20]
    bers = []
    # pdb.set_trace()
    for snr in snrs:
        ber = metric.bit_error_rate(predictions[0, 1, :, :, :], snr, modulation='16QAM')
        bers.append(ber)
        print(f"BER for SNR = {snr} dB: {ber}")

    # Plot BER for SNR values
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, bers, marker='o', color='red', label="BER for SNR values")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER for SNR values (0, 5, 10 dB)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ber_plot.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='path to processed_data dir', default="new_data")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2500)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--batchsz', type=int, help='meta batch size', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=100)

    args = argparser.parse_args()

    main(args)


# import torch
# import numpy as np
# from Data_Nshot import ChannelEstimationNShot
# from metrics import Metric  # Import the Metric class
# import argparse
# from meta import Meta
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import pdb

# def save_checkpoint(state, filename="checkpoint.pth.tar"):
#     torch.save(state, filename)
#     print(f"Checkpoint saved to {filename}")

# def visualize(input, output, num_samples=5, sub_title_1="sub_title_1", sub_title_2="sub_title_2", path="visualizations", fig_path="fig_path", title="Sample Visualization"):
#     save_dir = os.path.dirname(path)
#     fig_path = os.path.join(save_dir, fig_path)

#     fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(16, 4*num_samples))
#     fig.suptitle(title, fontsize=12)
    
#     cmap = sns.color_palette("viridis", as_cmap=True)

#     for i in range(num_samples):
#         # Plot real part
#         real_sample_input = input[0, i, 0, :, :]
#         real_sample_output = output[0, i, 0, :, :]
#         imag_sample_input = input[0, i, 1, :, :]
#         imag_sample_output = output[0, i, 1, :, :]

#         im0 = sns.heatmap(real_sample_input, ax=axes[i, 0], cmap=cmap, cbar=False)
#         axes[i, 0].set_title(sub_title_1 + f'Real Part {i + 1}')
#         axes[i, 0].axis('off')

#         im1 = sns.heatmap(real_sample_output, ax=axes[i, 1], cmap=cmap, cbar=False)
#         axes[i, 1].set_title(sub_title_2 + f'Real Part {i + 1}')
#         axes[i, 1].axis('off')

#         # Add shared color bar for real part
#         cbar_real = fig.colorbar(im1.get_children()[0], ax=axes[i, 1], location='right')
#         cbar_real.set_label('Intensity', rotation=270, labelpad=20)

#         # Plot imaginary part
#         im2 = sns.heatmap(imag_sample_input, ax=axes[i, 2], cmap=cmap, cbar=False)
#         axes[i, 2].set_title(sub_title_1 + f'Imag Part {i + 1}')
#         axes[i, 2].axis('off')

#         im3 = sns.heatmap(imag_sample_output, ax=axes[i, 3], cmap=cmap, cbar=False)
#         axes[i, 3].set_title(sub_title_2 + f'Imag Part {i + 1}')
#         axes[i, 3].axis('off')

#         # Add shared color bar for imaginary part
#         cbar_imag = fig.colorbar(im3.get_children()[0], ax=axes[i, 3], location='right')
#         cbar_imag.set_label('Intensity', rotation=270, labelpad=20)

#     # Adjust the space to fit color bars correctly
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.3)
    
#     plt.savefig(fig_path, dpi=300)
#     plt.close(fig)
#     print(f"Figure saved to {fig_path}")
    
# def main(args):
#     torch.manual_seed(222)
#     torch.cuda.manual_seed_all(222)
#     np.random.seed(222)

#     print(args)
#     metric = Metric()  # Create a Metric object for BER calculation

#     # Avg pooling to make output smoother
#     config = [
#         ('conv2d', [16, 2, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [16]),
#         ('conv2d', [8, 16, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [8]),
#         ('conv2d', [args.batchsz, 8, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [args.batchsz]),
#         ('conv2d', [16, args.batchsz, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [16]),
#         ('conv2d', [16, 16, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [16]),
#         ('conv2d', [args.batchsz, 16, 3, 3, 1, 1]),
#         ('tanh', [True]),
#         ('avg_pool2d', [3, 1, 1]),
#         ('bn', [args.batchsz]),
#         ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
#     ]
    
#     device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#     maml = Meta(args, config).to(device)

#     tmp = filter(lambda x: x.requires_grad, maml.parameters())
#     num = sum(map(lambda x: np.prod(x.shape), tmp))
#     print(maml)
#     print('Total trainable tensors:', num)

#     db_train = ChannelEstimationNShot(args.root,  
#                                       batchsz=args.batchsz,
#                                       n_way=args.n_way,
#                                       k_shot=args.k_spt,
#                                       k_query=args.k_qry)

#     train_losses = []
#     test_losses_history = []

#     for step in range(args.epoch):
#         # Get training data
#         (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld), (x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed), qry_denom, spt_denom = db_train.next()
        
#         x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld = torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
#                                                              torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
#                                                                 torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(xs_fixed_scld).to(device)
                                                             
                                                             
#         x_qry, y_qry, x_spt, y_spt, xs_fixed, ys_fixed = torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
#                                          torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
#                                             torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)

#         qry_denom = torch.from_numpy(qry_denom).to(device).squeeze()

#         # Training phase
#         losses = maml(x_qry_scld, y_qry, x_spt_scld, y_spt)
#         current_loss = losses[-1].item()

#         if step % 50 == 0 or step == args.epoch-1:
#             train_losses.append(current_loss)
#             print(f'step: {step}, training loss: {current_loss}')

#             train_pred = []
#             for i in range(min(len(xs_fixed_scld), args.batchsz)): 
#                 x_qry_one = xs_fixed_scld[i]
#                 training_pred = maml.predict(x_qry_one)
#                 train_pred.append(training_pred.cpu().numpy())

#             train_pred = np.stack(train_pred, axis=0)

#             # Visualize the first 5 samples
#             visualize(train_pred, ys_fixed.cpu().numpy(), num_samples=5, sub_title_1="scaled training prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"training_prediction_{step}.png", title=f"Prediction at step {step}")
            
#         if step % 500 == 0 or step == args.epoch-1:
#             test_losses = []
#             predictions = []
#             target_fixed_one = []

#             for _ in range(1000 // args.batchsz):
#                 # Get test data
#                 (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld), (x_qry, y_qry, x_spt, y_spt ,xs_fixed,ys_fixed), qry_denom, spt_denom = db_train.next('test')
#                 pdb.set_trace()
                
#                 x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld = torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
#                                                                     torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
#                                                                         torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(ys_fixed_scld).to(device)
#                 x_qry, y_qry, x_spt, y_spt, xs_fixed,ys_fixed = torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
#                                              torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
#                                                 torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)

#                 qry_denom = torch.from_numpy(qry_denom).to(device).squeeze()

#                 # Loop over the support set and query set for fine-tuning and predictions
#                 for x_qry_one, y_qry_one, x_spt_one, y_spt_one, xs_fixed_scld_one ,ys_fixed_one in zip(x_qry_scld, y_qry, x_spt_scld, y_spt, xs_fixed_scld,ys_fixed):
#                     test_loss = maml.finetunning(x_qry_one, y_qry_one, x_spt_one, y_spt_one)
#                     test_losses.append(test_loss.item())
                    
#             pdb.set_trace()
#             (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld), (x_qry, y_qry, x_spt, y_spt ,xs_fixed,ys_fixed), qry_denom, spt_denom = db_train.next('test')
                
#             x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, xs_fixed_scld ,ys_fixed_scld = torch.from_numpy(x_qry_scld).to(device), torch.from_numpy(y_qry_scld).to(device), \
#                                                                     torch.from_numpy(x_spt_scld).to(device), torch.from_numpy(y_spt_scld).to(device), \
#                                                                         torch.from_numpy(xs_fixed_scld).to(device), torch.from_numpy(ys_fixed_scld).to(device)
#             x_qry, y_qry, x_spt, y_spt, xs_fixed,ys_fixed = torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device), \
#                                              torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
#                                                 torch.from_numpy(xs_fixed).to(device), torch.from_numpy(ys_fixed).to(device)
#             for x_qry_one, y_qry_one, x_spt_one, y_spt_one, xs_fixed_scld_one ,ys_fixed_one in zip(x_qry_scld, y_qry, x_spt_scld, y_spt, xs_fixed_scld,ys_fixed):
#                     pred = maml.predict(xs_fixed_scld_one)                    
#                     predictions.append(pred.cpu().numpy())
#                     target_fixed_one.append(ys_fixed_one.cpu())


#             predictions = np.stack(predictions, axis=0)
#             target_fixed_one = np.stack(target_fixed_one, axis=0)

#             # Visualize the first 5 samples during testing
#             visualize(predictions, target_fixed_one, num_samples=5, sub_title_1="Unscaled test prediction", sub_title_2="Ground truth", path="visualizations", fig_path=f"test_prediction_{step}.png", title=f"test_Prediction at Step {step}")

#             test_loss_avg = np.array(test_losses).mean(axis=0).astype(np.float16)
#             test_losses_history.append(test_loss_avg)
#             print(f'Test loss: {test_loss_avg}')

#     # Plot the training and test losses
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))  # Adjust layout for loss plots
#     fig.suptitle("Training and Test Losses", fontsize=16)

#     # Plot training loss
#     axes[0].plot(np.arange(0, len(train_losses) * 50, 50), train_losses, label='Training Loss', marker='o')
#     axes[0].set_xlabel('Step')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Training Loss Over Time')
#     axes[0].legend()
#     axes[0].grid(True)

#     # Plot test loss
#     axes[1].plot(np.arange(0, len(test_losses_history) * 500, 500), test_losses_history, label='Test Loss', marker='o', color='orange')
#     axes[1].set_xlabel('Step')
#     axes[1].set_ylabel('Loss')
#     axes[1].set_title('Test Loss Over Time')
#     axes[1].legend()
#     axes[1].grid(True)

#     plt.savefig("losses_plot.png", dpi=300)
#     plt.show()

#     # **Calculate BER for SNR values 0, 5, and 10 dB using the last predicted channel**
#     snrs = [0, 5, 10]
#     last_predictions = predictions[-1]  # Use the last predicted channel for BER calculation
#     bers = []

#     for snr in snrs:
#         ber = metric.bit_error_rate(last_predictions.flatten(), snr, modulation='16QAM')
#         bers.append(ber)
#         print(f"BER for SNR = {snr} dB: {ber}")

#     # Plot BER for SNR values
#     plt.figure(figsize=(8, 6))
#     plt.plot(snrs, bers, marker='o', color='red', label="BER for SNR values")
#     plt.xlabel("SNR (dB)")
#     plt.ylabel("Bit Error Rate (BER)")
#     plt.title("BER for SNR values (0, 5, 10 dB)")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig("ber_plot.png", dpi=300)
#     plt.show()

# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--root', type=str, help='path to processed_data dir', default="new_data")   
#     argparser.add_argument('--epoch', type=int, help='epoch number', default=20)
#     argparser.add_argument('--n_way', type=int, help='n way', default=5)
#     argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
#     argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
#     argparser.add_argument('--batchsz', type=int, help='meta batch size', default=8)
#     argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
#     argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
#     argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
#     argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=100)

#     args = argparser.parse_args()

#     main(args)

