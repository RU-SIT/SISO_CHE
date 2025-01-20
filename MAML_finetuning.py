
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils import Utils
import argparse
from meta import Meta
import pdb



def fine_tune(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

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

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    maml_finetuning = Meta(args, config).to(device)

    data_dict = np.load(os.path.join(args.root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(args.root, 'channel_label_dict.npy'), allow_pickle=True).item()

    fine_tune_file_names = list(data_dict.keys())[10:]

    for outer_channel_name in fine_tune_file_names:
        
        data = torch.tensor(data_dict[outer_channel_name].transpose(0, 3, 1, 2), dtype=torch.float32)
        labels = torch.tensor(labels_dict[outer_channel_name].transpose(0, 3, 1, 2), dtype=torch.float32)

        train_data, eval_data = data[:30], data[30:]
        train_labels, eval_labels = labels[:30], labels[30:]
        data_shot, label_shot = train_data[:args.k_qry], train_labels[:args.k_qry]
        # pdb.set_trace()
        checkpoint_path = os.path.join(
            args.save_init,
            f"meta_model_nway_{args.n_way}",
            f"MAML_{args.k_qry}_shot_checkpoint_step_{args.step-1}_Meta_Lr{args.meta_lr}_Task_Lr{args.update_lr}.pth.tar"
        )
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        maml_finetuning.load_state_dict(checkpoint['state_dict'])

        optimizer = torch.optim.Adam(maml_finetuning.parameters(), lr=args.update_lr)

        print(f"Fine-tuning on channel: {outer_channel_name}")
        maml_finetuning.finetuning(optimizer, data_shot, label_shot, args.epoch, args.batchsz, device)

        # Save the fine-tuned model
        torch.save({'state_dict': maml_finetuning.state_dict()},
                   os.path.join(args.save_init,f"meta_model_nway_{args.n_way}", f"MAML_{args.k_qry}_shot_fine_tuned_model_{outer_channel_name}_lr{args.update_lr}.pth"))

        # Evaluate the model
        eval_save_path = os.path.join(args.save_init,f"meta_model_nway_{args.n_way}", f"MAML_{args.k_qry}_shot_{outer_channel_name}_predictions.npy")
        eval_loss, predictions = maml_finetuning.evaluate(eval_data, eval_labels, args.batchsz, device, save_path = eval_save_path)
        print(f"MAML_{args.k_qry}_shot[{outer_channel_name}] Evaluation Loss: {eval_loss:.4f}")
        print(f"Predictions saved to {eval_save_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, default="new_data")
    argparser.add_argument('--device', type=str, default='cuda:1')
    argparser.add_argument('--save_init', type=str, default="saved_init")
    argparser.add_argument('--step', type=int, default=3000)
    argparser.add_argument('--epoch', type=int, default=500)
    argparser.add_argument('--batchsz', type=int, default=8)
    argparser.add_argument('--k_qry', type=int, default=15)
    argparser.add_argument('--k_spt', type=int, default=15)
    argparser.add_argument('--update_lr', type=float, default=1e-3)
    argparser.add_argument('--meta_lr', type=float, default=1e-4)
    argparser.add_argument('--n_way', type=int, default=3)
    argparser.add_argument('--update_step', type=int, default=5)

    args = argparser.parse_args()
    fine_tune(args)


