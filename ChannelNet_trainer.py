import os
import numpy as np
from models import SRCNN_model, DNCNN_model, unit_scaling, visualize_and_save_results, DNCNN_predict
from ch_metrics import Metric
import matplotlib.pyplot as plt
# from utils import Utils
import pdb
import argparse

def data_loader(root, n_way, mode):
    data = []
    label = []
    # Load the data and labels
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()  
    labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()
    file_names = list(data_dict.keys())
    
    # Split filenames into training and testing
    train_file_names = file_names[:n_way * 2]
    finetune_file_names = file_names[n_way * 2:]
    
    if mode == 'train':
        # Collect training data and labels
        for file_name in train_file_names:
            data.append(data_dict[file_name])
            label.append(labels_dict[file_name])
    else:
        # Collect testing data and labels
        for file_name in finetune_file_names:
            data.append(data_dict[file_name])
            label.append(labels_dict[file_name])
            
    # pdb.set_trace()
    data_combined = np.concatenate(data, axis=0)   
    label_combined = np.concatenate(label, axis=0)     
    data , _, _  = unit_scaling(data_combined, label_combined)
    
    return data, label_combined



def channel_finder(root, n_way):

    # Load the data and labels
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()  
    file_names = list(data_dict.keys())
    
    # Split filenames into training and testing
    finetune_file_names = file_names[n_way * 2:]
    
    return finetune_file_names

def ft_data_loader(root, ch_name):
    data = []
    label = []
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()  
    labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()
    # Collect testing data and labels
    data = data_dict[ch_name]
    label = labels_dict[ch_name]
    # pdb.set_trace()
    return data, label

def plot_loss(history, title, filename):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def ChannelNet(args):
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    if args.mode == "train":
        # Training SRCNN with noisy labels
        
        train_data, train_label =  data_loader(args.root, args.n_way, args.mode)
        # pdb.set_trace()
        
        srcnn_model = SRCNN_model(lr = args.train_lr)
        srcnn_model.fit(train_data, train_label, args.batchsz, args.train_epoch, verbose=1)
        srcnn_model.save_weights(os.path.join(args.save_init, "SRCNN_trained.weights.h5"))
        srcnn_pred_train = srcnn_model.predict(train_data)
        
        dncnn_model = DNCNN_model(lr = args.train_lr)    
        history = dncnn_model.fit(srcnn_pred_train, train_label, args.batchsz, args.train_epoch, verbose=1)
        plot_loss(history, "DNCNN Training Loss", os.path.join(args.save_init, "DNCNN_loss.png"))
        
        dncnn_model.save_weights(os.path.join(args.save_init, "DNCNN_trained.weights.h5"))
        
    else:
        # Fine-tuning Loop for Each Channel
        finetune_file_names = channel_finder(args.root, args.n_way)
        
        for channel_name in finetune_file_names:

            print(f"Starting fine-tuning for channel: {channel_name}")
            srcnn_model = SRCNN_model(lr = args.train_lr)
            srcnn_model.load_weights(os.path.join(args.save_init, "SRCNN_trained.weights.h5"))
            dncnn_model = DNCNN_model(lr = args.train_lr)
            dncnn_model.load_weights(os.path.join(args.save_init, "DNCNN_trained.weights.h5"))

            # Fine-tune SRCNN with noisy labels
            print(f"SRCNN Fine-tuning for channel {channel_name}")
            data, label = ft_data_loader(args.root, channel_name)
            FT_data , FT_label = data[:30] , label[:30]
            data_shot, label_shot = FT_data[:args.k_qry] , FT_label[:args.k_qry]
            # pdb.set_trace()
            srcnn_model.fit(data_shot, label_shot, args.batchsz, args.finetuning_epoch, verbose=1)
            srcnn_model.save_weights(os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_SRCNN_weights.weights.h5"))
            print(f"DNCNN Fine-tuning for channel {channel_name}")                  
            srcnn_pred_train = srcnn_model.predict(data_shot)
            history = dncnn_model.fit(srcnn_pred_train, label_shot, args.batchsz, args.finetuning_epoch, verbose=1)
            plot_loss(history, f"DNCNN Fine-tuning Loss - {channel_name}",
                      os.path.join(args.save_init, f"DNCNN_{channel_name}_loss.png"))
            dncnn_model.save_weights(os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_DNCNN_.weights.h5"))
            
            eval_data , eval_label = data[30:] , label[30:]
            srcnn_pred_eval = srcnn_model.predict(eval_data)
            dncnn_pred = dncnn_model.predict(srcnn_pred_eval)
            np.save(os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_DNCNN_predictions.npy"), dncnn_pred)
            print(f"Predictions for channel {channel_name} saved.")
            
            

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='path to processed_data dir', default="new_data")
    argparser.add_argument('--device', type=str, help='device to run the process', default='2')
    argparser.add_argument('--save_init', type=str, help='path to save directory', default="saved_init")
    argparser.add_argument('--finetuning_epoch', type=int, help='epochs for fine_tuning', default=100)
    argparser.add_argument('--train_epoch', type=int, help='epoch number for fine-tuning', default=3000)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=8)
    argparser.add_argument('--n_way', type=int, help='n_task', default=5)  
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=15)
    argparser.add_argument('--train_lr', type=float, help='fine-tuning learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='fine-tuning learning rate', default=1e-3)
    argparser.add_argument('--mode', type=str, help='train or fine_tune', default="finetunng")
    
    
    
    
    args = argparser.parse_args()
    ChannelNet(args)
