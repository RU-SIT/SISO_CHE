import os
import numpy as np
from models import SRCNN_model, DNCNN_model, visualize_and_save_results, DNCNN_predict
from ch_metrics import Metric
import matplotlib.pyplot as plt
from utils import Utils  # <-- use your new scaler here
import argparse
import pdb
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

EPS = 1e-8

# ---------- helpers for (re)using saved params ----------
def _save_minmax_params(save_dir, x_params, y_params):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "minmax_params.npz"),
        x_min_real=x_params["min_real"], x_max_real=x_params["max_real"],
        x_min_imag=x_params["min_imag"], x_max_imag=x_params["max_imag"],
        y_min_real=y_params["min_real"], y_max_real=y_params["max_real"],
        y_min_imag=y_params["min_imag"], y_max_imag=y_params["max_imag"],
    )

def _load_minmax_params(save_dir):
    p = np.load(os.path.join(save_dir, "minmax_params.npz"))
    x_params = {
        "min_real": p["x_min_real"], "max_real": p["x_max_real"],
        "min_imag": p["x_min_imag"], "max_imag": p["x_max_imag"],
    }
    y_params = {
        "min_real": p["y_min_real"], "max_real": p["y_max_real"],
        "min_imag": p["y_min_imag"], "max_imag": p["y_max_imag"],
    }
    return x_params, y_params

def _apply_scaling_with_params(x, params, eps=EPS):
    """Scale x to [-1,1] using *provided* per-channel params (no refit)."""
    real = 2.0 * (x[..., 0] - params["min_real"]) / (params["max_real"] - params["min_real"] + eps) - 1.0
    imag = 2.0 * (x[..., 1] - params["min_imag"]) / (params["max_imag"] - params["min_imag"] + eps) - 1.0
    return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)
# --------------------------------------------------------


def data_loader(root, n_way, mode, dataset_type):
    data = []
    label = []
    # Load the data and labels
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()
    file_names = list(data_dict.keys())

    # Split filenames into training and testing based on dataset type
    if dataset_type.lower() == 'tdl':
        n_train_files = 10
    elif dataset_type.lower() == 'umi':
        n_train_files = 4
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'tdl' or 'umi'")
    
    train_file_names = file_names[:n_train_files]
    finetune_file_names = file_names[n_train_files:]

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

    data_combined = np.concatenate(data, axis=0).astype(np.float32)
    label_combined = np.concatenate(label, axis=0).astype(np.float32)

    # --- NEW: min–max scaling per channel (real/imag), separately for X and Y ---
    data_scaled, x_params = Utils.standard_scaling(data_combined)      # [-1,1]
    label_scaled, y_params = Utils.standard_scaling(label_combined)    # [-1,1]

    return data_scaled, label_scaled, x_params, y_params


def channel_finder(root, n_way, dataset_type):
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()
    file_names = list(data_dict.keys())
    # Split filenames into training and testing based on dataset type
    if dataset_type.lower() == 'tdl':
        n_train_files = 10
    elif dataset_type.lower() == 'umi':
        n_train_files = 4
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'tdl' or 'umi'")
    
    finetune_file_names = file_names[n_train_files:]
    return finetune_file_names


def ft_data_loader(root, ch_name):
    data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()
    labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()
    data = data_dict[ch_name].astype(np.float32)
    label = labels_dict[ch_name].astype(np.float32)
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def ChannelNet(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    
    if args.mode == "train":
        train_data, train_label, x_params, y_params = data_loader(args.root, args.n_way, args.mode, args.dataset_type)
        print(f"Training data shape: {train_data.shape}")
        print(f"Training label shape: {train_label.shape}")
        print(f"Total number of training samples: {train_data.shape[0]}")
        _save_minmax_params(args.save_init, x_params, y_params)
        # pdb.set_trace()
        os.makedirs(args.save_init, exist_ok=True)

        # ---------- callbacks (shared) ----------
        early = EarlyStopping(
            monitor='val_loss', patience=20, min_delta=1e-5, restore_best_weights=True, verbose=1
        )
        reduce = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1
        )

        # ---------- SRCNN ----------
        srcnn_model = SRCNN_model(lr=args.train_lr)
        srcnn_params = srcnn_model.count_params()
        print(f"\n{'='*60}")
        print(f"SRCNN Model Trainable Parameters: {srcnn_params:,}")
        print(f"{'='*60}\n")
        srcnn_model.summary()
        ckpt_srcnn = ModelCheckpoint(
            os.path.join(args.save_init, "SRCNN_best.weights.h5"),
            monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1
        )
        
        srcnn_model.fit(
            train_data, train_label,
            batch_size=args.batchsz, epochs=args.train_epoch,
            validation_split=0.1, shuffle=True, verbose=1,
            callbacks=[early, reduce, ckpt_srcnn]
        )
        # also keep a “final” snapshot
        srcnn_model.save_weights(os.path.join(args.save_init, "SRCNN_trained.weights.h5"))

        # ---------- DNCNN ----------
        srcnn_pred_train = srcnn_model.predict(train_data, verbose=0)
        dncnn_model = DNCNN_model(lr=args.train_lr)
        dncnn_params = dncnn_model.count_params()
        print(f"\n{'='*60}")
        print(f"DNCNN Model Trainable Parameters: {dncnn_params:,}")
        print(f"{'='*60}\n")
        dncnn_model.summary()
        total_params = srcnn_params + dncnn_params
        print(f"\n{'='*60}")
        print(f"Total ChannelNet Trainable Parameters (SRCNN + DNCNN): {total_params:,}")
        print(f"{'='*60}\n")
        ckpt_dncnn = ModelCheckpoint(
            os.path.join(args.save_init, "DNCNN_best.weights.h5"),
            monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1
        )
        history = dncnn_model.fit(
            srcnn_pred_train, train_label,
            batch_size=args.batchsz, epochs=args.train_epoch,
            validation_split=0.1, shuffle=True, verbose=1,
            callbacks=[early, reduce, ckpt_dncnn]
        )
        plot_loss(history, "DNCNN Training Loss", os.path.join(args.save_init, "DNCNN_loss.png"))
        dncnn_model.save_weights(os.path.join(args.save_init, "DNCNN_trained.weights.h5"))

        print("Training complete. Models and min–max params saved.")

    else:
        finetune_file_names = channel_finder(args.root, args.n_way, args.dataset_type)
        x_params, y_params = _load_minmax_params(args.save_init)

        for channel_name in finetune_file_names:
            print(f"Starting fine-tuning for channel: {channel_name}")

            # Load base models
            srcnn_model = SRCNN_model(lr=args.train_lr)
            srcnn_model.load_weights(os.path.join(args.save_init, "SRCNN_best.weights.h5"))  # use best
            dncnn_model = DNCNN_model(lr=args.train_lr)
            dncnn_model.load_weights(os.path.join(args.save_init, "DNCNN_best.weights.h5"))

            # Prepare data
            data, label = ft_data_loader(args.root, channel_name)
            data_s = _apply_scaling_with_params(data, x_params)
            label_s = _apply_scaling_with_params(label, y_params)

            # k-shot subset
            FT_data, FT_label = data_s[:30], label_s[:30]
            data_shot, label_shot = FT_data[:args.k_qry], FT_label[:args.k_qry]

            # tiny val split (ensure >=2 samples for val; if not, do leave-one-out)
            n = len(data_shot)
            n_train = max(1, int(0.8 * n))
            x_tr, y_tr = data_shot[:n_train], label_shot[:n_train]
            x_va, y_va = data_shot[n_train:], label_shot[n_train:] if n > 1 else (data_shot, label_shot)

            # callbacks: shorter patience for fine-tune
            early_ft = EarlyStopping(
                monitor='val_loss', patience=6, min_delta=1e-6, restore_best_weights=True, verbose=1
            )
            reduce_ft = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
            )

            # --- fine-tune SRCNN ---
            ckpt_srcnn_ft = ModelCheckpoint(
                os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_SRCNN_best.weights.h5"),
                monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1
            )
            srcnn_model.fit(
                x_tr, y_tr,
                batch_size=args.batchsz, epochs=args.finetuning_epoch,
                validation_data=(x_va, y_va), shuffle=True, verbose=1,
                callbacks=[early_ft, reduce_ft, ckpt_srcnn_ft]
            )

            # --- fine-tune DNCNN on SRCNN outputs ---
            srcnn_pred_tr = srcnn_model.predict(x_tr, verbose=0)
            srcnn_pred_va = srcnn_model.predict(x_va, verbose=0)
            ckpt_dncnn_ft = ModelCheckpoint(
                os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_DNCNN_best.weights.h5"),
                monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1
            )
            
            history = dncnn_model.fit(
                srcnn_pred_tr, y_tr,
                batch_size=args.batchsz, epochs=args.finetuning_epoch,
                validation_data=(srcnn_pred_va, y_va), shuffle=True, verbose=1,
                callbacks=[early_ft, reduce_ft, ckpt_dncnn_ft]
            )
            plot_loss(history, f"DNCNN Fine-tuning Loss - {channel_name}",
                    os.path.join(args.save_init, f"DNCNN_{channel_name}_loss.png"))

            # ---------- Evaluation ----------
            eval_data_s, eval_label_s = data_s[30:], label_s[30:]
            if eval_data_s.size == 0:
                print(f"[{channel_name}] No eval split beyond k-shot. Skipping eval.")
                continue

            srcnn_pred_eval_s = srcnn_model.predict(eval_data_s, verbose=0)
            dncnn_pred_s = dncnn_model.predict(srcnn_pred_eval_s, verbose=0)

            dncnn_pred = Utils.unscale_standard(dncnn_pred_s, y_params)
            eval_label = Utils.unscale_standard(eval_label_s, y_params)

            os.makedirs(args.save_init, exist_ok=True)
            np.save(os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_DNCNN_predictions.npy"), dncnn_pred)
            np.save(os.path.join(args.save_init, f"{args.k_qry}shot_{channel_name}_eval_labels.npy"), eval_label)

            print(f"[{channel_name}] Predictions (unscaled) saved.")
            

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='path to processed_data dir', default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak")
    argparser.add_argument('--device', type=str, help='device to run the process', default="0")
    argparser.add_argument('--save_init', type=str, help='path to save directory', default="home/rghasemi/Wireless_communication/SISO_UMi_init/std_scaler_interpolated_noleak")
    argparser.add_argument('--finetuning_epoch', type=int, help='epochs for fine_tuning', default=25)
    argparser.add_argument('--train_epoch', type=int, help='epoch number for fine-tuning', default=15000)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=8)
    argparser.add_argument('--n_way', type=int, help='n_task', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--train_lr', type=float, help='fine-tuning learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='fine-tuning learning rate', default=1e-3)
    argparser.add_argument('--mode', type=str, help='train or fine_tune', default="train")
    argparser.add_argument('--dataset_type', type=str, help='dataset type: tdl or umi', default="umi", choices=['tdl', 'umi'])

    args = argparser.parse_args()
    ChannelNet(args)

