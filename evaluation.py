import os
import sys
import numpy as np
import torch
import pandas as pd
import pdb
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import matplotlib.pyplot as plt
from models import SRCNN_model, DNCNN_model, unit_scaling
import os, sys
sys.path.insert(0, "/home/CAMPUS/rghasemi/projects/MyPrivaterepo")
from imaml_dev.examples.learner_model import Learner, make_conv_network




def plot_best_worst_samples(
    out_s: np.ndarray,
    lab_s: np.ndarray,
    sample_mses: np.ndarray,
    model_name: str,
    snr: int,
    output_dir: str
):
    """
    For a single SNR & model:
      • out_s:      (N, subc, sym, 2) predicted
      • lab_s:      (N, subc, sym, 2) ground-truth
      • sample_mses:(N,) per-sample MSE
    Make a 2×(2*10) grid: for each of 5 best & 5 worst:
       - col 2*j   = GT
       - col 2*j+1 = Pred
    Top row real; bottom row imag.
    """
    os.makedirs(output_dir, exist_ok=True)

    # pick 5 best + 5 worst
    idx_sorted = np.argsort(sample_mses)
    sel = np.concatenate([idx_sorted[:5], idx_sorted[-5:]])

    n = len(sel)  # =10
    fig, axes = plt.subplots(
        2, 2*n,
        figsize=(2.0*(2*n), 4.0),
        sharex=True, sharey=True
    )

    for j, idx in enumerate(sel):
        # REAL part
        #  ── GT
        ax = axes[0, 2*j]
        im = ax.imshow(lab_s[idx, ..., 0].T, aspect='auto')
        ax.set_title(f"GT #{idx}\n{sample_mses[idx]:.2e}")
        ax.axis('off')
        #  ── Pred
        ax = axes[0, 2*j+1]
        ax.imshow(out_s[idx, ..., 0].T, aspect='auto')
        ax.set_title("Pred")
        ax.axis('off')

        # IMAG part
        #  ── GT
        ax = axes[1, 2*j]
        ax.imshow(lab_s[idx, ..., 1].T, aspect='auto')
        ax.axis('off')
        #  ── Pred
        ax = axes[1, 2*j+1]
        ax.imshow(out_s[idx, ..., 1].T, aspect='auto')
        ax.axis('off')

    # super‐title
    fig.suptitle(f"{model_name}  —  SNR={snr} dB", fontsize=16, y=1.03)
    fig.tight_layout()
    out_path = os.path.join(
        output_dir,
        f"{model_name.replace(' ', '_')}_best_worst_snr_{snr}dB.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Saved detailed plot: {out_path}")
    
    
class ChannelNetWrapper(torch.nn.Module):
    def __init__(self, srcnn, dncnn):
        super().__init__()
        self.srcnn = srcnn
        self.dncnn = dncnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, subc, sym, 2)
        inp_np = x.cpu().numpy()
        # force Keras to run on CPU to avoid GPU contention with PyTorch
        with tf.device("/CPU:0"):
            mid    = self.srcnn.predict(inp_np, verbose=0)
            out_np = self.dncnn.predict(mid,   verbose=0)
        out_t  = torch.from_numpy(out_np).float().to(x.device)
        return out_t
    
def load_channelnet(args, channel_name, k_shot):
    """
    Instantiate and load SRCNN + DNCNN for one channel_name and k_shot.

    args.save_init: where .weights.h5 files live
    channel_name, k_shot: used to form the filenames
    """
    # 1) SRCNN
    srcnn = SRCNN_model(lr=args.train_lr)
    srcnn_path = os.path.join(
        args.save_init,
        # f"meta_model_nway_{args.n_way}",
        f"{k_shot}shot_{channel_name}_SRCNN_weights.weights.h5"
    )
    print(f"Loading SRCNN weights from {srcnn_path}")
    srcnn.load_weights(srcnn_path)

    # 2) DNCNN
    dncnn = DNCNN_model(lr=args.train_lr)
    dncnn_path = os.path.join(
        args.save_init,
        # f"meta_model_nway_{args.n_way}",
        f"{k_shot}shot_{channel_name}_DNCNN_.weights.h5"
    )
    print(f"Loading DNCNN weights from {dncnn_path}")
    dncnn.load_weights(dncnn_path)

    return srcnn, dncnn



# ─── 1) helper to load three models (plus LS baseline) for a given channel + k‐shot ───
def load_models_for_channel_kshot(args, channel_name, k_shot, device="cpu"):
    """
    args must have:
      • save_init
      • save_dir
      • n_way
      • update_lr
      • lam
    channel_name: string exactly matching how you named weights files
    k_shot: 5, 10, or 15
    """
    models = {}
    # LS baseline
    models["LS"] = None

    #ChannelNet
    srcnn, dncnn = load_channelnet(args, channel_name, k_shot)
    wrapper = ChannelNetWrapper(srcnn, dncnn).to(device).eval()
    models["ChannelNet"] = wrapper
    
    
    # MAML
    from meta import Meta 
    config = [
        ('conv2d', [64, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [64]),
        ('conv2d', [256, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [512, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [512]),
        ('conv2d', [256, 512, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [256]),
        ('conv2d', [32, 256, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [32]),
        ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('avg_pool2d', [3, 1, 1]),
        ('bn', [args.batchsz]),
        ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
    ]
    maml_path = os.path.join(
        args.maml_dir,
        f"meta_model_nway_{args.n_way}",
        f"MAML_{k_shot}_shot_fine_tuned_model_{channel_name}_lr{args.update_lr}.pth"
    )
    # pdb.set_trace()
    maml = Meta(args, config).to(device).eval()
    ckpt = torch.load(maml_path, map_location=device)
    # if it’s a full checkpoint dict, pull out the real state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    maml.load_state_dict(sd)
    models["MAML"] = maml.net

    # iMAML
    learner_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=args.batchsz)
    meta_learner = Learner(model=learner_net,
                           loss_function=torch.nn.MSELoss(),
                           inner_lr=args.inner_lr,
                           outer_lr=args.outer_lr,
                           GPU=(args.device.startswith("cuda")))
    
    imaml_path = os.path.join(
        args.save_dir,
        f"meta_model_nway_{args.n_way}",
        # f"wireless_IMAML_{k_shot}_shot_fine_tuned_model_{channel_name}_lr{args.update_lr}.pth"
        f"wireless_IMAML_{k_shot}_shot_fine_tuned_model_{channel_name}_lr{args.update_lr}_LAM2.0.pth"
        
    )
    imaml = meta_learner.model.to(device).eval()
    ckpt_i = torch.load(imaml_path, map_location=device)
    if isinstance(ckpt_i, dict) and "state_dict" in ckpt_i:
        ckpt_i = ckpt_i["state_dict"]
    imaml.load_state_dict(ckpt_i)
    models["iMAML"] = imaml
    


    return models


# ─── 2) 2-D LS + freq‐then‐time interpolation helper ───
def interpolate_H_ls(RX, TX, pilot_syms, p_spacing):
    """
    RX,TX: complex np.arrays of shape (N, Ns, Nsc)
    """
    eps = 1e-8
    H_ls = RX / (TX + eps)
    N, Ns, Nsc = H_ls.shape
    pilot_subs = np.arange(0, Nsc, p_spacing)
    all_subs   = np.arange(Nsc)
    all_syms   = np.arange(Ns)
    H_hat = np.zeros_like(H_ls, dtype=complex)

    for n in range(N):
        # freq‐axis interp at each pilot symbol
        for sym in pilot_syms:
            Hp = H_ls[n, sym, pilot_subs]
            re = np.interp(all_subs, pilot_subs, Hp.real)
            im = np.interp(all_subs, pilot_subs, Hp.imag)
            H_hat[n, sym, :] = re + 1j*im
        # time‐axis interp at each subcarrier
        for k in range(Nsc):
            Hp = H_hat[n, pilot_syms, k]
            re = np.interp(all_syms, pilot_syms, Hp.real)
            im = np.interp(all_syms, pilot_syms, Hp.imag)
            H_hat[n, :, k] = re + 1j*im

    return H_hat


# ─── 3) single-k_shot evaluation ───
def finetunning_evaluation(
    args,
    root_dir: str,
    channel_type: str,
    pilot_syms: list,
    pspacing: int,
    snr_dbs: list,
    k_shot: int,
    slice_start: int = 30,
    csv_path: str = None,
    device: str = "cpu"
):
    """
    Runs LS, ChannelNet, MAML, iMAML on the last 226 samples (256-30).
    Returns avg MSE per model (list aligned with snr_dbs).
    """
    # load the two dicts
    tx_file = os.path.expanduser(os.path.join(root_dir, "tx_signal_dict.npy"))
    h_file  = os.path.expanduser(os.path.join(root_dir, "channel_label_dict.npy"))
    tx_dict = np.load(tx_file, allow_pickle=True).item()
    h_dict  = np.load(h_file, allow_pickle=True).item()
    # pdb.set_trace()
    # pick keys
    keys = list(tx_dict.keys())
    if channel_type.lower() == "tdl":
        keys = keys[10:]
    elif channel_type.lower() == "umi":
        keys = [keys[-1]]
    else:
        raise ValueError("channel_type must be 'TDL' or 'umi'")

    # raw storage: model_name → list of [per‐SNR MSE] for each channel
    raw = {m: [] for m in ["LS","ChannelNet","MAML","iMAML"]}

    for ch_name in keys:
        # load models for THIS channel + k_shot
        models = load_models_for_channel_kshot(args, ch_name, k_shot, device)

        # grab TX & H arrays
        TX = tx_dict[ch_name]   # shape ((15454, 1))
        H_raw = h_dict[ch_name]              # shape (256, 612, 14, 2)
        Hc = H_raw[...,0] + 1j*H_raw[...,1]  # now (256, 612, 14)
        # pdb.set_trace()

        # # 1) define OFDM parameters
        # fft_size = 612        # number of subcarriers
        # cp_len   = 128        # cyclic prefix length 
        # nsym     = 14         # number of OFDM symbols per frame

        # # 2) chop the waveform into NSYM blocks of (FFT+CP)
        # tx = TX.flatten()[: nsym*(fft_size+cp_len)]        # drop any extra tail
        # X   = tx.reshape(nsym, fft_size+cp_len)            # (14, 740)
        # X_nocp = X[:, cp_len:]                             # strip CP → (14,612)

        # # 3) FFT along each row to get freq-domain symbols
        # Xf = np.fft.fft(X_nocp, axis=-1)   # → shape (14,612)

        # # 4) reorder to match H’s layout: (batch, subc, sym)
        # #    here batch=1 if we only had one waveform, or tile for 256 realizations
        # TXc = Xf.T[None,...]               # → (1,612,14)
        # if Hc.shape[0] > 1:
        #     TXc = np.tile(TXc, (Hc.shape[0],1,1))  # → (256,612,14)
        
        # 5) now apply the channel
        
        TXc = TX[...,0] + 1j*TX[...,1]  # pack real & imag into a complex array
        RXc = TXc * Hc  

        # build all LS‐interp estimates
        H_ls_list = []
        for snr in snr_dbs:
            lin = 10**(snr/10)
            N0  = 1/np.sqrt(lin)
            noise = N0*(np.random.randn(*RXc.shape) + 1j*np.random.randn(*RXc.shape))
            H_hat = interpolate_H_ls(RXc + noise, TXc, pilot_syms, pspacing)
            H_ls_list.append(H_hat)

        # get LS baseline MSEs
        raw["LS"].append([ np.mean(np.abs(Hh - Hc)**2) for Hh in H_ls_list ])

        detailed_dir = f"detailed_plots/{channel_type}_{k_shot}shot"
        for name, mdl in models.items():
            if name == "LS":
                continue

            out_s_list        = []
            lab_s_list        = []
            sample_mses_list  = []
            avg_mses          = []

            mdl.eval()
            for snr_idx, Hh in enumerate(H_ls_list):
                # stack real & imag
                inp    = np.stack([Hh.real, Hh.imag], axis=-1)
                labels = np.stack([Hc.real, Hc.imag], axis=-1)
                pdb.set_trace()
                inp_s, lab_s, _ = unit_scaling(inp, labels)

                # run model
                if name == "ChannelNet":
                    t    = torch.from_numpy(inp_s).float().to(device)
                    with torch.no_grad():
                        out_s = mdl(t).cpu().numpy()
                else:
                    t      = torch.from_numpy(inp_s).float().permute(0,3,1,2).to(device)
                    with torch.no_grad():
                        out_pf = mdl(t)
                    out_s  = out_pf.permute(0,2,3,1).cpu().numpy()

                # per-sample MSE over spatial dims
                diff            = out_s - lab_s
                sample_mses     = np.mean(diff[...,0]**2 + diff[...,1]**2, axis=(1,2))
                avg_mse_this_snr = sample_mses.mean()

                # store for plots + aggregate
                out_s_list.append(out_s)
                lab_s_list.append(lab_s)
                sample_mses_list.append(sample_mses)
                avg_mses.append(avg_mse_this_snr)

            # save the averaged MSEs just as before
            raw[name].append(avg_mses)

            # now make the 5–best & 5–worst plot for each snr
            for snr_idx, snr in enumerate(snr_dbs):
                plot_best_worst_samples(
                    out_s_list[snr_idx],
                    lab_s_list[snr_idx],
                    sample_mses_list[snr_idx],
                    model_name=f"{name}_{k_shot}shot",
                    snr=snr,
                    output_dir=detailed_dir
                )
            
    # pdb.set_trace()        
    # average over channels
    avg = { name: np.stack(raw[name]).mean(axis=0).tolist()
            for name in raw }

    # optional CSV
    if csv_path:
        df = pd.DataFrame(avg, index=snr_dbs)
        df.index.name = "SNR_dB"
        long = df.reset_index().melt(
            id_vars="SNR_dB", var_name="Model", value_name="MSE"
        )
        long.to_csv(csv_path, index=False)

    return avg

def plot_estimate_grid(preds, truth, model_names, snr, out_dir="plots_estimates"):
    """
    preds:   list of np.array shaped (N,612,14,2) for each model
    truth:   np.array shaped (N,612,14,2)
    model_names: list of strings, same order as preds
    snr:     int or float, used in title / filename
    """
    os.makedirs(out_dir, exist_ok=True)
    sample_idx = 0  # choose the first realization

    n_models = len(preds)
    fig, axes = plt.subplots(
        2, n_models+1,
        figsize=(3*(n_models+1), 6),
        sharex=True, sharey=True
    )

    # for each model, plot real (row0) and imag (row1)
    for i, (pred, name) in enumerate(zip(preds, model_names)):
        real = pred[sample_idx,...,0].T  # transpose so subc runs horizontally
        imag = pred[sample_idx,...,1].T
        axes[0,i].imshow(real, aspect="auto", origin="lower")
        axes[0,i].set_title(name)
        axes[1,i].imshow(imag, aspect="auto", origin="lower")
        if i == 0:
            axes[0,i].set_ylabel("Real")
            axes[1,i].set_ylabel("Imag")

    # last column: ground truth
    gt_real = truth[sample_idx,...,0].T
    gt_imag = truth[sample_idx,...,1].T
    axes[0,-1].imshow(gt_real, aspect="auto", origin="lower")
    axes[0,-1].set_title("Ground Truth")
    axes[1,-1].imshow(gt_imag, aspect="auto", origin="lower")

    fig.suptitle(f"Channel Estimate @ SNR={snr} dB (sample #{sample_idx})", y=0.95)
    fig.tight_layout(rect=[0,0,1,0.93])

    out_path = os.path.join(out_dir, f"estimates_snr_{snr}dB.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved estimates heat‐maps: {out_path}")

# ─── 4) loop over K‐shots and make a 3-panel plot ───

def run_all_kshots(
    args,
    root_dir: str,
    channel_type: str,
    pilot_syms: list,
    pspacing: int,
    snr_dbs: list,
    k_shot_list: list = [5,10,15],
    slice_start: int = 30,
    device: str = "cpu"
):
    all_avgs = {}
    # 1) run the evaluations and save per-kshot CSV
    for k in k_shot_list:
        csv = f"{channel_type}_{k}shot_results.csv"
        avg = finetunning_evaluation(
            args, root_dir, channel_type,
            pilot_syms, pspacing, snr_dbs,
            k, slice_start,
            csv_path=csv,
            device=device
        )
        all_avgs[k] = avg

    # 2) now make grouped‐bar plots **for each SNR** across k‑shots
    out_dir = "plots_by_shot"
    os.makedirs(out_dir, exist_ok=True)

    # list of methods in a stable order
    methods = list(all_avgs[k_shot_list[0]].keys())

    for idx, snr in enumerate(snr_dbs):
        # build a DataFrame where
        #   • index = ["5-shot","10-shot",…]
        #   • columns = ["LS","ChannelNet","MAML","iMAML"]
        #   • values = corresponding avg MSE at this snr
        data = {
            m: [ all_avgs[k][m][idx] for k in k_shot_list ]
            for m in methods
        }
        df = pd.DataFrame(
            data,
            index=[ f"{k}-shot" for k in k_shot_list ]
        )

        # plot
        fig, ax = plt.subplots(figsize=(6,4))
        df.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f"MSE by Method over {k_shot_list}‑shot Experiments\nat SNR = {snr} dB")
        ax.set_xlabel("k‑shot")
        ax.set_ylabel("MSE")
        ax.legend(title="Method")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        fig.tight_layout()
        save_path = os.path.join(out_dir, f"mse_by_shot_snr_{snr}dB.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✔ Saved bar plot for SNR={snr} dB → {save_path}")
        
        example_k = k_shot_list[-1]
        # reload dictionaries so we can grab a single channel’s data
        tx_file = os.path.expanduser(os.path.join(root_dir, "tx_signal_dict.npy"))
        h_file  = os.path.expanduser(os.path.join(root_dir, "channel_label_dict.npy"))
        tx_dict = np.load(tx_file, allow_pickle=True).item()
        h_dict  = np.load(h_file, allow_pickle=True).item()

        # pick the same key logic we use in finetunning_evaluation:
        keys = list(tx_dict.keys())
        if channel_type.lower()=="tdl":
            keys = keys[10:]
        else:  # "umi"
            keys = [keys[-1]]
        channel_name = keys[0]

        # load that channel’s models
        models = load_models_for_channel_kshot(args, channel_name, example_k, device)
        model_names = [m for m in models if m!="LS"] + ["LS"]  

        # build the clean Hc for that channel
        H_raw = h_dict[channel_name]
        Hc = H_raw[...,0] + 1j*H_raw[...,1]           # (256,612,14)
        truth = np.stack([Hc.real, Hc.imag], axis=-1) # (256,612,14,2)

        # for each SNR: build LS‐interp, then get each model’s out_s array
        for snr in snr_dbs:
            lin = 10**(snr/10)
            N0 = 1/np.sqrt(lin)
            TX = tx_dict[channel_name]
            TXc = TX[...,0] + 1j*TX[...,1]
            RXc = TXc * Hc
            noise = N0*(np.random.randn(*RXc.shape) + 1j*np.random.randn(*RXc.shape))
            Hh = interpolate_H_ls(RXc+noise, TXc, pilot_syms, pspacing)

            # prepare ground‐truth scaled
            inp  = np.stack([Hh.real, Hh.imag], axis=-1)
            lab  = truth
            inp_s, lab_s, _ = unit_scaling(inp, lab)

            preds = []
            for name in model_names:
                if name=="LS":
                    pred_s = inp_s.copy()  # LS = just the pilots‑interp result
                else:
                    mdl = models[name]
                    t = torch.from_numpy(inp_s).float().to(device)
                    if name!="ChannelNet":
                        t = t.permute(0,3,1,2)  # to (batch,2,612,14)
                    with torch.no_grad():
                        out = mdl(t)
                    out_np = out.cpu().numpy()
                    if name!="ChannelNet":
                        out_np = np.transpose(out_np, (0,2,3,1))
                    pred_s = out_np
                
                # but here we just compare the scaled values:
                preds.append(pred_s)

            # now call our grid‐plotter
            plot_estimate_grid(
                preds,      # list of (256,612,14,2)
                lab_s,      # (256,612,14,2)
                model_names,
                snr
            )

        return all_avgs




# ─── 5) main ───
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_init", default= "SISO_UMi_init")
    # parser.add_argument("--maml_dir", default= "TDL_FinalMAML_init")
    parser.add_argument("--maml_dir", default= "SISO_UMi_init")
    
    
    # parser.add_argument("--save_dir", default= "Wireless_iMAML" , help= "directory to the imaml check point")
    parser.add_argument("--save_dir", default= "SISO_UMi_init" , help= "directory to the imaml check point")
    
    parser.add_argument("--n_way",     type=int, default=2) # for umi = 2, for tdl = 5
    parser.add_argument("--update_lr", type=float, default=0.001)
    parser.add_argument("--lam",       type=float, default=1e-3)
    
    #####CHNET########
    parser.add_argument('--train_lr', type=float, help='fine-tuning learning rate', default=1e-4)
    
    #####MAML_ags######
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batchsz', type=int, default=8)
    parser.add_argument('--update_step', type=int, default=2)
    parser.add_argument('--meta_lr', type=float, default=1e-4)
    parser.add_argument('--k_spt', type=int, default=15) 
    parser.add_argument('--k_qry', type=int, default=15)
    
    
    
    ####iMAML_args###############
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--outer_lr", type=float, default=1e-4)

    
    args = parser.parse_args()

    root = "Sionna_datasets/ps2_p612/speed5/SISO-UMi"
    # root = "./new_data"
    snrs = list(range(0,31,5))
    pilot_syms = [3,10]
    pspacing   = 4

    run_all_kshots(
      args,
      root_dir=root,
      channel_type="umi",
      pilot_syms=pilot_syms,
      pspacing=pspacing,
      snr_dbs=snrs,
      k_shot_list=[5,10,15],
      slice_start=30,
      device="cuda:1"
    )
