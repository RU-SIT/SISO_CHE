import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from metrics import Metric
from utils import Utils
import pdb 
from sklearn.metrics import mean_squared_error
from metrics import extract_snr



def compute_mse_snr(preds, gt, snrs):
    """
    preds:  np.ndarray, shape (N, H, W, C)
    gt:     np.ndarray, same shape
    snrs:   list of SNR values
    returns: list of MSEs, one per snr
    """
    N = preds.shape[0]
    K = len(snrs)
    per = N // K
    mses = []
    for i, snr in enumerate(snrs):
        start = i * per
        end   = (i+1)*per if i < K-1 else N
        p = preds[start:end].reshape(-1)
        g = gt[start:end].reshape(-1)
        mses.append(mean_squared_error(g, p))
    return mses


def calculate_and_save_ber(metric, preds, snrs, modulation, out_path):
    bers = []
    for snr in snrs:
        bers.append(metric.bit_error_rate(preds, snr, modulation=modulation))
    np.save(out_path, bers)
    return bers

def plot_mse_snr_subplots(snrs, mse_dict, channel, out_dir, epoch, shots):
    """
    One figure with len(shots) subplots: each subplot is one shot,
    plotting LS, ChannelNet, iMAML, and MAML MSE vs. SNR.
    
    mse_dict keys should be:
      'LS', 'ChannelNet_{shot}', 'iMAML_{shot}', 'MAML_{shot}'
    """
    fig, axes = plt.subplots(1, len(shots), figsize=(5*len(shots), 4), sharey=True)
    for ax, shot in zip(axes, shots):
        ax.plot(snrs, mse_dict.get('LS', []),        marker='o', linestyle='-',  label='LS')
        for key, mkr, ln in [
            (f'ChannelNet_{shot}', 's', '-'),
            (f'iMAML_{shot}',      '^', '--'),
            (f'MAML_{shot}',       'd', '-.'),
        ]:
            arr = mse_dict.get(key)
            if arr is not None:
                ax.plot(snrs, arr, marker=mkr, linestyle=ln, label=key)
        ax.set_title(f"{shot}-shot")
        ax.set_xlabel("SNR (dB)")
        ax.grid(True)

    axes[0].set_ylabel("MSE")
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05,1))
    fig.suptitle(f"MSE vs SNR — {channel} (epoch {epoch})")
    plt.tight_layout(rect=[0,0,0.85,0.95])
    out_path = os.path.join(out_dir, f"MSE_SNR_{channel}_e{epoch}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved MSE plot: {out_path}")


    
def plot_ber_snr_subplots(snrs, ber_dict, channel, out_dir, epoch, shots):
    """
    One figure with 1×len(shots) subplots:
      each subplot is for one shot scenario,
      plotting LS, ChannelNet, iMAML, and MAML.
    """
    fig, axes = plt.subplots(1, len(shots), figsize=(5*len(shots), 4), sharey=True)

    for ax, shot in zip(axes, shots):
        # always plot LS
        ax.plot(snrs, ber_dict.get('LS', []),        marker='o', linestyle='-',  label='LS')
        # then each model if present
        for key, mkr, ls in [
            (f'ChannelNet_{shot}', 's', '-'),
            (f'iMAML_{shot}',      '^', '--'),
            (f'MAML_{shot}',       'd', '-.'),
        ]:
            vals = ber_dict.get(key)
            if vals is not None:
                ax.plot(snrs, vals, marker=mkr, linestyle=ls, label=key)

        ax.set_title(f"{shot}-shot")
        ax.set_xlabel("SNR (dB)")
        ax.grid(True)

    axes[0].set_ylabel("BER")
    # single legend on the right of the last subplot
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    fig.suptitle(f"BER vs SNR — {channel} (epoch {epoch})")
    plt.tight_layout(rect=[0,0,0.85,0.95])
    out_path = os.path.join(out_dir, f"BER_SNR_{channel}_e{epoch}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved BER plot: {out_path}")

def plot_preds_real_imag(gt, preds, shot, channel, out_dir, sample=0):
    """
    2 rows × 5 columns:
      row0 = Real; row1 = Imag.
    columns = GT, LS, ChannelNet, iMAML, MAML.
    """
    models = ['GT','LS','ChannelNet','iMAML','MAML']
    fig, axes = plt.subplots(2, len(models), figsize=(4*len(models), 8))

    # find vmin/vmax over all
    rmin,rmax = np.inf,-np.inf
    imin,imax = np.inf,-np.inf
    for m in models:
        arr = gt if m=='GT' else preds.get(m)
        if arr is None: continue
        samp = arr[sample]
        rmin,rmax = min(rmin,samp[...,0].min()), max(rmax,samp[...,0].max())
        imin,imax = min(imin,samp[...,1].min()), max(imax,samp[...,1].max())

    for j,m in enumerate(models):
        axr, axi = axes[0,j], axes[1,j]
        arr = gt if m=='GT' else preds.get(m)
        if arr is not None:
            samp = arr[sample]
            axr.imshow(samp[...,0], vmin=rmin, vmax=rmax, cmap='viridis', aspect='auto')
            axi.imshow(samp[...,1], vmin=imin, vmax=imax, cmap='plasma',  aspect='auto')
        axr.set_title(f"{m} — Real"); axr.axis('off')
        axi.set_title(f"{m} — Imag"); axi.axis('off')

    fig.colorbar(axes[0,-1].images[0], ax=axes[0,:].tolist(),
                 orientation='horizontal', fraction=0.05, pad=0.1)
    fig.colorbar(axes[1,-1].images[0], ax=axes[1,:].tolist(),
                 orientation='horizontal', fraction=0.05, pad=0.1)

    fig.suptitle(f"{channel} — {shot}-shot CSI (Real & Imag)")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out_dir, f"Preds_{channel}_{shot}shot.png"), dpi=300)
    plt.close()

def plot_abs_error(gt, preds, shot, channel, out_dir, sample_idx=0):
    """
    2×4 grid of absolute error |prediction − GT|:
      row 0 = real-error, row 1 = imag-error,
      cols = LS, ChannelNet, iMAML, MAML
    """
    import numpy as np
    import matplotlib.pyplot as plt
    models = ['LS','ChannelNet','iMAML','MAML']

    # helper to grab a (612,14,2) sample or None
    def get_sample(arr):
        if arr is None: 
            return None
        if arr.ndim == 4 and arr.shape[1:] == (612,14,2):
            return arr[sample_idx]
        if arr.ndim == 3 and arr.shape == (612,14,2):
            return arr
        return None  # skip anything else

    gt_s = get_sample(gt)
    if gt_s is None:
        raise ValueError("GT must have shape (N,612,14,2) or (612,14,2)")
    gr, gi = gt_s[...,0], gt_s[...,1]

    # find global error scale
    emin, emax = np.inf, -np.inf
    for m in models:
        s = get_sample(preds.get(m))
        if s is None: continue
        er = np.abs(s[...,0] - gr)
        ei = np.abs(s[...,1] - gi)
        emin, emax = min(emin, er.min(), ei.min()), max(emax, er.max(), ei.max())

    # plot
    fig, axes = plt.subplots(2, 4, figsize=(16,8))
    for j, m in enumerate(models):
        axr, axi = axes[0,j], axes[1,j]
        s = get_sample(preds.get(m))
        if s is not None:
            er = np.abs(s[...,0] - gr)
            ei = np.abs(s[...,1] - gi)
            imr = axr.imshow(er, vmin=emin, vmax=emax, cmap='inferno', aspect='auto')
            imi = axi.imshow(ei, vmin=emin, vmax=emax, cmap='inferno', aspect='auto')
        axr.set_title(f"{m} Real Err"); axr.axis('off')
        axi.set_title(f"{m} Imag Err"); axi.axis('off')

    # shared colorbars
    fig.colorbar(imr, ax=axes[0,:].tolist(), orientation='horizontal', fraction=0.05, pad=0.1)
    fig.colorbar(imi, ax=axes[1,:].tolist(), orientation='horizontal', fraction=0.05, pad=0.1)

    fig.suptitle(f"{channel} — {shot}-shot Absolute Error")
    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(out_dir, f"AbsErr_{channel}_{shot}shot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved AbsError plot: {out_path}")
    
def main():
    p = argparse.ArgumentParser()
    # p.add_argument('--root',      default="/path/to/new_data")
    # p.add_argument('--save_init', default="/path/to/saved_init")
    # p.add_argument('--imaml',     default="/path/to/Wireless_iMAML")
    p.add_argument('--root',      default="Sionna_datasets/ps2_p612/speed5/SISO-UMi")
    p.add_argument('--save_init', default="SISO_UMi_init")
    p.add_argument('--imaml',     default="SISO_UMi_init")
    p.add_argument('--n_way',     type=int, default=2)
    p.add_argument('--lam',       type=float, default=2.0)
    p.add_argument('--epoch',     type=int, default=500)
    args = p.parse_args()

    metric = Metric()
    snrs    = [-5,0,5,10,20,30]
    shots   = [5,10,15]

    D = np.load(os.path.join(args.root,'channel_data_dict.npy'),allow_pickle=True).item()
    L = np.load(os.path.join(args.root,'channel_label_dict.npy'),allow_pickle=True).item()
    # pdb.set_trace()
    for ch in list(D.keys())[4:]:
        # strip extension so output files are nicer
        ch_name = os.path.splitext(ch)[0]                             

        odir = os.path.join(args.root, "results", f"meta_model_nway_{args.n_way}", ch_name)
        os.makedirs(odir, exist_ok=True)

        LS_full = D[ch]; GT_full = L[ch]
        LS_eval = LS_full[30:]; GT_eval = GT_full[30:]

        # 1) BER‐SNR
        ber = {'LS': calculate_and_save_ber(metric, LS_eval, snrs, '16QAM', 
                                            os.path.join(odir,"LS_BER.npy"))}

        # load predictions
        preds_by_shot = {}
        for s in shots:
            # MAML
            p_maml = os.path.join(args.save_init,
                                  f"meta_model_nway_{args.n_way}",
                                  f"MAML_{s}_shot_{ch}_predictions.npy")
            maml = np.load(p_maml).transpose(0,2,3,1) if os.path.exists(p_maml) else None
            if maml is not None:
                ber[f'MAML_{s}'] = calculate_and_save_ber(metric, maml, snrs, '16QAM',
                                                         os.path.join(odir,f"MAML_{s}_BER.npy"))

            # iMAML   (FIXED: use args.imaml, not args.save_init)
            p_imaml = os.path.join(args.imaml,
                                   f"meta_model_nway_{args.n_way}",
                                #    f"wireless_IMAML_{s}_shot_{ch}_predictions.npy")  uncomment for TDL
            
            # pdb.set_trace()
                                   f"wireless_IMAML_{s}_shot_LAM{args.lam}_{ch}_predictions.npy")  # commnet fot TDL
            imaml = np.load(p_imaml).transpose(0,2,3,1) if os.path.exists(p_imaml) else None
            if imaml is not None:
                ber[f'iMAML_{s}'] = calculate_and_save_ber(metric, imaml, snrs, '16QAM',
                                                          os.path.join(odir,f"iMAML_{s}_BER.npy"))

            # ChannelNet
            p_ch = os.path.join(args.save_init, 
                                # f"meta_model_nway_{args.n_way}", uncomment for TDL
                                f"{s}shot_{ch}_DNCNN_predictions.npy")
            # pdb.set_trace()
            
            chnet = np.load(p_ch).transpose(0,2,3,1) if os.path.exists(p_ch) else None
            if chnet is not None:
                ber[f'ChannelNet_{s}'] = calculate_and_save_ber(metric, chnet, snrs, '16QAM',
                                                                os.path.join(odir,f"ChannelNet_{s}_BER.npy"))

            preds_by_shot[s] = {
                'LS': LS_eval, 
                'MAML': maml, 
                'iMAML': imaml, 
                'ChannelNet': chnet
            }

        # 0) MSE‐SNR  (build once, outside shot loop)
        mse_dict = {
            'LS': compute_mse_snr(LS_eval, GT_eval, snrs)
        }
        for s in shots:
            block = preds_by_shot[s]
            if block['MAML']    is not None: mse_dict[f'MAML_{s}']      = compute_mse_snr(block['MAML'],    GT_eval, snrs)
            if block['iMAML']   is not None: mse_dict[f'iMAML_{s}']     = compute_mse_snr(block['iMAML'],   GT_eval, snrs)
            if block['ChannelNet'] is not None: mse_dict[f'ChannelNet_{s}'] = compute_mse_snr(block['ChannelNet'], GT_eval, snrs)

        plot_mse_snr_subplots(snrs, mse_dict, ch_name, odir, args.epoch, shots)  

        # 1) BER‐SNR
        plot_ber_snr_subplots(snrs, ber, ch_name, odir, args.epoch, shots)

        # 2&3) Predictions & AbsError
        for s in shots:
            plot_preds_real_imag(GT_eval, preds_by_shot[s], s, ch_name, odir)
            plot_abs_error    (GT_eval, preds_by_shot[s], s, ch_name, odir)
            
if __name__=="__main__":
    main()

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--root',      default="Sionna_datasets/ps2_p612/speed5/SISO-UMi")
#     p.add_argument('--save_init', default="SISO_UMi_init")
#     p.add_argument('--imaml',     default="Wireless_iMAML")
#     p.add_argument('--n_way',     type=int, default=2)
#     p.add_argument('--lam',       type=float, default=2.0)
#     p.add_argument('--epoch',     type=int, default=500)
#     args = p.parse_args()

#     metric = Metric()
#     snrs    = [-5,0,5,10,20,30]
#     shots   = [5,10,15]

#     D = np.load(os.path.join(args.root,'channel_data_dict.npy'),allow_pickle=True).item()
#     L = np.load(os.path.join(args.root,'channel_label_dict.npy'),allow_pickle=True).item()
#     # We only need the fine tuning evaluation data for the channel, here ch is only the name the channel
#     for ch in list(D.keys())[4:]:
#         odir = os.path.join(args.root,"results",f"meta_model_nway_{args.n_way}",ch)
#         os.makedirs(odir, exist_ok=True)
#         pdb.set_trace()
#         # for each channel, load the evaluation LS and GT data where the index is 30:
#         LS_full = D[ch]; GT_full = L[ch]
#         LS_eval = LS_full[30:]; GT_eval = GT_full[30:] #shape: (226, 612, 14, 2)

#         ber = {}
#         ber['LS'] = calculate_and_save_ber(metric, LS_eval, snrs,'16QAM',os.path.join(odir,"LS_BER.npy"))

#         preds_by_shot = {}
#         for s in shots:
#             # MAML
#             path = os.path.join(args.save_init,
#                                 f"meta_model_nway_{args.n_way}",
#                                 f"MAML_{s}_shot_{ch}_predictions.npy")
#             maml = np.load(path).transpose(0,2,3,1) if os.path.exists(path) else None
#             # pdb.set_trace()
#             if maml is not None:
#                 ber[f'MAML_{s}'] = calculate_and_save_ber(metric, maml, snrs,'16QAM',
#                                                          os.path.join(odir,f"MAML_{s}_BER.npy"))
#             # iMAML
#             path = os.path.join(args.save_init,
#                                 f"meta_model_nway_{args.n_way}",
#                                 f"wireless_IMAML_{s}_shot_LAM{args.lam}_{ch}_predictions.npy")
#             imaml = np.load(path).transpose(0,2,3,1) if os.path.exists(path) else None
#             if imaml is not None:
#                 ber[f'iMAML_{s}'] = calculate_and_save_ber(metric, imaml, snrs,'16QAM',
#                                                           os.path.join(odir,f"iMAML_{s}_BER.npy"))
#             # ChannelNet
#             path = os.path.join(args.save_init,
#                                 f"{s}shot_{ch}_DNCNN_predictions.npy")
#             chnet = np.load(path).transpose(0,2,3,1) if os.path.exists(path) else None
#             if chnet is not None:
#                 ber[f'ChannelNet_{s}'] = calculate_and_save_ber(metric, chnet, snrs,'16QAM',
#                                                                 os.path.join(odir,f"ChannelNet_{s}_BER.npy"))

#             preds_by_shot[s] = {'LS':LS_eval, 'MAML':maml, 'iMAML':imaml, 'ChannelNet':chnet}
#         # 0) MSE-SNR
#         mse_dict = {}
#         mse_dict['LS'] = compute_mse_snr(LS_eval, GT_eval, snrs)
#         for s in shots:
#             block = preds_by_shot[s]
#             if block['MAML']    is not None: mse_dict[f'MAML_{s}']      = compute_mse_snr(block['MAML'],    GT_eval, snrs)
#             if block['iMAML']   is not None: mse_dict[f'iMAML_{s}']     = compute_mse_snr(block['iMAML'],   GT_eval, snrs)
#             if block['ChannelNet'] is not None: mse_dict[f'ChannelNet_{s}'] = compute_mse_snr(block['ChannelNet'], GT_eval, snrs)
        
#         plot_mse_snr_subplots(snrs, mse_dict, ch, odir, args.epoch, shots)
       
#         # 1) BER-SNR
#         plot_ber_snr_subplots(snrs, ber, ch, odir, args.epoch, shots)

#         # 2&3) Predictions & AbsError
#         for s in shots:
#             plot_preds_real_imag (GT_eval, preds_by_shot[s], s, ch, odir)
#             plot_abs_error     (GT_eval, preds_by_shot[s], s, ch, odir)
#             plot_mse_snr_subplots(snrs, mse_dict, ch, odir, args.epoch, shots)  
# if __name__=="__main__":
#     main()
