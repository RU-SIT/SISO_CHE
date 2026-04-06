import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pdb

# Optional TensorFlow import (only needed for ChannelNet)
try:
    import tensorflow as tf
    # Force TF on CPU so it doesn't fight PyTorch
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# Local imports
from utils import Utils

# Optional ChannelNet models import
try:
    from models import SRCNN_model, DNCNN_model
    CHANNELNET_AVAILABLE = True
except ImportError:
    CHANNELNET_AVAILABLE = False
    SRCNN_model = None
    DNCNN_model = None

sys.path.insert(0, "Wireless_communication")
from imaml_dev.examples.learner_model import Learner, make_conv_network

# Add multigrade_maml directory to path to import multigrade MAML
script_dir = os.path.dirname(os.path.abspath(__file__))
multigrade_maml_dir = os.path.join(script_dir, "multigrade_maml")
if os.path.exists(multigrade_maml_dir):
    sys.path.insert(0, multigrade_maml_dir)
else:
    # Fallback: try parent directory
    parent_dir = os.path.dirname(script_dir)
    multigrade_maml_dir = os.path.join(parent_dir, "multigrade_maml")
    if os.path.exists(multigrade_maml_dir):
        sys.path.insert(0, multigrade_maml_dir)

from multigrade_maml_stair import MultigradeMAMLStair

EPS = 1e-8
plt.switch_backend("Agg")


# -----------------------------
# Helpers: data, scaling, shapes
# -----------------------------
def apply_scaling_with_params(x, params):
    """Scale to [-1,1] using provided per-channel params; x shape (..., 2)."""
    real = 2.0 * (x[..., 0] - params["min_real"]) / (params["max_real"] - params["min_real"] + EPS) - 1.0
    imag = 2.0 * (x[..., 1] - params["min_imag"]) / (params["max_imag"] - params["min_imag"] + EPS) - 1.0
    out = np.stack([real, imag], axis=-1)
    return out.astype(np.float32, copy=False)


def to_torch_cf(x_cl, device):
    """[N,H,W,2] -> torch [N,2,H,W] float."""
    return torch.from_numpy(np.transpose(x_cl, (0, 3, 1, 2))).float().to(device)


def make_TXc_grid(TX, Hc, channel_type: str, fft_size, cp_len, tdl_cp_len=None):
    """
    Build the OFDM resource-grid TX to align with Hc.
    Handles both time-domain (TDL) and frequency-domain (UMi) inputs.
    """
    channel_type_l = channel_type.lower()
    if channel_type_l not in ("tdl", "umi"):
        raise ValueError("channel_type must be 'tdl' or 'umi'")
    
    if channel_type.lower() == "tdl" and tdl_cp_len is not None:
        cp_len = tdl_cp_len
        
    TX = np.asarray(TX)
    N, Nsc, Ns = Hc.shape

    def _ri_to_complex(a):
        return (a[..., 0] + 1j * a[..., 1]).astype(np.complex64)

    # ---------- Resource-grid inputs ----------
    if TX.ndim == 4 and TX.shape[-1] == 2 and TX.shape[1] == Nsc and TX.shape[2] == Ns:
        grid = _ri_to_complex(TX)  # (N, Nsc, Ns)
        return grid[0] if channel_type_l == "tdl" else grid

    if TX.ndim == 3 and TX.shape[1] == Nsc and TX.shape[2] == Ns and np.iscomplexobj(TX):
        grid = TX.astype(np.complex64)
        return grid[0] if channel_type_l == "tdl" else grid

    if TX.ndim == 3 and TX.shape[0] == Nsc and TX.shape[1] == Ns and TX.shape[2] == 2:
        grid = _ri_to_complex(TX)  # (Nsc, Ns)
        return grid if channel_type_l == "tdl" else np.tile(grid[np.newaxis, ...], (N, 1, 1)).astype(np.complex64)

    if TX.ndim == 2 and TX.shape[0] == Nsc and TX.shape[1] == Ns:
        grid = TX.astype(np.complex64)
        return grid if channel_type_l == "tdl" else np.tile(grid[np.newaxis, ...], (N, 1, 1)).astype(np.complex64)

    # ---------- Time-domain waveform inputs (for TDL) ----------
    tx_wav = TX.squeeze()
    if tx_wav.ndim == 2 and tx_wav.shape[1] > 1:
        tx_wav = tx_wav[:, 0]
    tx_wav = np.asarray(tx_wav).astype(np.complex64)

    total_per_sym = fft_size + cp_len
    needed = Ns * total_per_sym
    if tx_wav.size < needed:
        raise ValueError(f"TX waveform too short: need {needed} samples for Ns={Ns}, got {tx_wav.size}")

    tx_wav = tx_wav[:needed]
    X = tx_wav.reshape(Ns, total_per_sym)   # (Ns, fft+cp)
    X_nocp = X[:, cp_len:]                  # (Ns, fft)
    Xf = np.fft.fft(X_nocp, axis=-1)        # (Ns, Nsc)
    grid = Xf.T.astype(np.complex64)        # (Nsc, Ns)

    return grid if channel_type_l == "tdl" else np.tile(grid[np.newaxis, ...], (N, 1, 1)).astype(np.complex64)


def make_RXc_from_time_domain(RX_time, Hc, channel_type: str, fft_size, cp_len):
    """
    Convert time-domain RX signal to frequency-domain resource grid for TDL.
    RX_time: time-domain received signal (N, time_samples) or (time_samples,)
    Returns: RXc in frequency domain (N, Nsc, Ns) complex
    """
    if channel_type.lower() != "tdl":
        raise ValueError("This function is only for TDL time-domain signals")
    
    RX_time = np.asarray(RX_time)
    if RX_time.ndim == 1:
        RX_time = RX_time[np.newaxis, ...]  # Add batch dimension
    
    N, Nsc, Ns = Hc.shape
    total_per_sym = fft_size + cp_len
    needed = Ns * total_per_sym
    
    # Ensure we have enough samples
    rx_samples = RX_time.shape[-1]
    if rx_samples < needed:
        raise ValueError(f"RX waveform too short: need {needed} samples for Ns={Ns}, got {rx_samples}")
    
    RX_time = RX_time[..., :needed]
    
    # Reshape to (N, Ns, total_per_sym)
    RX_reshaped = RX_time.reshape(-1, Ns, total_per_sym)
    
    # Remove CP: (N, Ns, fft_size)
    RX_nocp = RX_reshaped[:, :, cp_len:]
    
    # FFT to frequency domain: (N, Ns, fft_size)
    RX_freq = np.fft.fft(RX_nocp, axis=-1)
    
    # Transpose to (N, Nsc, Ns)
    RXc = np.transpose(RX_freq, (0, 2, 1)).astype(np.complex64)
    
    return RXc


def robust_ls(RX, TX, eps=1e-6):
    """
    RX, TX: complex arrays (any broadcastable shape)
    Returns: H_ls with same shape as RX.
    """
    num = RX * np.conj(TX)
    den = (TX.real*TX.real + TX.imag*TX.imag) + eps
    return num / den


def interpolate_H_ls(RX, TX, pilot_syms, p_spacing):
    """
    RX, TX: complex arrays shape (N, Ns, Nsc)
    2D linear interpolation: freq @ pilot_syms, then time over all symbols.
    """
    H_ls = robust_ls(RX, TX, eps=1e-6)
    N, Ns, Nsc = H_ls.shape
    pilot_subs = np.arange(0, Nsc, p_spacing)
    all_subs   = np.arange(Nsc)
    all_syms   = np.arange(Ns)
    H_hat = np.zeros_like(H_ls, dtype=complex)

    for n in range(N):
        for sym in pilot_syms:
            Hp = H_ls[n, sym, pilot_subs]
            re = np.interp(all_subs, pilot_subs, Hp.real)
            im = np.interp(all_subs, pilot_subs, Hp.imag)
            H_hat[n, sym, :] = re + 1j * im
        for k in range(Nsc):
            Hp = H_hat[n, pilot_syms, k]
            re = np.interp(all_syms, pilot_syms, Hp.real)
            im = np.interp(all_syms, pilot_syms, Hp.imag)
            H_hat[n, :, k] = re + 1j * im
    return H_hat


def lmmse_interpolation_np(
    H_clean: np.ndarray, H_ls: np.ndarray, snr_db: float,
    pilot_syms: list, p_spacing: int, ridge: float = 1e-6
) -> np.ndarray:
    """
    Pilot-domain LMMSE at the pilot symbols only (frequency vectors).
    Returns: H_hat_pilots of shape (N, K, Nsc)
    """
    assert H_clean.shape == H_ls.shape
    N, Nsc, Ns = H_clean.shape
    pilot_syms = list(pilot_syms)
    K = len(pilot_syms)
    pilot_subs = _pilot_subcarrier_indices(Nsc, p_spacing)
    P = pilot_subs.size
    H_hat_pilots = np.zeros((N, K, Nsc), dtype=np.complex64)
    sigma2 = np.float32(10.0 ** (-snr_db / 10.0))  #Convert SNR(dB) to noise variance sigma2 = 1/SNR_linear.
    eyeP = np.eye(P, dtype=np.complex64) #dentity

    for n in range(N):
        for k, sym in enumerate(pilot_syms):
            hp = H_clean[n, pilot_subs, sym].astype(np.complex64) #hp : true channel at pilot subcarriers only (length P).
            hi = H_clean[n, :,          sym].astype(np.complex64) #hi : true channel at all subcarriers (length Nsc).
            hls_p = H_ls[n, pilot_subs, sym].astype(np.complex64) #hls_p: LS estimate at pilot subcarriers (noisy measurements you trust partially).
            #Covariances from ground truth
            Rhp  = np.outer(hp, np.conj(hp))  #Rhp = E[h_p h_p^H] approximated by hp hp^H
            Rhhp = np.outer(hi, np.conj(hp))  #Rhhp = E[h_all h_p^H] approximated by hi hp^H.
            # h_all_hat = R_{h h_p} @ (R_{h_p h_p} + sigma^2 I_P)^(-1) @ y_p
            A = Rhp + (sigma2 + ridge) * eyeP  # P×P , ridge is a tiny stabilizer to avoid numerical issues.
            x = np.linalg.solve(A, hls_p)      # length P
            H_hat_pilots[n, k, :] = Rhhp @ x    # (Nsc×P) @ (P) = Nsc
    return H_hat_pilots

def lmmse_baseline_np(
    H_clean: np.ndarray, H_ls: np.ndarray, snr_db: float,
    pilot_syms: list, p_spacing: int
) -> np.ndarray:
    """
    Full LMMSE baseline: pilot-domain LMMSE at pilot symbols + linear interpolation.
    Returns H_hat of shape (N, Nsc, Ns)
    """
    N, Nsc, Ns = H_clean.shape
    H_hat = np.zeros_like(H_ls, dtype=np.complex64)
    H_hat_pilots = lmmse_interpolation_np(H_clean, H_ls, snr_db, pilot_syms, p_spacing)
    all_syms = np.arange(Ns, dtype=int)
    pilot_syms = np.array(pilot_syms, dtype=int)
    for n in range(N):
        for sc in range(Nsc):
            vals = H_hat_pilots[n, :, sc]
            re = np.interp(all_syms, pilot_syms, vals.real)
            im = np.interp(all_syms, pilot_syms, vals.imag)
            H_hat[n, sc, :] = re + 1j * im
    return H_hat.astype(np.complex64)

def _pilot_subcarrier_indices(Nsc: int, p_spacing: int) -> np.ndarray:
    if p_spacing is None or p_spacing <= 0:
        raise ValueError("p_spacing must be a positive integer.")
    return np.arange(0, Nsc, int(p_spacing), dtype=int)


def _ls_only_at_pilots(RXc: np.ndarray,
                       TXc: np.ndarray,
                       pilot_syms: list,
                       p_spacing: int) -> np.ndarray:
    """
    LS ONLY at pilot REs, then leave other REs empty (filled later by interpolation).
    Shapes:
      RXc : (N, Nsc, Ns) complex
      TXc : (N, Nsc, Ns) or (Nsc, Ns) complex
    Returns:
      H_sparse : (N, Nsc, Ns) complex, nonzero only at pilot REs.
    """
    N, Nsc, Ns = RXc.shape

    # --- FIX: ensure numpy array before any comparisons
    pilot_syms = np.asarray(pilot_syms, dtype=int)
    pilot_syms = pilot_syms[(pilot_syms >= 0) & (pilot_syms < Ns)]
    if pilot_syms.size == 0:
        raise ValueError("pilot_syms empty after clipping to [0, Ns).")

    # --- Choose pilot subcarriers only from active carriers (skip guards/DC)
    if TXc.ndim == 3:      # (N, Nsc, Ns)
        active_mask = np.any(np.abs(TXc) > 0, axis=(0, 2))   # (Nsc,)
    elif TXc.ndim == 2:    # (Nsc, Ns)
        active_mask = np.any(np.abs(TXc) > 0, axis=1)        # (Nsc,)
    else:
        raise ValueError("TXc must be (N,Nsc,Ns) or (Nsc,Ns).")

    active_subs = np.flatnonzero(active_mask)
    if active_subs.size < 2:
        raise ValueError("Not enough active subcarriers in TXc.")

    pilot_subs = active_subs[::int(p_spacing)]
    if pilot_subs.size < 2:  # ensure we have at least two pilots for interp
        step = max(1, active_subs.size // 8)
        pilot_subs = active_subs[::step]

    # --- LS on pilots
    H_sparse = np.zeros((N, Nsc, Ns), dtype=np.complex64)
    for sym in pilot_syms:
        y_p = RXc[:, pilot_subs, sym]  # (N, P)
        if TXc.ndim == 3:
            x_p = TXc[:, pilot_subs, sym]  # (N, P)
        else:
            x_p = TXc[pilot_subs, sym]     # (P,)
        H_p = robust_ls(y_p, x_p, eps=1e-6)    # (N, P)
        H_sparse[:, pilot_subs, sym] = H_p.astype(np.complex64, copy=False)

    return H_sparse


def _interp_from_pilots(H_sparse: np.ndarray,
                        pilot_syms: list,
                        p_spacing: int) -> np.ndarray:
    """
    Fill non-pilot REs by linear interpolation (freq -> time) from pilot LS.
    Input/Output:
      (N, Nsc, Ns) complex
    """
    N, Nsc, Ns = H_sparse.shape
    pilot_syms = np.asarray(pilot_syms, dtype=int)
    pilot_subs = _pilot_subcarrier_indices(Nsc, p_spacing)
    if pilot_subs.size < 2:
        raise ValueError("Need >=2 pilot subcarriers for interpolation.")

    H_hat = np.zeros_like(H_sparse, dtype=np.complex64)

    # 1) Frequency interpolation at each pilot symbol
    all_subs = np.arange(Nsc, dtype=int)
    for n in range(N):
        for sym in pilot_syms:
            Hp = H_sparse[n, pilot_subs, sym]        # (P,)
            re = np.interp(all_subs, pilot_subs, Hp.real)
            im = np.interp(all_subs, pilot_subs, Hp.imag)
            H_hat[n, :, sym] = re + 1j * im

    # 2) Time interpolation at each subcarrier using pilot symbols
    all_syms = np.arange(Ns, dtype=int)
    for n in range(N):
        for sc in range(Nsc):
            vals = H_hat[n, sc, pilot_syms]          # (K,)
            re = np.interp(all_syms, pilot_syms, vals.real)
            im = np.interp(all_syms, pilot_syms, vals.imag)
            H_hat[n, sc, :] = re + 1j * im

    return H_hat

# -----------------------------
# Models
# -----------------------------
class ChannelNetWrapper(torch.nn.Module):
    """Bridges Keras SRCNN + DNCNN with PyTorch inference API."""
    def __init__(self, srcnn, dncnn):
        super().__init__()
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for ChannelNetWrapper")
        self.srcnn = srcnn
        self.dncnn = dncnn

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (N, H, W, 2) on CPU/GPU; we'll hop to CPU->numpy for TF predict
        inp_np = x_t.detach().cpu().numpy()
        with tf.device("/CPU:0"):
            mid    = self.srcnn.predict(inp_np, verbose=0)
            out_np = self.dncnn.predict(mid,   verbose=0)
        return torch.from_numpy(out_np).float().to(x_t.device)


def load_channelnet(args, channel_name, k_shot):
    """Load SRCNN + DNCNN weights for a specific channel + K-shot."""
    if not TF_AVAILABLE or not CHANNELNET_AVAILABLE:
        raise ImportError("TensorFlow and ChannelNet models are required for ChannelNet. Please install TensorFlow or disable ChannelNet.")
    
    srcnn = SRCNN_model(lr=args.train_lr)
    dncnn = DNCNN_model(lr=args.train_lr)

    srcnn_path = os.path.join(args.save_init, f"{k_shot}shot_{channel_name}_SRCNN_best.weights.h5")
    dncnn_path = os.path.join(args.save_init, f"{k_shot}shot_{channel_name}_DNCNN_best.weights.h5")

    print(f"[ChannelNet] Loading {srcnn_path}")
    srcnn.load_weights(srcnn_path)
    print(f"[ChannelNet] Loading {dncnn_path}")
    dncnn.load_weights(dncnn_path)

    return ChannelNetWrapper(srcnn, dncnn)


def load_multigrade_maml(args, channel_name, k_shot, device="cpu"):
    """
    Load multigrade MAML model for a specific channel + K-shot.
    """
    # Determine dataset type from root path
    root_parts = args.root.split('/')
    dataset_name_for_path = 'TDL'
    for part in root_parts:
        if 'UMi' in part or 'umi' in part.lower():
            dataset_name_for_path = 'UMi'
            for p in root_parts:
                if 'speed' in p.lower():
                    dataset_name_for_path = f'UMi_{p}'
                    break
            break
        elif 'TDL' in part or 'tdl' in part.lower():
            dataset_name_for_path = 'TDL'
            break
    
    # Use multigrade_results_dir from args, default to "multigrade_maml_results"
    multigrade_dir = getattr(args, 'multigrade_results_dir', 'multigrade_maml_results')
    
    # Find checkpoint directory
    checkpoint_base_dir = os.path.join(
        multigrade_dir,
        dataset_name_for_path,
        f"checkpoints_nway_{args.n_way}_grades_{args.grades}"
    )
    
    # Find the checkpoint for the last grade
    last_grade = args.grades
    checkpoint_files = [f for f in os.listdir(checkpoint_base_dir) if f.endswith('.pth.tar')] if os.path.exists(checkpoint_base_dir) else []
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_base_dir}")
    
    # Find latest checkpoint for last grade
    last_grade_checkpoints = [f for f in checkpoint_files if f'Grade{last_grade}' in f]
    if not last_grade_checkpoints:
        raise FileNotFoundError(f"No checkpoint found for Grade {last_grade}")
    
    # Sort by step number and take the last one
    def extract_step(fname):
        parts = fname.split('_')
        for p in parts:
            if p.startswith('Step'):
                return int(p.replace('Step', ''))
        return 0
    last_grade_checkpoints.sort(key=extract_step, reverse=True)
    checkpoint_file = last_grade_checkpoints[0]
    ckpt_path = os.path.join(checkpoint_base_dir, checkpoint_file)
    
    print(f"[MultigradeMAML] Loading base checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Get training configuration from checkpoint
    if 'batchsz' in checkpoint:
        training_batchsz = checkpoint['batchsz']
    else:
        state_dict = checkpoint['state_dict']
        if 'net.vars.20' in state_dict:
            training_batchsz = state_dict['net.vars.20'].shape[0]
        else:
            training_batchsz = args.batchsz
    
    training_n_way = checkpoint.get('n_way', args.n_way)
    training_grades = checkpoint.get('grades', args.grades)
    training_meta_lr = checkpoint.get('meta_lr', args.meta_lr)
    training_update_lr = checkpoint.get('update_lr', args.update_lr)
    
    # Create config with correct batchsz
    
    # Simple args class for MultigradeMAMLStair
    class TrainingArgs:
        def __init__(self):
            self.update_lr = training_update_lr
            self.meta_lr = training_meta_lr
            self.n_way = training_n_way
            self.k_spt = args.k_spt
            self.k_qry = args.k_qry
            self.batchsz = training_batchsz
            self.update_step = args.update_step
            self.num_grades = training_grades

    training_args = TrainingArgs()
    
    # Create multigrade MAML model
    multigrade_model = MultigradeMAMLStair(training_args, config, num_grades=training_grades).to(device)
    multigrade_model.load_state_dict(checkpoint['state_dict'])
    
    # Load fine-tuned model if available
    finetuning_subdir = getattr(args, 'multigrade_finetuning_subdir', 'finetuning')
    finetuned_path = os.path.join(
        multigrade_dir,
        dataset_name_for_path,
        finetuning_subdir,
        f"MultigradeMAML_{k_shot}shot_fine_tuned_model_{channel_name}_lr{args.update_lr}.pth"
    )
    
    if os.path.exists(finetuned_path):
        print(f"[MultigradeMAML] Loading fine-tuned model: {finetuned_path}")
        finetuned_checkpoint = torch.load(finetuned_path, map_location=device, weights_only=True)
        if isinstance(finetuned_checkpoint, dict) and 'state_dict' in finetuned_checkpoint:
            multigrade_model.load_state_dict(finetuned_checkpoint['state_dict'])
        else:
            multigrade_model.load_state_dict(finetuned_checkpoint)
    
    multigrade_model.eval()
    return multigrade_model


def load_models_for_channel_kshot(args, channel_name, k_shot, device="cpu",
                                  enable_maml=True, enable_imaml=False, enable_channelnet=True,
                                  enable_multigrade_maml=True):
    """
    Returns dict of models (PyTorch modules) keyed by name.
    LS baseline is implicit (computed directly), so not included.
    """
    models = {}

    if enable_maml:
        from meta import Meta
        config = [
            ('conv2d', [32, 2, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [32]),
            ('conv2d', [128, 32, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [128]),
            ('conv2d', [256, 128, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [256]),
            ('conv2d', [128, 256, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [128]),
            ('conv2d', [32, 128, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [32]),
            ('conv2d', [args.batchsz, 32, 3, 3, 1, 1]),
            ('tanh', [True]),
            ('avg_pool2d', [3, 1, 1]),
            ('bn', [args.batchsz]),
            ('conv2d', [2, args.batchsz, 3, 3, 1, 1])
        ]
        ckpt_path = os.path.join(
            args.maml_dir, f"meta_model_nway_{args.n_way}",
            f"MAML_{k_shot}_shot_fine_tuned_model_{channel_name}_lr{args.update_lr}.pth"
        )
        print(f"[MAML] Loading {ckpt_path}")
        maml = Meta(args, config).to(device).eval()
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        maml.load_state_dict(state)
        models["MAML"] = maml.net

    if enable_imaml:
        learner_net = make_conv_network(in_channels=2, out_dim=2, task='Channel_estimation', batchsz=args.batchsz)
        meta_learner = Learner(model=learner_net,
                               loss_function=torch.nn.MSELoss(),
                               inner_lr=args.inner_lr,
                               outer_lr=args.outer_lr,
                               GPU=(args.device.startswith("cuda")))
        ckpt_path = os.path.join(
            args.save_dir, f"meta_model_nway_{args.n_way}",
            f"wireless_IMAML_{k_shot}_shot_fine_tuned_model_{channel_name}_lr{args.inner_lr}_LAM3.0.pth"
        )
        print(f"[iMAML] Loading {ckpt_path}")
        imaml = meta_learner.model.to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        imaml.load_state_dict(state)
        models["iMAML"] = imaml

    if enable_channelnet:
        if not TF_AVAILABLE or not CHANNELNET_AVAILABLE:
            print(f"[WARNING] ChannelNet requested but TensorFlow/models not available. Skipping ChannelNet.")
        else:
            try:
                models["ChannelNet"] = load_channelnet(args, channel_name, k_shot).to(device).eval()
            except Exception as e:
                print(f"[WARNING] Failed to load ChannelNet: {e}. Skipping ChannelNet.")
                enable_channelnet = False

    if enable_multigrade_maml:
        models["MultigradeMAML"] = load_multigrade_maml(args, channel_name, k_shot, device=device)

    return models


# -----------------------------
# Plotting
# -----------------------------
def per_sample_mse_split(pred_cl, gt_cl):
    """
    Returns per-sample MSEs:
      total: E[(Re)^2 + (Im)^2] over (H,W)
      real : E[(Re)^2]           over (H,W)
      imag : E[(Im)^2]           over (H,W)
    Shapes: pred_cl, gt_cl = (N,H,W,2)
    """
    if pred_cl is None:
        return None, None, None
    diff_r = pred_cl[..., 0] - gt_cl[..., 0]
    diff_i = pred_cl[..., 1] - gt_cl[..., 1]
    mse_r  = np.mean(diff_r * diff_r, axis=(1, 2))
    mse_i  = np.mean(diff_i * diff_i, axis=(1, 2))
    mse    = mse_r + mse_i
    return mse, mse_r, mse_i

def plot_mse_vs_snr(avg_dict, snr_dbs, k_shot, out_dir=None, logy=True, args=None):
    if args is not None:
        os.makedirs(os.path.join(args.root, args.expr_version, f"plots_by_shot_pspacing_{args.p_spacing}"), exist_ok=True)
    plt.figure(figsize=(6, 4))
    for name, mses in avg_dict.items():
        if logy:
            plt.semilogy(snr_dbs, mses, marker='o', label=name)
        else:
            plt.plot(snr_dbs, mses, marker='o', label=name)

    plt.title(f"MSE vs SNR — {k_shot}-shot")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE" + (" (log)" if logy else ""))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Model", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10 )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"mse_vs_snr_{k_shot}shot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✔ Saved: {save_path}")


def _imshow(ax, arr2d, vmin, vmax, title=None):
    im = ax.imshow(arr2d.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return im

def _annotate_mse(ax, val, where="top-left"):
    """Overlay readable MSE text inside the axes."""
    if val is None:
        return
    # choose corner
    if where == "top-left":
        xy = (0.02, 0.98); va = "top"; ha = "left"
    elif where == "bottom-left":
        xy = (0.02, 0.02); va = "bottom"; ha = "left"
    else:
        xy = (0.98, 0.98); va = "top"; ha = "right"
    ax.text(
        xy[0], xy[1], f"MSE={val:.2e}",
        transform=ax.transAxes, ha=ha, va=va, fontsize=8,
        color="w", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none")
    )
    
    
def _plot_six_cols_one_instance(fig, axes, sample_idx, arrays_dict, vmin_r, vmax_r, vmin_i, vmax_i):
    """
    arrays_dict keys (each (N,H,W,2) or None):
      'MAML','iMAML','ChannelNet','MultigradeMAML','LS','LMMSE','GT'
    axes: shape (2, 7) - updated to include MultigradeMAML
    Colorbars: only for GT (both rows).
    """
    cols = ["MAML", "iMAML", "ChannelNet", "MultigradeMAML", "LS", "LMMSE", "GT"]
    arrs = [arrays_dict.get(k) for k in cols]
    gt   = arrays_dict["GT"]

    # compute per-sample MSEs for titles/overlays
    mses_total = {}
    mses_real  = {}
    mses_imag  = {}
    for name, arr in zip(cols, arrs):
        if name == "GT" or arr is None:
            mses_total[name] = mses_real[name] = mses_imag[name] = None
        else:
            mt, mr, mi = per_sample_mse_split(arr, gt)
            mses_total[name] = mt[sample_idx]
            mses_real[name]  = mr[sample_idx]
            mses_imag[name]  = mi[sample_idx]

    for c, (name, arr) in enumerate(zip(cols, arrs)):
        if arr is None:
            axes[0, c].axis("off")
            axes[1, c].axis("off")
            continue

        # REAL row (row 0)
        im0 = _imshow(axes[0, c], arr[sample_idx, ..., 0], vmin_r, vmax_r, title=name)
        # show per-model real-part MSE in the top-left corner
        if name != "GT":
            _annotate_mse(axes[0, c], mses_real[name], where="top-left")

        # IMAG row (row 1)
        im1 = _imshow(axes[1, c], arr[sample_idx, ..., 1], vmin_i, vmax_i)
        # show per-model imag-part MSE in the bottom-left corner (so it doesn't overlap title)
        if name != "GT":
            _annotate_mse(axes[1, c], mses_imag[name], where="bottom-left")

        # Colorbars ONLY for GT column (both rows)
        if name == "GT":
            fig.colorbar(im0, ax=axes[0, c], fraction=0.046, pad=0.01)
            fig.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.01)

def plot_ranked_by_maml_for_snr(
    out_dir_base, ch_name, k_shot, snr_db,
    maml_out, imaml_out, chnet_out_or_none, multigrade_out_or_none, lmmse_out, ls_out, gt_out,
    sample_mses_maml, top_k=5
):
    """
    Create 10 figures (Best1..5, Worst1..5) for this SNR, ranking by MAML MSE.
    All arrays are (N_eval, H, W, 2) and UNscaled (original domain).
    Updated to include MultigradeMAML.
    """
    os.makedirs(out_dir_base, exist_ok=True)

    N = sample_mses_maml.shape[0]
    k = min(top_k, N)
    order = np.argsort(sample_mses_maml)
    groups = [("BEST", order[:k]), ("WORST", order[-k:][::-1])]

    # Prepare dict for convenience
    arrays_dict = {
        "MAML": maml_out,
        "iMAML": imaml_out,
        "ChannelNet": chnet_out_or_none,
        "MultigradeMAML": multigrade_out_or_none,
        "LS": ls_out,
        "LMMSE": lmmse_out,
        "GT": gt_out,
    }
    
    # Compute color ranges per figure (per sample) to be comparable across columns
    for tag, idxs in groups:
        for rank, idx in enumerate(idxs, start=1):
            # vmin/vmax across all seven columns for this sample
            vmin_r = gt_out[idx, ..., 0].min()
            vmax_r = gt_out[idx, ..., 0].max()
            vmin_i = gt_out[idx, ..., 1].min()
            vmax_i = gt_out[idx, ..., 1].max()

            fig, axes = plt.subplots(2, 7, figsize=(21, 5), sharex=True, sharey=True)  # Updated to 7 columns
            _plot_six_cols_one_instance(
                fig, axes, idx, arrays_dict, vmin_r, vmax_r, vmin_i, vmax_i
            )
            fig.suptitle(
                f"{tag} #{rank} — sample {idx} — SNR={snr_db} dB — {k_shot}-shot — {ch_name}",
                y=0.98
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fname = f"{ch_name}_snr{snr_db}dB_{k_shot}shot_{tag.lower()}_{rank:02d}.png"
            fpath = os.path.join(out_dir_base, fname)
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✔ Saved: {fpath}")


def plot_models_vs_gt_random(preds, truth, model_names, snr, out_dir, args=None):
    """
    One random sample; cols = models + GT; rows = real/imag.
    preds: [ (N,H,W,2), ... ]; truth: (N,H,W,2)
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    sample_idx = int(rng.integers(low=0, high=truth.shape[0]))

    n_models = len(preds)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(3*(n_models+1), 6), sharex=True, sharey=True)

    all_real = [p[sample_idx, ..., 0] for p in preds] + [truth[sample_idx, ..., 0]]
    all_imag = [p[sample_idx, ..., 1] for p in preds] + [truth[sample_idx, ..., 1]]
    vmin_r, vmax_r = np.min([a.min() for a in all_real]), np.max([a.max() for a in all_real])
    vmin_i, vmax_i = np.min([a.min() for a in all_imag]), np.max([a.max() for a in all_imag])

    for i, (pred, name) in enumerate(zip(preds, model_names)):
        im0 = axes[0, i].imshow(pred[sample_idx, ..., 0].T, aspect="auto", origin="lower", vmin=vmin_r, vmax=vmax_r)
        axes[0, i].set_title(name)
        fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.01)
        im1 = axes[1, i].imshow(pred[sample_idx, ..., 1].T, aspect="auto", origin="lower", vmin=vmin_i, vmax=vmax_i)
        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.01)

    im0 = axes[0, -1].imshow(truth[sample_idx, ..., 0].T, aspect="auto", origin="lower", vmin=vmin_r, vmax=vmax_r)
    axes[0, -1].set_title("Ground Truth")
    fig.colorbar(im0, ax=axes[0, -1], fraction=0.046, pad=0.01)
    im1 = axes[1, -1].imshow(truth[sample_idx, ..., 1].T, aspect="auto", origin="lower", vmin=vmin_i, vmax=vmax_i)
    fig.colorbar(im1, ax=axes[1, -1], fraction=0.046, pad=0.01)

    fig.suptitle(f"Channel Estimates @ SNR={snr} dB (sample #{sample_idx})", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if args is not None:
        out_path = os.path.join(os.path.join(args.root, args.expr_version, out_dir), f"estimates_snr_{snr}dB.png")
    else:
        out_path = os.path.join(out_dir, f"estimates_snr_{snr}dB.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved: {out_path}")


# -----------------------------
# Core evaluation
# -----------------------------
def build_ls_estimates(Hc, TXc, RXc_time_or_none, snr_dbs, channel_type, umi_interp, pilot_syms, p_spacing, fft_size, cp_len, tdl_cp_len):
    """
    Create LS inputs per SNR, interpolated over the full grid for BOTH TDL and UMi.
    For TDL: if RXc_time_or_none is provided, use it (time-domain RX signal).
    Otherwise, compute RXc = TXc * Hc in frequency domain.
    
    Shapes:
      Hc  : (N, Nsc, Ns) complex
      TXc : (N, Nsc, Ns) or (Nsc, Ns) complex
      RXc_time_or_none : (N, time_samples) complex or None (for TDL time-domain RX)
    Returns:
      list of (N, Nsc, Ns) complex estimates (fully dense), one per SNR.
    """
    if channel_type.lower() == "tdl" and RXc_time_or_none is not None:
        # TDL with time-domain RX: convert to frequency domain first
        use_cp_len = tdl_cp_len if tdl_cp_len is not None else cp_len
        RXc = make_RXc_from_time_domain(RXc_time_or_none, Hc, channel_type="tdl", fft_size=fft_size, cp_len=use_cp_len)
    else:
        # Standard frequency-domain computation: RXc = TXc * Hc
        if TXc.ndim == 2:  # (Nsc, Ns) - broadcast to (N, Nsc, Ns)
            RXc = TXc[np.newaxis, ...] * Hc
        else:  # (N, Nsc, Ns)
            RXc = TXc * Hc
    
    rx_pow = np.mean(np.abs(RXc)**2)
    H_ls_list = []

    for snr in snr_dbs:
        lin   = 10.0**(snr/10.0) #todo: -SNR ?
        sigma = np.sqrt(rx_pow / (2.0 * lin))
        noise = sigma * (np.random.randn(*RXc.shape).astype(np.float32)
                         + 1j*np.random.randn(*RXc.shape).astype(np.float32))
        noise = noise.astype(np.complex64)

        # LS only at pilot REs …
        H_sparse = _ls_only_at_pilots(RXc + noise, TXc, pilot_syms, p_spacing)  # (N,Nsc,Ns)

        # … then ALWAYS interpolate to a full grid (freq -> time), for TDL and UMi
        H_hat = _interp_from_pilots(H_sparse, pilot_syms, p_spacing)            # (N,Nsc,Ns)

        H_ls_list.append(H_hat.astype(np.complex64))

    return H_ls_list

def finetuning_evaluation(
    args,
    root_dir: str,
    channel_type: str,
    umi_interp: bool,
    pilot_syms: list,
    p_spacing: int,
    snr_dbs: list,
    k_shot: int,
    slice_start: int = 30,
    csv_path: str = None,
    device: str = "cpu",
    enable_maml: bool = True,
    enable_imaml: bool = False,
    enable_channelnet: bool = True,
    enable_multigrade_maml: bool = True,
):
    """
    Evaluates on last N_eval samples (after slice_start) across channels for one K-shot.
    Returns avg MSE over channels per SNR for each model (dict name -> list).
    """
    # Load dicts
    tx_dict  = np.load(os.path.join(root_dir, "tx_signal_dict.npy"), allow_pickle=True).item()
    rx_dict  = np.load(os.path.join(root_dir, "rx_signal_dict.npy"), allow_pickle=True).item()
    h_dict   = np.load(os.path.join(root_dir, "channel_label_dict.npy"), allow_pickle=True).item()
    x_dict   = np.load(os.path.join(root_dir, "channel_data_dict.npy"),  allow_pickle=True).item()  # for scaler pool

    # Pick channel keys 
    keys = list(tx_dict.keys())
    tdl_channel_offset = getattr(args, 'tdl_channel_offset', 10)
    if   channel_type.lower() == "tdl": keys = keys[tdl_channel_offset:]
    elif channel_type.lower() == "umi": keys = [keys[-1]]
    else: raise ValueError("channel_type must be 'tdl' or 'umi'")

    # Containers for curves
    model_names = ["LS", "LMMSE"]
    if enable_channelnet: model_names.append("ChannelNet")
    if enable_maml:       model_names.append("MAML")
    if enable_imaml:      model_names.append("iMAML")
    if enable_multigrade_maml: model_names.append("MultigradeMAML")
    raw = {m: [] for m in model_names}

    detailed_dir = f"detailed_plots_pspacing_{args.p_spacing}/{channel_type}_{'interp' if (channel_type=='umi' and umi_interp) else 'nointerp'}_{k_shot}shot"
    os.makedirs(os.path.join(args.root, args.expr_version, detailed_dir), exist_ok=True)

    for ch_name in keys:
        print(f"\n=== Channel: {ch_name} | K-shot: {k_shot} ===")
        # Load models for this channel
        models = load_models_for_channel_kshot(
            args, ch_name, k_shot, device=device,
            enable_maml=enable_maml, enable_imaml=enable_imaml, enable_channelnet=enable_channelnet,
            enable_multigrade_maml=enable_multigrade_maml
        )

        # Truth channel (complex) and TX on grid
        H_raw = h_dict[ch_name]
        Hc    = H_raw[..., 0] + 1j * H_raw[..., 1]
        TX    = tx_dict[ch_name]
        fft_size = getattr(args, 'fft_size', 612)
        cp_len = getattr(args, 'cp_len', 28)
        tdl_cp_len = getattr(args, 'tdl_cp_len', None)
        TXc = make_TXc_grid(TX, Hc, channel_type=channel_type, fft_size=fft_size, cp_len=cp_len, tdl_cp_len=tdl_cp_len)

        # For TDL: get time-domain RX signal if available
        RXc_time = None
        if channel_type.lower() == "tdl" and ch_name in rx_dict:
            RX_raw = rx_dict[ch_name]
            # Check if RX is in time domain (1D or 2D with single column)
            if RX_raw.ndim <= 2:
                RXc_time = RX_raw

        # scaler fitted on the fine-tune pool
        pool_size = getattr(args, 'pool_size', 30)
        X_pool = x_dict[ch_name][:pool_size].astype(np.float32)
        Y_pool = h_dict[ch_name][:pool_size].astype(np.float32)
        _, x_params = Utils.standard_scaling(X_pool)
        _, y_params = Utils.standard_scaling(Y_pool)

        eval_idx = slice(slice_start, None)  # last samples
        N_eval = Hc[eval_idx].shape[0]

        # Build LS inputs for each SNR
        H_ls_list = build_ls_estimates(Hc, TXc, RXc_time, snr_dbs, channel_type, umi_interp, pilot_syms, p_spacing, fft_size, cp_len, tdl_cp_len)

        # --- LS baseline curve (global) ---
        ls_mses = [ np.mean(np.abs(Hh[eval_idx] - Hc[eval_idx])**2) for Hh in H_ls_list ]
        per_model_curves = {"LS": ls_mses}
        raw["LS"].append(ls_mses)

        # --- LMMSE baseline: keep full grids for later panels ---
        lmmse_mses = []
        lmmse_full_list = []
        for snr_idx, Hh in enumerate(H_ls_list):
            H_lmmse = lmmse_baseline_np(
                H_clean=Hc, H_ls=Hh, snr_db=snr_dbs[snr_idx],
                pilot_syms=pilot_syms, p_spacing=p_spacing
            )
            lmmse_full_list.append(H_lmmse)
            lmmse_mses.append(np.mean(np.abs(H_lmmse[eval_idx] - Hc[eval_idx])**2))
        per_model_curves["LMMSE"] = lmmse_mses
        raw["LMMSE"].append(lmmse_mses)

        # Prepare per-model per-SNR averages
        if enable_channelnet: chnet_avgs = []
        if enable_maml:       maml_avgs  = []
        if enable_imaml:      imaml_avgs = []
        if enable_multigrade_maml: multigrade_avgs = []

        for snr_idx, (H_ls_complex, H_lmmse_complex) in enumerate(zip(H_ls_list, lmmse_full_list)):
            # Build inputs/labels (channel-last)
            inp_eval = np.stack([H_ls_complex.real, H_ls_complex.imag], axis=-1)[eval_idx]   # (N_eval, 612,14,2)
            lab_eval = np.stack([Hc.real,            Hc.imag],            axis=-1)[eval_idx] # (N_eval, 612,14,2)
            lmmse_eval = np.stack([H_lmmse_complex.real, H_lmmse_complex.imag], axis=-1)[eval_idx]

            # Scale with pool params (consistent with finetuning)
            inp_s = apply_scaling_with_params(inp_eval, x_params)
            lab_s = apply_scaling_with_params(lab_eval, y_params)

            # ----- Forward passes -----
            maml_out = None
            chnet_out = None
            imaml_out = None
            multigrade_out = None

            if enable_maml:
                t = to_torch_cf(inp_s, device)                          # (N_eval,2,612,14)
                with torch.no_grad():
                    out_pf = models["MAML"](t, vars=None, bn_training=False)
                out_s = np.transpose(out_pf.detach().cpu().numpy(), (0, 2, 3, 1))
                maml_out = Utils.unscale_standard(out_s, y_params)
                # MAML per-sample MSE for ranking
                diff_maml = maml_out - lab_eval
                sample_mses_maml = np.mean(diff_maml[..., 0]**2 + diff_maml[..., 1]**2, axis=(1, 2))
                maml_avgs.append(sample_mses_maml.mean())
            else:
                sample_mses_maml = None

            if enable_channelnet and "ChannelNet" in models:
                t = torch.from_numpy(inp_s).float().to(device)
                with torch.no_grad():
                    out_s = models["ChannelNet"](t).cpu().numpy()
                chnet_out = Utils.unscale_standard(out_s, y_params)
                diff_chn = chnet_out - lab_eval
                chnet_avgs.append(np.mean(diff_chn[..., 0]**2 + diff_chn[..., 1]**2, axis=(1, 2)).mean())

            if enable_imaml and "iMAML" in models:
                t = to_torch_cf(inp_s, device)
                with torch.no_grad():
                    out_pf = models["iMAML"](t)
                out_s = np.transpose(out_pf.detach().cpu().numpy(), (0, 2, 3, 1))
                imaml_out = Utils.unscale_standard(out_s, y_params)
                diff_im = imaml_out - lab_eval
                imaml_avgs.append(np.mean(diff_im[..., 0]**2 + diff_im[..., 1]**2, axis=(1, 2)).mean())

            if enable_multigrade_maml and "MultigradeMAML" in models:
                t = to_torch_cf(inp_s, device)
                with torch.no_grad():
                    out_pf = models["MultigradeMAML"].net(t, vars=None, bn_training=False)
                out_s = np.transpose(out_pf.detach().cpu().numpy(), (0, 2, 3, 1))
                multigrade_out = Utils.unscale_standard(out_s, y_params)
                diff_mg = multigrade_out - lab_eval
                multigrade_avgs.append(np.mean(diff_mg[..., 0]**2 + diff_mg[..., 1]**2, axis=(1, 2)).mean())

            # ----- Ranked multi-model panels (by MAML) -----
            if enable_maml and sample_mses_maml is not None:
                out_dir_panels = os.path.join(args.root, args.expr_version, detailed_dir, ch_name, f"SNR_{snr_dbs[snr_idx]}dB")

                arrays_dict = {
                    "maml_out": maml_out,
                    "imaml_out": imaml_out if enable_imaml else None,
                    "chnet_out": chnet_out if enable_channelnet else None,
                    "multigrade_out": multigrade_out if enable_multigrade_maml else None,
                    "lmmse_out": lmmse_eval,
                    "ls_out": inp_eval,
                    "gt_out": lab_eval,
                }
                plot_ranked_by_maml_for_snr(
                    out_dir_base=out_dir_panels, ch_name=ch_name, k_shot=k_shot, snr_db=snr_dbs[snr_idx],
                    maml_out=arrays_dict["maml_out"],
                    imaml_out=arrays_dict["imaml_out"],
                    chnet_out_or_none=arrays_dict["chnet_out"],
                    multigrade_out_or_none=arrays_dict["multigrade_out"],
                    lmmse_out=arrays_dict["lmmse_out"],
                    ls_out=arrays_dict["ls_out"],
                    gt_out=arrays_dict["gt_out"],
                    sample_mses_maml=sample_mses_maml,
                    top_k=5
                )

        # Fill curve dicts for learned models
        if enable_channelnet:
            per_model_curves["ChannelNet"] = chnet_avgs
            raw["ChannelNet"].append(chnet_avgs)
        if enable_maml:
            per_model_curves["MAML"] = maml_avgs
            raw["MAML"].append(maml_avgs)
        if enable_imaml:
            per_model_curves["iMAML"] = imaml_avgs
            raw["iMAML"].append(imaml_avgs)
        if enable_multigrade_maml:
            per_model_curves["MultigradeMAML"] = multigrade_avgs
            raw["MultigradeMAML"].append(multigrade_avgs)

        # ---- per-channel curves (plot + CSV) ----
        interp_str = "interp" if (channel_type.lower() == "umi" and umi_interp) else "nointerp"
        per_ch_out_dir = os.path.join(args.root, args.expr_version, 
            f"plots_by_channel_pspacing_{args.p_spacing}",
            f"{channel_type.lower()}_{interp_str}",
            f"{k_shot}shot",
            ch_name
        )
        os.makedirs(per_ch_out_dir, exist_ok=True)

        plot_mse_vs_snr(per_model_curves, snr_dbs, k_shot=k_shot, out_dir=per_ch_out_dir, logy=True, args=args)

        df_ch = pd.DataFrame(per_model_curves, index=snr_dbs)
        df_ch.index.name = "SNR_dB"
        df_ch.to_csv(os.path.join(per_ch_out_dir, "mse_vs_snr.csv"))

    # Average across channels
    avg = { name: np.stack(raw[name]).mean(axis=0).tolist() for name in raw if len(raw[name]) > 0 }

    if csv_path:
        df = pd.DataFrame(avg, index=snr_dbs)
        df.index.name = "SNR_dB"
        long = df.reset_index().melt(id_vars="SNR_dB", var_name="Model", value_name="MSE")
        long.to_csv(csv_path, index=False)

    return avg


def make_grid_example(
    args,
    root_dir: str,
    channel_type: str,
    umi_interp: bool,
    pilot_syms: list,
    p_spacing: int,
    snr: int,
    k_shot: int,
    slice_start: int,
    device: str = "cpu",
    enable_maml: bool = True,
    enable_imaml: bool = False,
    enable_channelnet: bool = True,
    enable_multigrade_maml: bool = True,
):
    """
    Returns (preds, truth, model_names) for one channel & one SNR.
    All arrays UNscaled: (N_eval, 612, 14, 2).
    """
    tx_dict  = np.load(os.path.join(root_dir, "tx_signal_dict.npy"), allow_pickle=True).item()
    rx_dict  = np.load(os.path.join(root_dir, "rx_signal_dict.npy"), allow_pickle=True).item()
    h_dict   = np.load(os.path.join(root_dir, "channel_label_dict.npy"), allow_pickle=True).item()
    x_dict   = np.load(os.path.join(root_dir, "channel_data_dict.npy"),  allow_pickle=True).item()

    keys = list(tx_dict.keys())
    tdl_channel_offset = getattr(args, 'tdl_channel_offset', 10)
    if   channel_type.lower() == "tdl": keys = keys[tdl_channel_offset:]
    elif channel_type.lower() == "umi": keys = [keys[-1]]

    ch_name = keys[0]  # pick one
    H_raw = h_dict[ch_name]
    Hc    = H_raw[..., 0] + 1j * H_raw[..., 1]
    TX    = tx_dict[ch_name]
    fft_size = getattr(args, 'fft_size', 612)
    cp_len = getattr(args, 'cp_len', 28)
    tdl_cp_len = getattr(args, 'tdl_cp_len', None)
    TXc = make_TXc_grid(TX, Hc, channel_type=channel_type, fft_size=fft_size, cp_len=cp_len, tdl_cp_len=tdl_cp_len)
    
    # For TDL: get time-domain RX signal if available
    RXc_time = None
    if channel_type.lower() == "tdl" and ch_name in rx_dict:
        RX_raw = rx_dict[ch_name]
        if RX_raw.ndim <= 2:
            RXc_time = RX_raw
    
    eval_idx = slice(slice_start, None)

    # scaler params
    pool_size = getattr(args, 'pool_size', 30)
    X_pool = x_dict[ch_name][:pool_size].astype(np.float32)
    Y_pool = h_dict[ch_name][:pool_size].astype(np.float32)
    _, x_params = Utils.standard_scaling(X_pool)
    _, y_params = Utils.standard_scaling(Y_pool)

    # LS estimate at requested SNR
    H_ls = build_ls_estimates(Hc, TXc, RXc_time, [snr], channel_type, umi_interp, pilot_syms, p_spacing, fft_size, cp_len, tdl_cp_len)[0]
    inp_eval = np.stack([H_ls.real, H_ls.imag], axis=-1)[eval_idx]
    lab_eval = np.stack([Hc.real,  Hc.imag], axis=-1)[eval_idx]

    # LMMSE for this SNR
    H_lmmse = lmmse_baseline_np(H_clean=Hc, H_ls=H_ls, snr_db=snr, pilot_syms=pilot_syms, p_spacing=p_spacing)
    lmmse_eval = np.stack([H_lmmse.real, H_lmmse.imag], axis=-1)[eval_idx]

    # Load models
    models = load_models_for_channel_kshot(
        args, ch_name, k_shot, device=device,
        enable_maml=enable_maml, enable_imaml=enable_imaml, enable_channelnet=enable_channelnet,
        enable_multigrade_maml=enable_multigrade_maml
    )

    preds = []
    names = []

    if "MAML" in models:
        t_in = to_torch_cf(apply_scaling_with_params(inp_eval, x_params), device)
        with torch.no_grad():
            out_pf = models["MAML"](t_in, vars=None, bn_training=False)
        out_s = np.transpose(out_pf.detach().cpu().numpy(), (0, 2, 3, 1))
        preds.append(Utils.unscale_standard(out_s, y_params))
        names.append("MAML")

    if "ChannelNet" in models:
        t_in = torch.from_numpy(apply_scaling_with_params(inp_eval, x_params)).float().to(device)
        with torch.no_grad():
            out_s = models["ChannelNet"](t_in).cpu().numpy()
        preds.append(Utils.unscale_standard(out_s, y_params))
        names.append("ChannelNet")

    if "MultigradeMAML" in models:
        t_in = to_torch_cf(apply_scaling_with_params(inp_eval, x_params), device)
        with torch.no_grad():
            out_pf = models["MultigradeMAML"].net(t_in, vars=None, bn_training=False)
        out_s = np.transpose(out_pf.detach().cpu().numpy(), (0, 2, 3, 1))
        preds.append(Utils.unscale_standard(out_s, y_params))
        names.append("MultigradeMAML")

    preds.append(lmmse_eval.copy()); names.append("LMMSE")
    preds.append(inp_eval.copy());   names.append("LS")

    truth = lab_eval
    return preds, truth, names


def run_all_kshots(
    args,
    root_dir: str,
    channel_type: str,
    umi_interp: bool,
    pilot_syms: list,
    p_spacing: int,
    snr_dbs: list,
    k_shot_list: list,
    slice_start: int,
    device: str
):
    out_dir_curves = os.path.join(args.root, args.expr_version, f"New_plotting_pspacing_{args.p_spacing}")
    os.makedirs(out_dir_curves, exist_ok=True)

    grid_snr = snr_dbs[len(snr_dbs)//2]
    all_avgs = {}

    for k in k_shot_list:
        csv_path = os.path.join(args.root, args.expr_version, f"{channel_type}_{'interp' if (channel_type=='umi' and umi_interp) else 'nointerp'}_{k}shot_results.csv")
        avg = finetuning_evaluation(
            args=args,
            root_dir=root_dir,
            channel_type=channel_type,
            umi_interp=umi_interp,
            pilot_syms=pilot_syms,
            p_spacing=p_spacing,
            snr_dbs=snr_dbs,
            k_shot=k,
            slice_start=slice_start,
            csv_path=csv_path,
            device=device,
            enable_maml=enable_maml,
            enable_imaml=enable_imaml,
            enable_channelnet=enable_channelnet,
            enable_multigrade_maml=enable_multigrade_maml,
        )
        all_avgs[k] = avg
        plot_mse_vs_snr(avg, snr_dbs, k_shot=k, out_dir=out_dir_curves, logy=True, args=args)

    return all_avgs


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Checkpoint roots
    p.add_argument("--save_init", help="channelnet checkpoints directory", type=str, required=True)
    p.add_argument("--maml_dir",  help="maml checkpoints directory", type=str, required=True)
    p.add_argument("--save_dir",  help="imaml checkpoints directory", type=str, required=True)
    p.add_argument("--multigrade_results_dir", help="multigrade MAML results base directory", type=str, default="multigrade_maml_results")
    p.add_argument("--multigrade_finetuning_subdir", help="subdirectory name for fine-tuned models", type=str, default="finetuning")

    # MAML/iMAML hyperparams for checkpoint names
    p.add_argument("--n_way",     type=int,   default=5)
    p.add_argument("--update_lr", type=float, default=1e-3)
    p.add_argument("--lam",       type=float, default=1e-3)

    # ChannelNet FT LR (only if enabled)
    p.add_argument("--train_lr",  type=float, default=1e-4)

    # Meta-learner details
    p.add_argument("--device",    type=str,   default="cuda:0")
    p.add_argument("--batchsz",   type=int,   default=8)
    p.add_argument("--update_step", type=int, default=2)
    p.add_argument("--meta_lr",   type=float, default=1e-4)
    p.add_argument("--k_spt",     type=int,   default=5)
    p.add_argument("--k_qry",     type=int,   default=5)

    # iMAML specific
    p.add_argument("--inner_lr",  type=float, default=1e-3)
    p.add_argument("--outer_lr",  type=float, default=1e-3)

    # Multigrade MAML
    p.add_argument("--grades", type=int, default=3, help="Number of grades in multigrade MAML")

    # Data & eval config
    p.add_argument("--root", type=str, required=True, help="Root directory containing dataset files")
    p.add_argument("--channel_type", type=str, choices=["tdl", "umi"], required=True, help="Channel type: 'tdl' or 'umi'")
    p.add_argument("--umi_interp", type=str, choices=["on", "off"], default="on",
                   help="UMi only: turn LS 2D interpolation on/off")
    p.add_argument("--pilot_syms", type=int, nargs="+", required=True, help="Pilot symbol indices")
    p.add_argument("--p_spacing", type=int, required=True, help="Pilot spacing in frequency domain")
    p.add_argument("--snr_start", type=int, required=True, help="Start SNR in dB")
    p.add_argument("--snr_stop", type=int, required=True, help="Stop SNR in dB")
    p.add_argument("--snr_step", type=int, required=True, help="SNR step size in dB")
    p.add_argument("--k_shots", type=int, nargs="+", required=True, help="List of k-shot values to evaluate")
    p.add_argument("--slice_start", type=int, required=True, help="Starting index for evaluation samples")
    p.add_argument("--expr_version", type=str, required=True, help="Experiment version identifier for output paths")
    
    # OFDM parameters
    p.add_argument("--fft_size", type=int, default=612, help="FFT size for OFDM")
    p.add_argument("--cp_len", type=int, default=28, help="Cyclic prefix length for UMi")
    p.add_argument("--tdl_cp_len", type=int, default=None, help="Cyclic prefix length for TDL (overrides cp_len if set)")
    
    # Data processing parameters
    p.add_argument("--pool_size", type=int, default=30, help="Number of samples for scaler fitting pool")
    p.add_argument("--tdl_channel_offset", type=int, default=10, help="Channel offset for TDL dataset (number of channels to skip)")
    p.add_argument("--random_seed", type=int, default=222, help="Random seed for reproducibility")
    
    # Model enable/disable flags
    p.add_argument("--disable_channelnet", action="store_true", help="Disable ChannelNet evaluation")
    p.add_argument("--disable_maml", action="store_true", help="Disable MAML evaluation")
    p.add_argument("--disable_multigrade_maml", action="store_true", help="Disable MultigradeMAML evaluation")
    p.add_argument("--enable_imaml", action="store_true", help="Enable iMAML evaluation")
    

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed if provided
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
    
    # Determine which models to enable/disable
    enable_channelnet = not args.disable_channelnet and TF_AVAILABLE and CHANNELNET_AVAILABLE
    enable_maml = not args.disable_maml
    enable_multigrade_maml = not args.disable_multigrade_maml
    enable_imaml = args.enable_imaml
    
    if not enable_channelnet and TF_AVAILABLE == False:
        print("[INFO] ChannelNet disabled: TensorFlow not available")
    elif args.disable_channelnet:
        print("[INFO] ChannelNet explicitly disabled via --disable_channelnet")
    
    device = args.device if torch.cuda.is_available() else "cpu"

    snrs = list(range(args.snr_start, args.snr_stop + 1, args.snr_step))
    umi_interp = (args.umi_interp.lower() == "on")

    all_avgs = run_all_kshots(
        args=args,
        root_dir=args.root,
        channel_type=args.channel_type.lower(),
        umi_interp=umi_interp,
        pilot_syms=args.pilot_syms,
        p_spacing=args.p_spacing,
        snr_dbs=snrs,
        k_shot_list=args.k_shots,
        slice_start=args.slice_start,
        device=device
    )
    print("\nDone.")

