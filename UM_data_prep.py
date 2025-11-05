import os
import glob
import h5py
import numpy as np
import argparse

EPS = 1e-8


def _robust_ls(y, x, eps=EPS):
    """Robust LS: y/x = y * conj(x) / (|x|^2 + eps). Works on any broadcastable shape."""
    num = y * np.conj(x)
    den = (x.real * x.real + x.imag * x.imag) + eps
    return num / den

def _pilot_subcarriers(Nsc: int, p_spacing: int) -> np.ndarray:
    if p_spacing is None or p_spacing <= 0:
        raise ValueError("p_spacing must be a positive integer.")
    return np.arange(0, Nsc, int(p_spacing), dtype=int)

def load_and_prepare(path, pilot_syms, p_spacing, umi_interp=True):
    """
    Build inputs/labels where LS is performed **only at pilot REs**.
      - umi_interp=False: keep only pilot-RE LS (non-pilot REs are zeros).
      - umi_interp=True : interpolate non-pilot REs from pilot-RE LS (freq->time).
    Returns:
      inputs: (N, Ns, Nsc, 2)   RI of estimated H (from pilot LS, optionally interpolated)
      labels: (N, Ns, Nsc, 2)   RI of true H
    """
    import h5py
    with h5py.File(path, "r") as f:
        x = np.squeeze(f["x"][...])    # (N, Ns, Nsc) complex
        y = np.squeeze(f["y"][...])    # (N, Ns, Nsc) complex
        h = np.squeeze(f["h"][...])    # (N, Ns, Nsc) complex

    N, Ns, Nsc = h.shape
    # sanitize pilot indices
    pilot_syms = np.asarray(pilot_syms, dtype=int)
    pilot_syms = pilot_syms[(pilot_syms >= 0) & (pilot_syms < Ns)]
    if pilot_syms.size == 0:
        raise ValueError("pilot_syms empty after clipping to [0, Ns). Provide valid pilot symbols.")
    pilot_subcs = _pilot_subcarriers(Nsc, p_spacing)
    if pilot_subcs.size < 2 and umi_interp:
        raise ValueError("Not enough pilot subcarriers for interpolation; increase density (smaller spacing).")

    # ---- LS **only** at pilot REs ----
    # Initialize sparse LS grid with zeros elsewhere
    H_sparse = np.zeros((N, Ns, Nsc), dtype=np.complex64)

    # Vectorized over batch for each pilot symbol
    for sym in pilot_syms:
        # y/x on pilot subcarriers only
        y_p = y[:, sym, pilot_subcs]   # (N, P)
        x_p = x[:, sym, pilot_subcs]   # (N, P)
        H_p = _robust_ls(y_p, x_p)     # (N, P)
        # scatter to sparse grid
        H_sparse[:, sym, pilot_subcs] = H_p.astype(np.complex64, copy=False)

    if not umi_interp:
        # keep only pilot-RE LS; others remain zero
        inputs = np.stack([H_sparse.real, H_sparse.imag], axis=-1).astype(np.float32, copy=False)
        labels = np.stack([h.real,         h.imag],       axis=-1).astype(np.float32, copy=False)
        return inputs, labels

    # ---- Interpolate non-pilot REs from pilot LS ----
    H_hat = np.zeros_like(H_sparse, dtype=np.complex64)

    # 1) Frequency interpolation at each pilot symbol
    all_subs = np.arange(Nsc, dtype=int)
    for n in range(N):
        for sym in pilot_syms:
            Hp = H_sparse[n, sym, pilot_subcs]      # (P,) complex (pilot LS only)
            # Linear interp (real/imag separately)
            re = np.interp(all_subs, pilot_subcs, Hp.real)
            im = np.interp(all_subs, pilot_subcs, Hp.imag)
            H_hat[n, sym, :] = re + 1j * im

    # 2) Time interpolation at each subcarrier using values at pilot symbols
    all_syms = np.arange(Ns, dtype=int)
    for n in range(N):
        for k in range(Nsc):
            vals = H_hat[n, pilot_syms, k]          # (K,) complex at pilot symbols
            re = np.interp(all_syms, pilot_syms, vals.real)
            im = np.interp(all_syms, pilot_syms, vals.imag)
            H_hat[n, :, k] = re + 1j * im

    inputs = np.stack([H_hat.real, H_hat.imag], axis=-1).astype(np.float32, copy=False)
    labels = np.stack([h.real,     h.imag],     axis=-1).astype(np.float32, copy=False)
    return inputs, labels



def save_all_dicts(data_dir, pilot_syms, p_spacing, n_select=256, umi_interp=True, out_dir=None):
    """
    Reads all 'data_snr*.hdf5' under data_dir, builds dictionaries and saves them as .npy.

    When umi_interp=True: use 2-D pilot interpolation for inputs.
    When umi_interp=False: inputs are raw LS (no interpolation).

    Saves to:
      <out_dir or data_dir/{interpolated|nointerp}>/channel_data_dict.npy
      <...>/channel_label_dict.npy
      <...>/tx_signal_dict.npy
      <...>/rx_signal_dict.npy
    """
    pattern = os.path.join(data_dir, "data_snr*.hdf5")
    files   = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched {pattern}")

    # Default subfolder to avoid overwriting the other mode’s outputs
    if out_dir is None:
        mode = "interpolated_noleak" if umi_interp else "nointerp"
        out_dir = os.path.join(data_dir, mode)
    os.makedirs(out_dir, exist_ok=True)

    inp_dict, lab_dict = {}, {}
    tx_dict,  rx_dict  = {}, {}

    for path in files:
        name = os.path.basename(path)

        # 1) load & squeeze raw x,y for TX/RX dicts
        with h5py.File(path, "r") as f:
            x_raw = np.squeeze(f["x"][...])  # (N, Ns, Nsc) complex
            y_raw = np.squeeze(f["y"][...])  # (N, Ns, Nsc) complex

        # 2) build inputs & labels (N, Ns, Nsc, 2) according to flag
        inputs, labels = load_and_prepare(path, pilot_syms, p_spacing, umi_interp=umi_interp)

        # 3) turn x_raw/y_raw into 2-channel real/imag (N, Ns, Nsc, 2)
        x_stack = np.stack([x_raw.real, x_raw.imag], axis=-1)
        y_stack = np.stack([y_raw.real, y_raw.imag], axis=-1)

        # 4) select first n_select, then transpose to (N, Nsc, Ns, 2)
        inp_sel = inputs[:n_select].transpose(0, 2, 1, 3)
        lab_sel = labels[:n_select].transpose(0, 2, 1, 3)
        x_sel   = x_stack[:n_select].transpose(0, 2, 1, 3)
        y_sel   = y_stack[:n_select].transpose(0, 2, 1, 3)

        # 5) stash in dicts
        inp_dict[name] = inp_sel
        lab_dict[name] = lab_sel
        tx_dict[name]  = x_sel
        rx_dict[name]  = y_sel

    # save each dict to .npy under out_dir
    np.save(os.path.join(out_dir, "channel_data_dict.npy"),  inp_dict)
    np.save(os.path.join(out_dir, "channel_label_dict.npy"), lab_dict)
    np.save(os.path.join(out_dir, "tx_signal_dict.npy"),     tx_dict)
    np.save(os.path.join(out_dir, "rx_signal_dict.npy"),     rx_dict)
    print(f"Saved {len(files)} entries into four .npy files in {out_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare UM data for channel estimation with pilot interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/rghasemi/Wireless_communication/Sionna_datasets/ps2_p612/speed5/SISO_pspacing_17/speed5",
        help="Directory containing data_snr*.hdf5 files"
    )
    
    parser.add_argument(
        "--pilot_syms", 
        type=int, 
        nargs="+", 
        default=[3, 10],
        help="Pilot symbol indices (time axis)"
    )
    
    parser.add_argument(
        "--p_spacing", 
        type=int, 
        default=17,
        help="Spacing between pilot subcarriers (frequency axis)"
    )
    
    parser.add_argument(
        "--n_select", 
        type=int, 
        default=256,
        help="Number of samples to select from each file"
    )
    
    parser.add_argument(
        "--umi_interp", 
        action="store_true", 
        default=True,
        help="Enable 2D pilot interpolation (frequency then time)"
    )
    
    parser.add_argument(
        "--no_interp", 
        action="store_true", 
        help="Disable interpolation (use raw LS estimates only)"
    )
    
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default=None,
        help="Output directory (default: data_dir/{interpolated_noleak|nointerp})"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle interpolation flag
    if args.no_interp:
        args.umi_interp = False
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        print(f"Data directory: {args.data_dir}")
        print(f"Pilot symbols: {args.pilot_syms}")
        print(f"Pilot spacing: {args.p_spacing}")
        print(f"Number of samples: {args.n_select}")
        print(f"Interpolation enabled: {args.umi_interp}")
        print(f"Output directory: {args.out_dir}")
        print()
    
    try:
        # Run data preparation
        save_all_dicts(
            data_dir=args.data_dir,
            pilot_syms=args.pilot_syms,
            p_spacing=args.p_spacing,
            n_select=args.n_select,
            umi_interp=args.umi_interp,
            out_dir=args.out_dir
        )
        
        if args.verbose:
            print("Data preparation completed successfully!")
            
    except Exception as e:
        print(f"Error during data preparation: {e}")
        exit(1)

# def _robust_ls(y, x, eps=EPS):
#     """
#     Robust LS: y/x computed as y * conj(x) / (|x|^2 + eps)
#     Shapes: (N, Ns, Nsc) complex
#     """
#     num = y * np.conj(x)
#     den = (x.real * x.real + x.imag * x.imag) + eps
#     return num / den


# def load_and_prepare(path, pilot_syms, p_spacing, umi_interp=True):
#     """
#     path       : str, path to your .h5 (or .mat) file with datasets "x","y","h"
#     pilot_syms : list or 1D array of symbol indices where pilots live (time axis)
#     p_spacing  : int, spacing between pilot subcarriers (frequency axis)
#     umi_interp : bool, if True do 2-D pilot interpolation; else return raw LS

#     returns:
#       inputs  : np.array, shape (N, Ns, Nsc, 2) real/imag of estimated H (input to models)
#       labels  : np.array, shape (N, Ns, Nsc, 2) real/imag of true h (ground truth)
#     """
#     # --- load raw x, y, true h
#     with h5py.File(path, "r") as f:
#         # Expect complex datasets stored as complex dtype; if split, adjust here
#         x = np.squeeze(f["x"][...])    # (N, Ns, Nsc) complex
#         y = np.squeeze(f["y"][...])    # (N, Ns, Nsc) complex
#         h = np.squeeze(f["h"][...])    # (N, Ns, Nsc) complex

#     # --- LS estimate at every TF point (robust)
#     H_ls = _robust_ls(y, x)            # (N, Ns, Nsc) complex
#     N, Ns, Nsc = H_ls.shape

#     if not umi_interp:
#         # No interpolation: inputs are just raw LS everywhere
#         inputs = np.stack([H_ls.real, H_ls.imag], axis=-1)     # (N, Ns, Nsc, 2)
#         labels = np.stack([h.real,   h.imag],   axis=-1)       # (N, Ns, Nsc, 2)
#         return inputs, labels

#     # ---------- Interpolation path (UMi pilot interpolation in freq then time) ----------
#     # Validate/clip pilot indices
#     pilot_syms = np.asarray(pilot_syms, dtype=int)
#     pilot_syms = pilot_syms[(pilot_syms >= 0) & (pilot_syms < Ns)]
#     if pilot_syms.size == 0:
#         raise ValueError("pilot_syms empty after clipping to [0, Ns). Provide valid pilot symbols.")
#     if p_spacing is None or p_spacing <= 0:
#         raise ValueError("p_spacing must be a positive integer for interpolation.")

#     pilot_subcs = np.arange(0, Nsc, int(p_spacing))
#     if pilot_subcs.size < 2:
#         # Need at least two pilot tones for np.interp to work meaningfully
#         raise ValueError("Not enough pilot subcarriers; increase p_spacing density (smaller spacing).")

#     H_hat = np.zeros((N, Ns, Nsc), dtype=complex)

#     # 1) frequency-axis interpolation at each pilot symbol
#     all_subs = np.arange(Nsc)
#     for n in range(N):
#         for sym in pilot_syms:
#             Hp = H_ls[n, sym, pilot_subcs]     # (P,) complex
#             re = np.interp(all_subs, pilot_subcs, Hp.real)
#             im = np.interp(all_subs, pilot_subcs, Hp.imag)
#             H_hat[n, sym, :] = re + 1j * im

#     # 2) time-axis interpolation at each subcarrier using values at pilot symbols
#     all_syms = np.arange(Ns)
#     for n in range(N):
#         for k in range(Nsc):
#             Hp = H_hat[n, pilot_syms, k]       # (K,) complex at pilot symbols
#             re = np.interp(all_syms, pilot_syms, Hp.real)
#             im = np.interp(all_syms, pilot_syms, Hp.imag)
#             H_hat[n, :, k] = re + 1j * im

#     inputs = np.stack([H_hat.real, H_hat.imag], axis=-1)       # (N, Ns, Nsc, 2)
#     labels = np.stack([h.real,     h.imag],     axis=-1)       # (N, Ns, Nsc, 2)
#     return inputs, labels

