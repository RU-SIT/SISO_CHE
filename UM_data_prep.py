import os
import glob
import h5py
import numpy as np
import pdb
# import tensorflow as tf
from scipy.interpolate import griddata


def load_and_prepare(path, pilot_syms, p_spacing):
    """
    path       : str, path to your .h5 (or .mat) file with datasets "x","y","h"
    pilot_syms : list or 1D array of symbol indices where pilots live (time axis)
    p_spacing  : int, spacing between pilot subcarriers (frequency axis)
    
    returns:
      inputs  : np.array, shape (N, Ns, Nsc, 2) real/imag of estimated H
      labels  : np.array, shape (N, Ns, Nsc, 2) real/imag of true h
    """
    # --- load raw x, y, true h
    with h5py.File(path, "r") as f:
        x = np.squeeze(f["x"][...])    # shape (N, Ns, Nsc)
        y = np.squeeze(f["y"][...])
        h = np.squeeze(f["h"][...])
    
    # --- LS estimate at every point, but we'll only use pilots
    H_ls = y / x                   # complex, shape (N, Ns, Nsc)
    N, Ns, Nsc = H_ls.shape

    # pilot subcarrier indices
    pilot_subcs = np.arange(0, Nsc, p_spacing)

    # prepare output array for estimated H
    H_hat = np.zeros((N, Ns, Nsc), dtype=complex)

    # 1) freq‐axis interpolation at each pilot symbol
    all_subs = np.arange(Nsc)
    for n in range(N):
        for sym in pilot_syms:
            # extract LS at pilot subcarriers
            Hp = H_ls[n, sym, pilot_subcs]
            # interp real and imag separately
            re = np.interp(all_subs, pilot_subcs, Hp.real)
            im = np.interp(all_subs, pilot_subcs, Hp.imag)
            H_hat[n, sym, :] = re + 1j*im

    # 2) time‐axis interpolation at each subcarrier
    all_syms = np.arange(Ns)
    for n in range(N):
        for k in range(Nsc):
            # take the freq‐interpolated values at pilot symbols
            Hp = H_hat[n, pilot_syms, k]
            re = np.interp(all_syms, pilot_syms, Hp.real)
            im = np.interp(all_syms, pilot_syms, Hp.imag)
            H_hat[n, :, k] = re + 1j*im
    # pdb.set_trace()
    # stack real & imag into last dim
    inputs = np.stack([H_hat.real, H_hat.imag], axis=-1)
    labels = np.stack([h.real,     h.imag],     axis=-1)

    return inputs, labels


def save_all_dicts(data_dir, pilot_syms, p_spacing, n_select=256):
    pattern = os.path.join(data_dir, "data_snr*.hdf5")
    files   = sorted(glob.glob(pattern))

    inp_dict, lab_dict = {}, {}
    tx_dict,  rx_dict  = {}, {}

    for path in files:
        name = os.path.basename(path)

        # 1) load & squeeze raw x,y
        with h5py.File(path, "r") as f:
            x_raw = np.squeeze(f["x"][...])  # (10000,14,612) complex
            y_raw = np.squeeze(f["y"][...])

        # 2) build inputs & labels (10000,14,612,2)
        inputs, labels = load_and_prepare(path, pilot_syms, p_spacing)

        # 3) turn x_raw/y_raw into 2-channel real/imag
        x_stack = np.stack([x_raw.real, x_raw.imag], axis=-1)  # (10000,14,612,2)
        y_stack = np.stack([y_raw.real, y_raw.imag], axis=-1)

        # 4) select first n_select, then transpose to (256,612,14,2)
        inp_sel = inputs[:n_select].transpose(0, 2, 1, 3)
        lab_sel = labels[:n_select].transpose(0, 2, 1, 3)
        x_sel   = x_stack[:n_select].transpose(0, 2, 1, 3)
        y_sel   = y_stack[:n_select].transpose(0, 2, 1, 3)

        # 5) stash in dicts
        inp_dict[name] = inp_sel
        lab_dict[name] = lab_sel
        tx_dict[name]  = x_sel
        rx_dict[name]  = y_sel

    # save each dict to .npy
    np.save("/home/CAMPUS/rghasemi/projects/MyPrivaterepo/Sionna_datasets/ps2_p612/speed5/SISO-UMi/channel_data_dict.npy", inp_dict)
    np.save("/home/CAMPUS/rghasemi/projects/MyPrivaterepo/Sionna_datasets/ps2_p612/speed5/SISO-UMi/channel_label_dict.npy", lab_dict)
    np.save("/home/CAMPUS/rghasemi/projects/MyPrivaterepo/Sionna_datasets/ps2_p612/speed5/SISO-UMi/tx_signal_dict.npy",    tx_dict)
    np.save("/home/CAMPUS/rghasemi/projects/MyPrivaterepo/Sionna_datasets/ps2_p612/speed5/SISO-UMi/rx_signal_dict.npy",    rx_dict)
    print(f"Saved {len(files)} entries into four .npy files in {os.getcwd()}")

if __name__ == "__main__":
    data_dir   = "/home/CAMPUS/rghasemi/projects/MyPrivaterepo/Sionna_datasets/ps2_p612/speed5/SISO-UMi/"
    pilot_syms = [3, 10]
    p_spacing  = 4
    save_all_dicts(data_dir, pilot_syms, p_spacing, n_select=256)