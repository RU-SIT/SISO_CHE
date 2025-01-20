import numpy as np

import pdb
import re

class Metric():
    def __init__(self):
        pass

    def modulate(self, bits, modulation='BPSK'):
        """
        Modulate the bits into symbols.
        Supports BPSK and 16-QAM modulation.
        """
        if modulation == 'BPSK':
            return 2 * bits - 1  # BPSK: 0 -> -1, 1 -> 1
        elif modulation == '16QAM':
            # Reshape the bit stream into groups of 4 bits for 16-QAM
            bits = bits.reshape(-1, 4)
            # Convert groups of 4 bits into decimal values (0-15)
            decimal_values = bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))
            # Map decimal values to 16-QAM constellation points
            mapping = {
                0: -3 - 3j, 1: -3 - 1j, 2: -3 + 3j, 3: -3 + 1j,
                4: -1 - 3j, 5: -1 - 1j, 6: -1 + 3j, 7: -1 + 1j,
                8:  3 - 3j, 9:  3 - 1j, 10: 3 + 3j, 11: 3 + 1j,
                12: 1 - 3j, 13: 1 - 1j, 14: 1 + 3j, 15: 1 + 1j
            }
            symbols = np.vectorize(mapping.get)(decimal_values)
            return symbols
        else:
            raise ValueError('Unsupported modulation scheme')

    def demodulate(self, symbols, modulation='BPSK'):
        """
        Demodulate symbols back into bits.
        Supports BPSK and 16-QAM demodulation.
        """
        if modulation == 'BPSK':
            return (symbols.real >= 0).astype(int)  # BPSK: < 0 -> 0, >= 0 -> 1
        elif modulation == '16QAM':
            # Define decision boundaries for 16-QAM demodulation
            def decision_boundary(value):
                if value < -2:
                    return 0
                elif value < 0:
                    return 1
                elif value < 2:
                    return 3
                else:
                    return 2

            # Demodulate the symbols back into 4-bit values
            real_part = np.vectorize(decision_boundary)(symbols.real)
            imag_part = np.vectorize(decision_boundary)(symbols.imag)
            
            # Combine real and imaginary parts to form bits
            bits = np.zeros((len(real_part), 4), dtype=int)
            
            for i, (r, j) in enumerate(zip(real_part, imag_part)):
                # Reconstruct the original 4-bit group from constellation point
                bits[i] = np.array([r // 2, r % 2, j // 2, j % 2])
            
            return bits.flatten()  # Return as a flat bit array
        else:
            raise ValueError('Unsupported modulation scheme')

    def add_noise(self, signal, snr_db):
        """
        Add AWGN noise to the signal based on the specified SNR in dB.
        
        """
        
        snr_linear = 10**(snr_db / 10)
        power_signal = np.mean(np.abs(signal)**2)
        noise_power = power_signal / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise
    
    def bit_error_rate(self, channel, snr_db, num_bits=10000, modulation='BPSK'):
        """
        Calculate the Bit Error Rate (BER) for a given set of transmitted symbols.

        Parameters:
        - channel: Predicted output from the model, containing real and imaginary parts.
        - snr_db: Signal-to-noise ratio in dB.
        - num_bits: Number of bits to transmit (default: 10000).
        - modulation: Modulation scheme ('BPSK' or '16QAM').

        Returns:
        - BER: Bit Error Rate.
        """
        # Adjust number of bits for 16-QAM
        if modulation == '16QAM':
            num_bits = (num_bits // 4) * 4  # Ensure num_bits is a multiple of 4 for 16-QAM

        # Generate random bits
        bits = np.random.randint(0, 2, size=num_bits)
        # pdb.set_trace()

        # Modulate bits into symbols
        tx_modulated_symbols = self.modulate(bits, modulation=modulation)
        
        # The channel contains real and imaginary parts, so we need to handle that separately
        real_channel = channel[0,...].flatten()  # Real part of the channel
        imag_channel = channel[1,...].flatten()  # Imaginary part of the channel

        # Ensure the channel and tx_modulated_symbols are compatible for multiplication
        if len(real_channel) < len(tx_modulated_symbols):
            tx_modulated_symbols = tx_modulated_symbols[:len(real_channel)]
        else:
            real_channel = real_channel[:len(tx_modulated_symbols)]
            imag_channel = imag_channel[:len(tx_modulated_symbols)]

        # Combine the real and imaginary parts to form a complex channel
        complex_channel = real_channel + 1j * imag_channel

        # Perform element-wise multiplication between the channel and the modulated symbols
        tx_symbols_channel = complex_channel * tx_modulated_symbols

        # Add noise to the signal
        rx_symbols_noisy = self.add_noise(tx_symbols_channel, snr_db)

        # Demodulate received symbols back into bits
        rx_bits = self.demodulate(rx_symbols_noisy, modulation=modulation)

        # For 16-QAM, each symbol corresponds to 4 bits, so ensure the output matches num_bits
        if modulation == '16QAM':
            # Ensure rx_bits has the same length as bits (10000 in this case)
            if len(rx_bits) != num_bits:
                rx_bits = rx_bits[:num_bits]  # Trim the demodulated bits if necessary
        else:
            rx_bits = rx_bits[:num_bits]  # Trim the demodulated bits to match the original bit count

        # Calculate the number of bit errors
        num_errors = np.sum(bits != rx_bits)

        # Calculate BER
        ber = num_errors / len(bits)
        return ber

def extract_snr(file_name):
    match = re.search(r'SNR_(\d+)db', file_name)
    if match:
        return int(match.group(1))  # Extract the number and convert to integer
    return None 

def mmse_channel_estimation(rx, ch, channel_var, noise_var):
    """
    MMSE channel estimation with alignment of time-domain received signal and subcarrier-symbol domain channel.
    :param y: Received signal in the time domain.
    :param ch: Channel in the subcarrier-symbol domain (frequency domain).
    :param noise_var: Noise variance.
    :param channel_var: channel variance.
    :return: Estimated transmitted signal in the frequency domain, split into real and imaginary components.
    TODO: compute h_hat based o tx and rx : rx/tx at pilot location
    """
    # pdb.set_trace()
    
    # Convert channel to complex numbers
    ch = ch[..., 0] + 1j * ch[..., 1]

    # Reshape or truncate y to match the grid of ch
    num_subcarriers, num_symbols = ch.shape
    expected_size = num_subcarriers * num_symbols

    if rx.size > expected_size:
        # Truncate excess samples from y
        rx = rx[:expected_size]
        # print(f"Truncated y to size {rx.size} to match channel grid.")
    elif y.size < expected_size:
        raise ValueError(f"Insufficient samples in y: {rx.size}, expected {expected_size} to align with ch.")

    # Reshape y to match the grid of ch
    rx = rx.reshape(num_subcarriers, num_symbols)

    # Perform FFT on y along the subcarrier dimension to convert to the frequency domain
    rx_freq = np.fft.fft(rx, axis=0)

    # Ensure alignment of dimensions
    if rx_freq.shape != ch.shape:
        raise ValueError(f"Shape mismatch after FFT: y_freq shape {rx_freq.shape} vs ch shape {ch.shape}")

    # Calculate the MMSE estimator
    h_conj = np.conj(ch)  # Complex conjugate of channel h
    # mmse_factor = h_conj / (np.abs(ch)**2 + noise_var / channel_var)
    mmse_factor = h_conj / (np.abs(ch)**2 + noise_var)
    
    x_hat = mmse_factor * rx_freq

    # Split into real and imaginary components
    mmse_factor_real = np.real(mmse_factor)
    mmse_factor_imag = np.imag(mmse_factor)
    mmse_factor_split = np.stack((mmse_factor_real, mmse_factor_imag), axis=-1)
    
    x_hat_real = np.real(x_hat)
    x_hat_imag = np.imag(x_hat)
    x_hat_split = np.stack((x_hat_real, x_hat_imag), axis=-1)# Combine as [real, imag]

    return x_hat_split, mmse_factor_split



# def MIMO_MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR):
#     """
#     MMSE Channel Estimation for OFDM

#     Parameters:
#         Y (np.ndarray): Frequency-domain received signal.
#         Xp (np.ndarray): Pilot signal.
#         pilot_loc (np.ndarray): Pilot locations.
#         Nfft (int): FFT size.
#         Nps (int): Pilot spacing.
#         h (np.ndarray): Channel impulse response.
#         SNR (float): Signal-to-Noise Ratio (in dB).

#     Returns:
#         np.ndarray: MMSE channel estimate (H_MMSE).
#     """
#     # Convert SNR from dB to linear scale
#     snr = 10**(SNR / 10)

#     # Number of pilots
#     Np = Nfft // Nps

#     # Least Squares (LS) Estimate
#     H_tilde = Y[pilot_loc] / Xp

#     # RMS Delay Spread Calculation
#     k = np.arange(len(h))
#     hh = np.sum(np.abs(h)**2)
#     tmp = np.abs(h)**2 * k
#     r = np.sum(tmp) / hh
#     r2 = np.sum(tmp * k) / hh
#     tau_rms = np.sqrt(r2 - r**2)

#     # Frequency-domain correlation
#     df = 1 / Nfft
#     j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
#     K1 = np.tile(np.arange(Nfft).reshape(-1, 1), (1, Np))
#     K2 = np.tile(np.arange(Np), (Nfft, 1))
#     rf = 1 / (1 + j2pi_tau_df * (K1 - K2 * Nps))

#     # Pilot autocorrelation
#     K3 = np.tile(np.arange(Np).reshape(-1, 1), (1, Np))
#     K4 = np.tile(np.arange(Np), (Np, 1))
#     rf2 = 1 / (1 + j2pi_tau_df * Nps * (K3 - K4))
#     Rpp = rf2 + np.eye(Np) / snr
#     Rhp = rf

#     # MMSE Channel Estimation
#     H_MMSE = np.dot(np.dot(Rhp, np.linalg.inv(Rpp)), H_tilde).T

#     return H_MMSE



# def siso_ofdm_mmse(pilot_indices, num_subcarriers=64, num_symbols=12, pilot_spacing=4, SNR_dB=20):
#     """
#     Simulates a SISO OFDM system with MMSE channel estimation.
    
#     Parameters:
#         num_subcarriers (int): Number of OFDM subcarriers (FFT size).
#         num_symbols (int): Number of OFDM symbols.
#         pilot_spacing (int): Spacing between pilot subcarriers.
#         SNR_dB (float): Signal-to-Noise Ratio in dB.
    
#     Returns:
#         dict: Contains transmitted symbols, received symbols, channel, and estimates.
#     """
#     # Parameters
#     snr_linear = 10**(SNR_dB / 10)  # SNR in linear scale
#     noise_variance = 1 / snr_linear
#     num_pilots = len(pilot_indices)

#     # Generate pilot symbols
#     pilot_symbols = (np.random.randint(0, 2, num_pilots) * 2 - 1) + \
#                     1j * (np.random.randint(0, 2, num_pilots) * 2 - 1)
 
#     # MMSE Channel Estimation
#     def mmse_channel_estimation(rx, channel_impulse pilot_symbols, pilot_indices, num_symbols, snr_linear):
#         """MMSE Channel Estimation for OFDM."""
#         # LS Estimate
#         H_tilde = rx[pilot_indices, :] / pilot_symbols[:, np.newaxis]

#         # Calculate RMS delay spread (using a simplified random model)
#         k = np.arange(num_symbols)
#         hh = np.sum(np.abs(channel_impulse)**2)
#         tmp = np.abs(channel_impulse)**2 * k
#         r = np.sum(tmp) / hh
#         r2 = np.sum(tmp * k) / hh
#         tau_rms = np.sqrt(r2 - r**2)
        
#         # Frequency correlation
#         df = 1 / num_subcarriers
#         j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
#         K1 = np.tile(np.arange(num_subcarriers).reshape(-1, 1), (1, num_pilots))
#         K2 = np.tile(np.arange(num_pilots), (num_subcarriers, 1))
#         rf = 1 / (1 + j2pi_tau_df * (K1 - K2 * pilot_spacing))
        
#         # Autocorrelation
#         K3 = np.tile(np.arange(num_pilots).reshape(-1, 1), (1, num_pilots))
#         K4 = np.tile(np.arange(num_pilots), (num_pilots, 1))
#         rf2 = 1 / (1 + j2pi_tau_df * pilot_spacing * (K3 - K4))
#         Rpp = rf2 + np.eye(num_pilots) / snr_linear
#         Rhp = rf

#         # MMSE Estimation
#         H_mmse = np.dot(np.dot(Rhp, np.linalg.inv(Rpp)), H_tilde).T
#         return H_mmse
    
    