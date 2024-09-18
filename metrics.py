import numpy as np
import pdb
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

    # def bit_error_rate(self, channel, snr_db, num_bits=10000, modulation='BPSK'):
    #     """
    #     Calculate the Bit Error Rate (BER) for a given set of transmitted symbols.
        
    #     Parameters:
    #     - channel channel (output from the model)
    #     - snr_db: Signal-to-noise ratio in dB
    #     - num_bits: Number of bits to transmit (default: 10000)
    #     - modulation: Modulation scheme ('BPSK' or '16QAM')
        
    #     Returns:
    #     - BER: Bit Error Rate
    #     """
    #     # Adjust number of bits for 16-QAM
    #     if modulation == '16QAM':
    #         num_bits = (num_bits // 4) * 4  # Ensure num_bits is a multiple of 4 for 16-QAM

    #     # Generate random bits
    #     bits = np.random.randint(0, 2, size=num_bits)
        
    #     # Modulate bits into symbols
    #     tx_modulated_symbols = self.modulate(bits, modulation=modulation)

    #     # Reshape channel (the predicted output) to match the modulated symbol shape
    #     # if modulation == '16QAM':
    #     #     channel = channel.flatten()[:len(tx_modulated_symbols)]  # Adjust length for 16-QAM
    #     pdb.set_trace()
    #     tx_symbols_channel = np.dot(channel, tx_modulated_symbols)
    #     # Add noise to the signal
    #     rx_symbols_noisy = self.add_noise(tx_symbols_channel, snr_db, num_bits)
        
    #     # Demodulate received symbols back into bits
    #     rx_bits = self.demodulate(rx_symbols_noisy, modulation=modulation)
        
    #     # Ensure the length of received bits matches the transmitted bits
    #     rx_bits = rx_bits[:len(bits)]
        
    #     # Calculate the number of bit errors
    #     num_errors = np.sum(bits != rx_bits)
        
    #     # Calculate BER
    #     ber = num_errors / len(bits)
    #     return ber

   