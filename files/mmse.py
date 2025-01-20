import numpy as np
import torch
from scipy.stats import multivariate_normal

def calculate_mmse_estimate(y, constellation, noise_var, prior_probs=None):
    """
    Calculate MMSE estimate for received symbols.
    x̂_mmse = E[x|y] = ∑(x_i * P(x_i|y))
    Args:
        y (numpy.ndarray): Received signal samples (N x 2)
        constellation (numpy.ndarray): Constellation points (M x 2)
        noise_var (float): Noise variance
        prior_probs (numpy.ndarray, optional): Prior probabilities for each constellation point
        
    Returns:
        numpy.ndarray: MMSE estimates for each received sample
    """
    if prior_probs is None:
        prior_probs = np.ones(len(constellation)) / len(constellation)
        
    N = len(y)
    M = len(constellation)
    mmse_estimates = np.zeros_like(y)
    eps = 1e-300  # Small constant to prevent division by zero
    
    # For each received sample
    for i in range(N):
        posteriors = np.zeros(M)
        
        # Calculate posterior probability for each constellation point
        for j in range(M):
            # Likelihood using Gaussian noise model
            likelihood = multivariate_normal.pdf(
                y[i], 
                mean=constellation[j], 
                cov=noise_var * np.eye(2)
            )
            posteriors[j] = likelihood * prior_probs[j]
            
        # Normalize posteriors with numerical stability
        sum_posteriors = np.sum(posteriors) + eps
        posteriors = posteriors / sum_posteriors
        
        # Calculate MMSE estimate as weighted sum
        for j in range(M):
            mmse_estimates[i] += constellation[j] * posteriors[j]
            
    return mmse_estimates

def calculate_mse(original, estimated):
    """
    Calculate Mean Square Error between original and estimated signals.
    
    Args:
        original (numpy.ndarray): Original signals
        estimated (numpy.ndarray): Estimated signals
        
    Returns:
        float: Mean Square Error
    """
    return np.mean(np.sum((original - estimated) ** 2, axis=1))

def estimate_channel_mmse(y, x, noise_var):
    """
    Calculate MMSE estimate for channel coefficient H in y = Hx + n
    
    Args:
        y (numpy.ndarray): Received signal samples (N x 2)
        x (numpy.ndarray): Transmitted symbols (N x 2)
        noise_var (float): Noise variance
        
    Returns:
        numpy.ndarray: MMSE estimate of channel coefficient H
    """
    # For MMSE channel estimation: H_mmse = (X^H X)^(-1) X^H y
    x_H = x.T  # Hermitian transpose
    H_mmse = np.linalg.inv(x_H @ x + noise_var * np.eye(2)) @ x_H @ y
    
    return H_mmse

def calculate_ber(original_symbols, estimated_symbols, n_bits):
    """
    Calculate Bit Error Rate (BER)
    
    Args:
        original_symbols: Original symbol indices
        estimated_symbols: Estimated symbol indices
        n_bits: Number of bits per symbol
    
    Returns:
        float: Bit Error Rate
    """
    # Convert symbols to binary strings
    original_bits = np.unpackbits(np.array(original_symbols, dtype=np.uint8))
    estimated_bits = np.unpackbits(np.array(estimated_symbols, dtype=np.uint8))
    
    # Calculate bit errors
    bit_errors = np.sum(original_bits != estimated_bits)
    total_bits = len(original_bits)
    
    return bit_errors / total_bits

if __name__ == "__main__":
    from data_generation import generate_channel_data_simulated, generate_qam_constellation
    import matplotlib.pyplot as plt
    
    # Configuration
    config = {
        'n_bits': 4,  # 16-QAM
        'dim_encoding': 2,
        'type_channel': 'fading_ricean',  # or 'fading', 'fading_ricean', etc.
        'sigma_noise_measurement': 0.15,
        'avg_power_symbols': 1.0,
        'rate_comm': 4/5,
        'EbNodB_min': 0.,
        'EbNo_min': 10. ** (0. / 10.),
        'mod_order': 16
    }
    
    # Generate constellation
    constellation = generate_qam_constellation(config['mod_order'], avg_power=config['avg_power_symbols'])
    
    # SNR range
    snr_range = np.arange(0, 35, 5)
    
    # Initialize arrays for both MSE and BER
    mse_values = []
    ber_values = []
    
    # Increase number of samples for better statistical significance
    n_samples = 1000  # Increased from 1000 to 10000
    
    # Calculate MSE and BER for each SNR
    for snr in snr_range:
        config['SNR_channel_dB'] = snr
        config['EbNodB_range'] = np.array(range(-5,15))
        
        # Adjust noise variance based on SNR
        config['sigma_noise_measurement'] = np.sqrt(1 / (10 ** (snr/10)))
        
        # Generate channel data
        x_train, y_train, x_val, y_val, symbols_train, symbols_val, const, csi = generate_channel_data_simulated(
            config['type_channel'],
            config['SNR_channel_dB'],
            n_samples,
            config,
            constellation
        )
        
        # Calculate MMSE estimates
        noise_var = config['sigma_noise_measurement']**2
        mmse_estimates = calculate_mmse_estimate(y_train.numpy(), constellation, noise_var)
        
        # Calculate MSE
        mse = calculate_mse(x_train.numpy(), mmse_estimates)
        mse_values.append(mse)
        
        # Find nearest constellation points for BER calculation
        distances = np.array([np.sum((mmse_estimates - const_point)**2, axis=1) 
                            for const_point in constellation])
        estimated_symbols = np.argmin(distances, axis=0)
        original_symbols = np.argmax(symbols_train.numpy(), axis=1)
        
        # Calculate BER
        ber = calculate_ber(original_symbols, estimated_symbols, config['n_bits'])
        ber_values.append(ber)
        
        print(f"SNR: {snr:2d} dB, Noise Var: {noise_var:.6e}, MSE: {mse:.10e}, BER: {ber:.10e}")
    
    # Create output directory if it doesn't exist
    import os
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save MSE vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, mse_values, 'b.-', linewidth=2, markersize=10)
    plt.grid(True)
    plt.title('MMSE Performance vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Square Error')
    plt.yscale('log')
    plt.ylim(bottom=1e-5)  # Adjust y-axis limit to show smaller values
    plt.savefig(os.path.join(output_dir, f'mmse_performance_{config["type_channel"]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot and save BER vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, ber_values, 'r.-', linewidth=2, markersize=10)
    plt.grid(True)
    plt.title('Bit Error Rate vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.ylim(bottom=1e-5)  # Adjust y-axis limit to show smaller values
    plt.savefig(os.path.join(output_dir, f'ber_performance_{config["type_channel"]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save numerical results
    results = np.column_stack((snr_range, mse_values, ber_values))
    np.savetxt(os.path.join(output_dir, f'mmse_performance_{config["type_channel"]}.csv'), 
               results, delimiter=',', header='SNR,MSE,BER', comments='',
               fmt=['%d', '%.10e', '%.10e'])  # Increased precision in saved file

# # Configuration
#     config = {
#         'n_bits': 4,  # 16-QAM
#         'dim_encoding': 2,
#         'type_channel': 'awgn',  # or 'fading', 'fading_ricean', etc.
#         'SNR_channel_dB': 15,
#         'sigma_noise_measurement': 0.1,
#         'avg_power_symbols': 1.0,
#         'rate_comm': 4/5,
#         'EbNodB_range': np.array([15]),
#         'EbNodB_min': 0,
#         'mod_order': 16
#     }
    
#     # Generate constellation
#     constellation = generate_qam_constellation(config['mod_order'], avg_power=config['avg_power_symbols'])
    
#     # Generate channel data
#     n_samples = 1000
#     x_train, y_train, x_val, y_val, symbols_train, symbols_val, const, csi = generate_channel_data_simulated(
#         config['type_channel'],
#         config['SNR_channel_dB'],
#         n_samples,
#         config,
#         constellation
#     )
    
#     # Calculate MMSE estimates
#     noise_var = config['sigma_noise_measurement']**2
#     mmse_estimates = calculate_mmse_estimate(y_train.numpy(), constellation, noise_var)
    
#     # Calculate MSE
#     mse = calculate_mse(x_train.numpy(), mmse_estimates)
#     print(f"Mean Square Error: {mse:.6f}")
    
#     # Add channel estimation
#     H_est = estimate_channel_mmse(y_train.numpy(), x_train.numpy(), noise_var)
#     print(f"Estimated channel matrix H:\n{H_est}")
    
#     # You can now use H_est to reconstruct the signal
#     x_reconstructed = y_train.numpy() @ np.linalg.inv(H_est)
    
#     # Plotting
#     plt.figure(figsize=(10, 10))
#     plt.scatter(constellation[:, 0], constellation[:, 1], c='black', label='Constellation', marker='x', s=100)
#     plt.scatter(x_train.numpy()[:, 0], x_train.numpy()[:, 1], c='blue', label='Original', alpha=0.5)
#     plt.scatter(y_train.numpy()[:, 0], y_train.numpy()[:, 1], c='red', label='Received', alpha=0.3)
#     plt.scatter(mmse_estimates[:, 0], mmse_estimates[:, 1], c='green', label='MMSE Estimate', alpha=0.5)
#     plt.legend()
#     plt.grid(True)
#     plt.title(f'MMSE Estimation (MSE: {mse:.6f})')
#     plt.xlabel('In-phase')
#     plt.ylabel('Quadrature')
#     plt.axis('equal')
#     plt.show()
    
#     # Additional plotting for channel estimation results
#     plt.figure(figsize=(10, 10))
#     plt.scatter(x_train.numpy()[:, 0], x_train.numpy()[:, 1], c='blue', label='Original', alpha=0.5)
#     plt.scatter(x_reconstructed[:, 0], x_reconstructed[:, 1], c='green', label='Reconstructed', alpha=0.5)
#     plt.legend()
#     plt.grid(True)
#     plt.title('Channel Estimation Results')
#     plt.xlabel('In-phase')
#     plt.ylabel('Quadrature')
#     plt.axis('equal')
#     plt.show()
