#!/usr/bin/env python3
"""
Dataset Diagnostic Script

This script helps diagnose issues with the dataset loading and structure.
"""

import os
import numpy as np
import sys

def diagnose_dataset(data_dir):
    """Diagnose dataset structure and identify issues."""
    print(f"Diagnosing dataset at: {data_dir}")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Directory does not exist: {data_dir}")
        return False
    
    # Check required files
    required_files = [
        'channel_data_dict.npy',
        'channel_label_dict.npy', 
        'rx_signal_dict.npy',
        'tx_signal_dict.npy'
    ]
    
    print("📁 Checking required files:")
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - MISSING")
            return False
    
    # Load and check data dictionaries
    print("\n📊 Checking data structure:")
    
    try:
        # Load data dictionary
        data_dict_path = os.path.join(data_dir, 'channel_data_dict.npy')
        data_dict = np.load(data_dict_path, allow_pickle=True).item()
        print(f"  ✓ channel_data_dict.npy loaded successfully")
        print(f"    - Number of files: {len(data_dict)}")
        
        if len(data_dict) == 0:
            print("  ❌ Data dictionary is empty!")
            return False
        
        # Check first few files
        file_names = list(data_dict.keys())
        print(f"    - File names: {file_names[:5]}{'...' if len(file_names) > 5 else ''}")
        
        # Check data shape for first file
        first_file = file_names[0]
        first_data = data_dict[first_file]
        print(f"    - First file '{first_file}' shape: {first_data.shape}")
        
        if len(first_data) == 0:
            print("  ❌ First file has no data samples!")
            return False
        
        # Check if we have enough files for N_way
        n_way = 5  # From your updated parameters
        if len(file_names) < n_way:
            print(f"  ❌ Not enough files for N_way={n_way}. Only {len(file_names)} files available.")
            return False
        
        # Check labels dictionary
        labels_dict_path = os.path.join(data_dir, 'channel_label_dict.npy')
        labels_dict = np.load(labels_dict_path, allow_pickle=True).item()
        print(f"  ✓ channel_label_dict.npy loaded successfully")
        print(f"    - Number of files: {len(labels_dict)}")
        
        # Check if labels match data
        if set(labels_dict.keys()) != set(data_dict.keys()):
            print("  ❌ Label files don't match data files!")
            return False
        
        # Check first file labels
        first_labels = labels_dict[first_file]
        print(f"    - First file labels shape: {first_labels.shape}")
        
        if len(first_labels) == 0:
            print("  ❌ First file has no labels!")
            return False
        
        # Check rx_signal dictionary
        rx_signal_path = os.path.join(data_dir, 'rx_signal_dict.npy')
        rx_signal = np.load(rx_signal_path, allow_pickle=True).item()
        print(f"  ✓ rx_signal_dict.npy loaded successfully")
        print(f"    - Number of files: {len(rx_signal)}")
        
        # Check tx_signal dictionary
        tx_signal_path = os.path.join(data_dir, 'tx_signal_dict.npy')
        tx_signal = np.load(tx_signal_path, allow_pickle=True).item()
        print(f"  ✓ tx_signal_dict.npy loaded successfully")
        print(f"    - Number of files: {len(tx_signal)}")
        
        # Check data consistency
        print(f"\n🔍 Checking data consistency:")
        for i, file_name in enumerate(file_names[:3]):  # Check first 3 files
            data_shape = data_dict[file_name].shape
            labels_shape = labels_dict[file_name].shape
            rx_shape = rx_signal[file_name].shape if file_name in rx_signal else "N/A"
            tx_shape = tx_signal[file_name].shape if file_name in tx_signal else "N/A"
            
            print(f"  File {i+1} '{file_name}':")
            print(f"    - Data: {data_shape}")
            print(f"    - Labels: {labels_shape}")
            print(f"    - RX Signal: {rx_shape}")
            print(f"    - TX Signal: {tx_shape}")
            
            # Check if data and labels have same number of samples
            if data_shape[0] != labels_shape[0]:
                print(f"    ❌ Data and labels have different number of samples!")
                return False
        
        # Check if we have enough samples for K_shot
        k_shot = 5  # From your updated parameters
        min_samples = min([data_dict[file].shape[0] for file in file_names])
        print(f"\n📈 Sample analysis:")
        print(f"  - Minimum samples per file: {min_samples}")
        print(f"  - Required K_shot: {k_shot}")
        
        if min_samples < k_shot:
            print(f"  ❌ Not enough samples for K_shot={k_shot}. Minimum available: {min_samples}")
            return False
        
        # Check train/test split
        print(f"\n📚 Train/Test split:")
        train_files = file_names[:4]
        test_files = file_names[4:]
        print(f"  - Train files: {len(train_files)}")
        print(f"  - Test files: {len(test_files)}")
        
        if len(train_files) < n_way:
            print(f"  ❌ Not enough train files for N_way={n_way}. Available: {len(train_files)}")
            return False
        
        print(f"\n✅ Dataset appears to be valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_data_loading(data_dir):
    """Test the actual data loading process."""
    print(f"\n🧪 Testing data loading process:")
    print("="*60)
    
    try:
        # Import the data loader
        sys.path.insert(0, os.path.dirname(__file__))
        from Data_Nshot import ChannelEstimationNShot
        
        # Try to create the dataset
        print("Creating ChannelEstimationNShot instance...")
        dataset = ChannelEstimationNShot(
            root=data_dir,
            batchsz=8,
            n_way=5,
            k_shot=5,
            k_query=5
        )
        print("✅ Dataset created successfully!")
        
        # Try to get a batch
        print("Testing batch loading...")
        try:
            batch = dataset.next(mode='train')
            print("✅ Batch loaded successfully!")
            print(f"  - Batch contains {len(batch)} elements")
            return True
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

def main():
    """Main diagnostic function."""
    import sys
    from pathlib import Path

    _WC_ROOT = Path(__file__).resolve().parents[2]
    if str(_WC_ROOT) not in sys.path:
        sys.path.insert(0, str(_WC_ROOT))
    from paths import default_dataset_umi_pspacing

    data_dir = default_dataset_umi_pspacing()
    
    print("🔍 iMAML Dataset Diagnostic Tool")
    print("="*60)
    
    # Step 1: Diagnose dataset structure
    if not diagnose_dataset(data_dir):
        print("\n❌ Dataset diagnosis failed!")
        return False
    
    # Step 2: Test data loading
    if not test_data_loading(data_dir):
        print("\n❌ Data loading test failed!")
        return False
    
    print("\n🎉 All tests passed! Dataset is ready for use.")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
