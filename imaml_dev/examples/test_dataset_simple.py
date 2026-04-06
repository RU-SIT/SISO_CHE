#!/usr/bin/env python3
"""
Simple dataset test without PyTorch dependencies
"""

import os
import sys
import numpy as np
from pathlib import Path

_WC_ROOT = Path(__file__).resolve().parents[2]
if str(_WC_ROOT) not in sys.path:
    sys.path.insert(0, str(_WC_ROOT))
from paths import default_dataset_umi_pspacing

def test_dataset_simple():
    """Test dataset loading without PyTorch."""
    data_dir = default_dataset_umi_pspacing()
    
    print("🔍 Simple Dataset Test")
    print("="*50)
    
    # Check files exist
    required_files = [
        'channel_data_dict.npy',
        'channel_label_dict.npy', 
        'rx_signal_dict.npy',
        'tx_signal_dict.npy'
    ]
    
    print("📁 Checking files:")
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - MISSING")
            return False
    
    # Load data
    print("\n📊 Loading data:")
    try:
        data_dict = np.load(os.path.join(data_dir, 'channel_data_dict.npy'), allow_pickle=True).item()
        labels_dict = np.load(os.path.join(data_dir, 'channel_label_dict.npy'), allow_pickle=True).item()
        
        file_names = list(data_dict.keys())
        print(f"  ✓ Loaded {len(file_names)} files: {file_names}")
        
        # Check first file
        first_file = file_names[0]
        first_data = data_dict[first_file]
        first_labels = labels_dict[first_file]
        
        print(f"  ✓ First file '{first_file}':")
        print(f"    - Data shape: {first_data.shape}")
        print(f"    - Labels shape: {first_labels.shape}")
        print(f"    - Samples: {first_data.shape[0]}")
        
        # Check if we can do N_way=4, K_shot=5
        n_way = 4
        k_shot = 5
        
        print(f"\n🧪 Testing parameters:")
        print(f"  - N_way: {n_way}")
        print(f"  - K_shot: {k_shot}")
        
        # Check train/test split
        train_files = file_names[:4]
        test_files = file_names[4:]
        
        print(f"  - Train files: {len(train_files)} ({train_files})")
        print(f"  - Test files: {len(test_files)} ({test_files})")
        
        if len(train_files) >= n_way:
            print(f"  ✓ Enough train files for N_way={n_way}")
        else:
            print(f"  ❌ Not enough train files for N_way={n_way}")
            return False
        
        # Check samples per file
        min_samples = min([data_dict[file].shape[0] for file in train_files])
        print(f"  - Min samples per train file: {min_samples}")
        
        if min_samples >= k_shot:
            print(f"  ✓ Enough samples for K_shot={k_shot}")
        else:
            print(f"  ❌ Not enough samples for K_shot={k_shot}")
            return False
        
        print(f"\n✅ Dataset is ready for analysis!")
        print(f"  - Use N_way={n_way}, K_shot={k_shot}")
        print(f"  - Train files: {train_files}")
        print(f"  - Test files: {test_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_dataset_simple()
    if success:
        print("\n🎉 Dataset test passed!")
    else:
        print("\n❌ Dataset test failed!")
