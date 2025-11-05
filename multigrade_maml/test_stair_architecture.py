#!/usr/bin/env python3
"""
Test the stair-like multigrade MAML architecture
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigrade_maml_stair import MultigradeMAMLStair

class SimpleArgs:
    def __init__(self):
        self.update_lr = 1e-3
        self.meta_lr = 1e-4
        self.n_way = 2
        self.k_spt = 5
        self.k_qry = 5
        self.batchsz = 4
        self.update_step = 2
        self.num_grades = 3

def create_dummy_data():
    """Create dummy data for testing"""
    batchsz, setsz, c_, h, w = 2, 5, 2, 14, 14
    
    x_qry = torch.randn(batchsz, setsz, c_, h, w)
    y_qry = torch.randn(batchsz, setsz, c_, h, w)
    x_spt = torch.randn(batchsz, setsz, c_, h, w)
    y_spt = torch.randn(batchsz, setsz, c_, h, w)
    
    return x_qry, y_qry, x_spt, y_spt

def test_stair_architecture():
    """Test the stair-like multigrade MAML architecture"""
    print("Testing Stair-like Multigrade MAML Architecture")
    print("=" * 60)
    
    # Create the same config as original MAML
    config = [
        ('conv2d', [32, 2, 3, 3, 1, 1]),    # Grade 1: layers 0-1
        ('tanh', [True]),                    # Grade 1: layer 2
        ('bn', [32]),                        # Grade 1: layer 3
        ('conv2d', [64, 32, 3, 3, 1, 1]),   # Grade 2: layers 4-5
        ('tanh', [True]),                    # Grade 2: layer 6
        ('bn', [64]),                        # Grade 2: layer 7
        ('conv2d', [128, 64, 3, 3, 1, 1]),  # Grade 3: layers 8-9
        ('tanh', [True]),                    # Grade 3: layer 10
        ('bn', [128]),                       # Grade 3: layer 11
        ('conv2d', [64, 128, 3, 3, 1, 1]),  # Grade 3: layers 12-13
        ('tanh', [True]),                    # Grade 3: layer 14
        ('bn', [64]),                        # Grade 3: layer 15
        ('conv2d', [32, 64, 3, 3, 1, 1]),   # Grade 3: layers 16-17
        ('tanh', [True]),                    # Grade 3: layer 18
        ('bn', [32]),                        # Grade 3: layer 19
        ('conv2d', [2, 32, 3, 3, 1, 1])     # Grade 3: layers 20-21
    ]
    
    args = SimpleArgs()
    
    # Create multigrade MAML with stair architecture
    maml = MultigradeMAMLStair(args, config, num_grades=args.num_grades)
    print(f"✓ Created MultigradeMAMLStair with {args.num_grades} grades")
    
    # Print grade layer divisions
    print("\nGrade Layer Divisions:")
    for i, (start, end) in enumerate(maml.grade_layers):
        print(f"  Grade {i+1}: layers {start}-{end-1} ({end-start} layers)")
        for j in range(start, end):
            if j < len(config):
                print(f"    Layer {j}: {config[j][0]} - {config[j][1]}")
    
    # Count parameters per grade
    print("\nParameters per Grade:")
    for i in range(args.num_grades):
        grade_params = maml._get_grade_parameters(i)
        param_count = sum(p.numel() for p in grade_params)
        print(f"  Grade {i+1}: {param_count} parameters")
    
    # Create dummy data
    x_qry, y_qry, x_spt, y_spt = create_dummy_data()
    print("\n✓ Created dummy data")
    
    # Test training loop
    print("\nTesting training loop...")
    prev_losses = None
    
    for step in range(5):
        print(f"\nStep {step + 1}/5:")
        
        # Forward pass
        losses = maml(x_qry, y_qry, x_spt, y_spt)
        
        # Print losses for each grade
        current_losses = []
        for grade_idx in range(args.num_grades):
            grade_loss = losses[grade_idx][-1]
            print(f"  Grade {grade_idx + 1} loss: {grade_loss:.6f}")
            current_losses.append(grade_loss)
        
        # Check if losses are decreasing
        if prev_losses is not None:
            total_prev = sum(prev_losses)
            total_curr = sum(current_losses)
            change = total_curr - total_prev
            print(f"  Total loss change: {change:.6f}")
            
            if change < 0:
                print("  ✓ Losses are decreasing!")
            else:
                print("  ⚠ Losses not decreasing yet")
        
        prev_losses = current_losses
    
    print("\n✅ Stair-like Multigrade MAML test completed!")
    return True

def main():
    """Main test function"""
    try:
        test_stair_architecture()
        print("\n🎉 Stair architecture test passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
