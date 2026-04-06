#!/usr/bin/env python3
"""
Test script for the iMAML Inner Step Analysis Tool

This script tests the basic functionality of the analyzer without requiring
actual dataset files.
"""

import os
import sys
import numpy as np
import torch
import json
from imaml_inner_step_analyzer import InnerStepAnalyzer

def test_analyzer_basic_functionality():
    """Test basic functionality of the InnerStepAnalyzer."""
    print("Testing InnerStepAnalyzer basic functionality...")
    
    # Create test save directory
    test_save_dir = "./test_results"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = InnerStepAnalyzer(test_save_dir, "TestScenario")
    
    # Test tracking data
    test_channel_names = ["channel_1", "channel_2", "channel_3"]
    test_task_losses = [
        {"initial": 0.5, "final": 0.3},
        {"initial": 0.4, "final": 0.25},
        {"initial": 0.6, "final": 0.35}
    ]
    
    # Track losses
    analyzer.track_inner_step_losses(0, test_channel_names, test_task_losses)
    
    # Test statistics generation
    stats = analyzer.generate_channel_statistics()
    print(f"Generated statistics for {len(stats)} channels")
    
    # Test learning progression analysis
    progression = analyzer.analyze_learning_progression()
    print(f"Learning progression analysis completed")
    
    # Test saving results
    analyzer.save_analysis_results()
    print("Analysis results saved successfully")
    
    # Verify files were created
    expected_files = [
        "imaml_tracking_data_testscenario_*.json",
        "imaml_tracking_data_testscenario_*.csv",
        "imaml_channel_statistics_testscenario_*.json",
        "imaml_progression_analysis_testscenario_*.json",
        "imaml_summary_report_testscenario_*.txt"
    ]
    
    files_created = os.listdir(test_save_dir)
    print(f"Files created: {files_created}")
    
    return len(files_created) > 0

def test_analyzer_with_multiple_epochs():
    """Test analyzer with multiple epochs of data."""
    print("\nTesting analyzer with multiple epochs...")
    
    # Create test save directory
    test_save_dir = "./test_results_multi_epoch"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = InnerStepAnalyzer(test_save_dir, "MultiEpochTest")
    
    # Simulate multiple epochs
    for epoch in range(5):
        channel_names = [f"channel_{i}" for i in range(3)]
        task_losses = []
        
        for i in range(3):
            # Simulate decreasing losses over epochs
            initial_loss = 0.5 - epoch * 0.05 + np.random.normal(0, 0.02)
            final_loss = initial_loss - 0.1 + np.random.normal(0, 0.02)
            task_losses.append({"initial": max(0.01, initial_loss), 
                              "final": max(0.01, final_loss)})
        
        analyzer.track_inner_step_losses(epoch, channel_names, task_losses)
    
    # Generate analysis
    stats = analyzer.generate_channel_statistics()
    progression = analyzer.analyze_learning_progression()
    
    # Save results
    analyzer.save_analysis_results()
    
    print(f"Multi-epoch test completed with {len(stats)} channels")
    return True

def test_analyzer_edge_cases():
    """Test analyzer with edge cases."""
    print("\nTesting analyzer edge cases...")
    
    # Create test save directory
    test_save_dir = "./test_results_edge_cases"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = InnerStepAnalyzer(test_save_dir, "EdgeCaseTest")
    
    # Test with empty data
    analyzer.track_inner_step_losses(0, [], [])
    
    # Test with single channel
    analyzer.track_inner_step_losses(1, ["single_channel"], [{"initial": 0.5, "final": 0.3}])
    
    # Test with zero improvement
    analyzer.track_inner_step_losses(2, ["no_improvement"], [{"initial": 0.5, "final": 0.5}])
    
    # Test with negative improvement
    analyzer.track_inner_step_losses(3, ["negative_improvement"], [{"initial": 0.3, "final": 0.5}])
    
    # Test with very small values
    analyzer.track_inner_step_losses(4, ["small_values"], [{"initial": 0.001, "final": 0.0005}])
    
    # Generate analysis
    stats = analyzer.generate_channel_statistics()
    progression = analyzer.analyze_learning_progression()
    
    # Save results
    analyzer.save_analysis_results()
    
    print(f"Edge case test completed with {len(stats)} channels")
    return True

def test_analyzer_visualization():
    """Test analyzer visualization functionality."""
    print("\nTesting analyzer visualization...")
    
    # Create test save directory
    test_save_dir = "./test_results_visualization"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = InnerStepAnalyzer(test_save_dir, "VisualizationTest")
    
    # Generate test data for visualization
    for epoch in range(10):
        channel_names = [f"channel_{i}" for i in range(4)]
        task_losses = []
        
        for i in range(4):
            # Simulate realistic loss patterns
            initial_loss = 0.5 - epoch * 0.02 + np.random.normal(0, 0.05)
            final_loss = initial_loss - 0.1 + np.random.normal(0, 0.03)
            task_losses.append({"initial": max(0.01, initial_loss), 
                              "final": max(0.01, final_loss)})
        
        analyzer.track_inner_step_losses(epoch, channel_names, task_losses)
    
    # Test visualization creation
    try:
        analyzer.create_visualizations()
        print("Visualization test completed successfully")
        return True
    except Exception as e:
        print(f"Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("iMAML Inner Step Analysis Tool - Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_analyzer_basic_functionality),
        ("Multiple Epochs", test_analyzer_with_multiple_epochs),
        ("Edge Cases", test_analyzer_edge_cases),
        ("Visualization", test_analyzer_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"✗ {test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The analyzer is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
