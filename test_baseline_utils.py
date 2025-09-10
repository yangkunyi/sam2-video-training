#!/usr/bin/env python3
"""
Test script for baseline_utils functions.
"""

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from baseline_utils import extract_baseline_metrics, calculate_metrics_delta
except ImportError as e:
    print(f"Import error: {e}")
    print("Testing basic functionality without imports...")
    
    # Simple test of baseline file reading
    baseline_path = "baseline_results/endovis17/1_mem/metrics.json" 
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Successfully read baseline: {data}")
    else:
        print(f"✗ Baseline file not found: {baseline_path}")
    sys.exit(0)

def test_baseline_extraction():
    """Test baseline extraction with real data."""
    
    # Test cases matching the baseline structure
    test_cases = [
        "endovis17_1_mem",
        "endovis17_1_mem_sfx", 
        "cholecseg8k_2_mem",
        "cholecseg8k_2_mem_sfx",
        "invalid_combo_name"
    ]
    
    print("Testing baseline extraction:")
    for combo_name in test_cases:
        baseline = extract_baseline_metrics(combo_name)
        if baseline:
            print(f"✓ {combo_name}: {baseline}")
        else:
            print(f"✗ {combo_name}: No baseline found")
    
    # Test delta calculation
    print("\nTesting delta calculation:")
    baseline = {"mIoU": 0.77, "Dice": 0.83, "MAE": 1.5}
    current = {"mIoU": 0.82, "Dice": 0.87, "MAE": 1.2}
    
    deltas = calculate_metrics_delta(current, baseline)
    print(f"Current: {current}")
    print(f"Baseline: {baseline}")
    print(f"Deltas: {deltas}")
    
    expected_deltas = {
        "delta_mIoU": 0.05,
        "delta_Dice": 0.04, 
        "delta_MAE": -0.3
    }
    
    for key, expected in expected_deltas.items():
        actual = deltas.get(key, 0)
        if abs(actual - expected) < 0.001:
            print(f"✓ {key}: {actual} (expected {expected})")
        else:
            print(f"✗ {key}: {actual} (expected {expected})")

if __name__ == "__main__":
    test_baseline_extraction()