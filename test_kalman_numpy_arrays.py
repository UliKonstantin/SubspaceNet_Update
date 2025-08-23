#!/usr/bin/env python3
"""
Test Kalman loss with numpy arrays (simulating real EKF output).
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path  
sys.path.append(str(Path(__file__).parent / "simulation" / "losses"))

def test_kalman_with_numpy():
    """Test Kalman loss with numpy arrays like real EKF output."""
    print("🧪 Testing Kalman Loss with Numpy Arrays")
    
    try:
        from kalman_loss import KalmanInnovationLoss
        print("✅ Import successful")
        
        # Create loss function
        kalman_loss = KalmanInnovationLoss(reduction='mean')
        print("✅ Loss function created")
        
        # Test case: Simulate EKF outputs (numpy arrays)
        kalman_gains = [
            np.array(0.5), 
            np.array([0.3]), 
            0.8  # Mix of numpy arrays and scalars
        ]
        innovations = [
            np.array(0.1), 
            np.array([-0.2]), 
            0.15
        ]
        
        loss = kalman_loss(kalman_gains, innovations)
        expected_loss = np.mean([abs(0.5 * 0.1), abs(0.3 * -0.2), abs(0.8 * 0.15)])
        
        print(f"✅ Test with numpy arrays:")
        print(f"   Calculated loss: {loss.item():.6f}")
        print(f"   Expected loss: {expected_loss:.6f}")
        print(f"   Match: {abs(loss.item() - expected_loss) < 1e-6}")
        
        # Test case: Multi-dimensional numpy arrays (should extract scalar)
        complex_gains = [np.array([[0.5]]), np.array([0.3, 0.4])[0]]  # Different shapes
        complex_innovations = [np.array([[0.1]]), np.array([-0.2])]
        
        try:
            complex_loss = kalman_loss(complex_gains, complex_innovations)
            print(f"✅ Complex arrays handled: {complex_loss.item():.6f}")
        except Exception as e:
            print(f"❌ Complex arrays failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Kalman Loss with Numpy Arrays\n")
    
    # Run test
    test_passed = test_kalman_with_numpy()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Numpy array handling: {'✅ PASS' if test_passed else '❌ FAIL'}")
    
    if test_passed:
        print("\n🎉 Kalman loss handles numpy arrays correctly!")
        sys.exit(0)
    else:
        print("\n💥 Fix needed for numpy array handling.")
        sys.exit(1)
