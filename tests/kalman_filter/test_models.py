"""
Unit tests for state evolution models used in Extended Kalman Filter.
"""

import unittest
import numpy as np
from simulation.kalman_filter.models import SineAccelStateModel, MultNoiseStateModel

class TestSineAccelStateModel(unittest.TestCase):
    """Test the sine acceleration state evolution model."""
    
    def setUp(self):
        # Create a model with known parameters for testing
        self.omega0 = 0.1  # rad/s
        self.kappa = 0.5   # rad/s²
        self.noise_std = 0.1  # rad
        self.time_step = 1.0  # s
        
        self.model = SineAccelStateModel(
            omega0=self.omega0,
            kappa=self.kappa,
            noise_std=self.noise_std,
            time_step=self.time_step
        )
    
    def test_initialization(self):
        """Test that model parameters are correctly initialized."""
        self.assertEqual(self.model.omega0, self.omega0)
        self.assertEqual(self.model.kappa, self.kappa)
        self.assertEqual(self.model.base_noise_variance, self.noise_std**2)
        self.assertEqual(self.model.time_step, self.time_step)
    
    def test_state_transition(self):
        """Test the state transition function."""
        # For theta = 0, sin(theta) = 0, so f(theta) = theta + omega0*T
        x = 0.0
        expected = x + np.degrees(self.omega0 * self.time_step)
        actual = self.model.f(x)
        self.assertAlmostEqual(actual, expected)
        
        # For theta = 90, sin(theta) = 1, so f(theta) = theta + (omega0 + kappa)*T
        x = 90.0
        expected = x + np.degrees((self.omega0 + self.kappa) * self.time_step)
        actual = self.model.f(x)
        self.assertAlmostEqual(actual, expected)
    
    def test_jacobian(self):
        """Test the Jacobian calculation."""
        # For theta = 0, cos(theta) = 1, so jacobian = 1 + kappa*T*cos(theta)
        x = 0.0
        expected = 1 + self.kappa * np.cos(np.radians(x)) * self.time_step * np.pi/180.0
        actual = self.model.F_jacobian(x)
        self.assertAlmostEqual(actual, expected)
        
        # For theta = 90, cos(theta) = 0, so jacobian = 1
        x = 90.0
        expected = 1 + self.kappa * np.cos(np.radians(x)) * self.time_step * np.pi/180.0
        actual = self.model.F_jacobian(x)
        self.assertAlmostEqual(actual, expected)
    
    def test_noise_variance(self):
        """Test the noise variance."""
        # Base noise variance should be constant
        x = 0.0
        expected = self.noise_std**2
        actual = self.model.noise_variance(x)
        self.assertEqual(actual, expected)


class TestMultNoiseStateModel(unittest.TestCase):
    """Test the multiplicative noise state evolution model."""
    
    def setUp(self):
        # Create a model with known parameters for testing
        self.omega0 = 0.2  # rad/s
        self.amp = 0.5     # unitless
        self.base_std = 0.1  # rad
        self.time_step = 1.0  # s
        
        self.model = MultNoiseStateModel(
            omega0=self.omega0,
            amp=self.amp,
            base_std=self.base_std,
            time_step=self.time_step
        )
    
    def test_initialization(self):
        """Test that model parameters are correctly initialized."""
        self.assertEqual(self.model.omega0, self.omega0)
        self.assertEqual(self.model.amp, self.amp)
        self.assertEqual(self.model.base_std, self.base_std)
        self.assertEqual(self.model.base_noise_variance, self.base_std**2)
        self.assertEqual(self.model.time_step, self.time_step)
    
    def test_state_transition(self):
        """Test the state transition function."""
        # Deterministic part is always theta + omega0*T
        x = 30.0
        expected = x + np.degrees(self.omega0 * self.time_step)
        actual = self.model.f(x)
        self.assertAlmostEqual(actual, expected)
    
    def test_jacobian(self):
        """Test the Jacobian calculation."""
        # Jacobian is always 1 for this model
        x = 45.0
        expected = 1.0
        actual = self.model.F_jacobian(x)
        self.assertEqual(actual, expected)
    
    def test_noise_variance(self):
        """Test the state-dependent noise variance."""
        # For theta = 0, sin²(theta) = 0
        x = 0.0
        std = self.base_std * (1.0 + self.amp * np.sin(np.radians(x))**2)
        expected = std**2
        actual = self.model.noise_variance(x)
        self.assertAlmostEqual(actual, expected)
        
        # For theta = 90, sin²(theta) = 1
        x = 90.0
        std = self.base_std * (1.0 + self.amp * np.sin(np.radians(x))**2)
        expected = std**2
        actual = self.model.noise_variance(x)
        self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main() 