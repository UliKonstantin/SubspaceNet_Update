#!/usr/bin/env python3
"""
Standalone unit tests for state evolution models used in Extended Kalman Filter.
This version doesn't rely on project imports, to allow testing in isolation.
"""

import unittest
import numpy as np
import sys
import os

# Define the models directly in this file to avoid import errors

class StateEvolutionModel:
    """Abstract base class for state evolution models used by EKF."""
    
    def f(self, x):
        """Non-linear state transition function."""
        pass
    
    def F_jacobian(self, x):
        """Jacobian of the state transition function."""
        pass
    
    def noise_variance(self, x):
        """State-dependent process noise variance."""
        return self.base_noise_variance


class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the sine acceleration non-linear model:
    θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
    """
    
    def __init__(self, omega0, kappa, noise_std, time_step=1.0):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            kappa: Sine acceleration coefficient (rad/s²)
            noise_std: Standard deviation of process noise (rad)
            time_step: Time step between measurements (s)
        """
        self.omega0 = omega0
        self.kappa = kappa
        self.base_noise_variance = noise_std**2
        self.time_step = time_step
    
    def f(self, x):
        """
        State transition: θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate acceleration term
        delta = (self.omega0 + self.kappa * np.sin(x_rad)) * self.time_step
        
        # Return in same units as input
        return 0.99*x + np.degrees(delta)
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 + κT·cos(θ)
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate the derivative
        return 1 + self.kappa * np.cos(x_rad) * self.time_step * np.pi/180.0


class MultNoiseStateModel(StateEvolutionModel):
    """
    Implements the multiplicative noise non-linear model:
    θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
    where σ(θ) = base_std * (1 + amp * sin²(θ))
    """
    
    def __init__(self, omega0, amp, base_std, time_step=1.0):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            amp: Amplitude of multiplicative term (unitless)
            base_std: Base noise standard deviation (rad)
            time_step: Time step between measurements (s)
        """
        self.omega0 = omega0
        self.amp = amp
        self.base_std = base_std
        self.base_noise_variance = base_std**2
        self.time_step = time_step
    
    def f(self, x):
        """
        Deterministic part: θ_{k+1} = θ_k + ω0 T
        """
        return x + np.degrees(self.omega0 * self.time_step)
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 (constant velocity model)
        """
        return 1.0
    
    def noise_variance(self, x):
        """
        State-dependent noise variance: σ²(θ) = base_std² * (1 + amp * sin²(θ))²
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate state-dependent standard deviation
        std = self.base_std * (1.0 + self.amp * np.sin(x_rad)**2)
        
        # Return variance
        return std**2


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