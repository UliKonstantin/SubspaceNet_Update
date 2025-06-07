"""
Unit tests for ExtendedKalmanFilter1D implementation.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from simulation.kalman_filter.models import SineAccelStateModel, MultNoiseStateModel
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
from config.schema import TrajectoryType

class TestExtendedKalmanFilter1D(unittest.TestCase):
    """Test the ExtendedKalmanFilter1D implementation."""
    
    def setUp(self):
        """Set up a filter with a mock state model for testing."""
        # Create a mock state model
        self.state_model = MagicMock()
        self.state_model.f.return_value = 11.0  # Arbitrary prediction
        self.state_model.F_jacobian.return_value = 1.2  # Arbitrary Jacobian
        self.state_model.noise_variance.return_value = 0.25  # Arbitrary noise variance
        
        # Create the filter
        self.R = 0.1  # Measurement noise variance
        self.P0 = 1.0  # Initial state covariance
        self.filter = ExtendedKalmanFilter1D(
            state_model=self.state_model,
            R=self.R,
            P0=self.P0
        )
        
        # Initialize the state
        self.x0 = 10.0
        self.filter.initialize_state(self.x0)
    
    def test_initialization(self):
        """Test that filter parameters are correctly initialized."""
        self.assertEqual(self.filter.state_model, self.state_model)
        self.assertEqual(self.filter.R, self.R)
        self.assertEqual(self.filter.P0, self.P0)
        self.assertEqual(self.filter.x, self.x0)
        self.assertEqual(self.filter.P, self.P0)
    
    def test_predict(self):
        """Test the prediction step."""
        # Perform prediction
        x_pred = self.filter.predict()
        
        # Check that model methods were called correctly
        self.state_model.f.assert_called_once_with(self.x0)
        self.state_model.F_jacobian.assert_called_once_with(self.x0)
        self.state_model.noise_variance.assert_called_once_with(self.x0)
        
        # Check that prediction is correct
        self.assertEqual(x_pred, 11.0)
        
        # Check that state and covariance were updated correctly
        self.assertEqual(self.filter.x, 11.0)
        # P_pred = F * P * F + Q = 1.2 * 1.0 * 1.2 + 0.25 = 1.69
        self.assertAlmostEqual(self.filter.P, 1.69)
    
    def test_update(self):
        """Test the update step."""
        # Set up filter state after prediction
        self.filter.x = 11.0
        self.filter.P = 1.69
        
        # Perform update with measurement
        z = 12.0
        x_new = self.filter.update(z)
        
        # Calculate expected values
        y = z - 11.0  # Innovation
        S = 1.69 + 0.1  # Innovation covariance
        K = 1.69 / S  # Kalman gain
        expected_x = 11.0 + K * y  # Updated state
        expected_P = (1 - K) * 1.69  # Updated covariance
        
        # Check that update is correct
        self.assertAlmostEqual(x_new, expected_x)
        
        # Check that state and covariance were updated correctly
        self.assertAlmostEqual(self.filter.x, expected_x)
        self.assertAlmostEqual(self.filter.P, expected_P)
    
    @patch('simulation.kalman_filter.extended.SineAccelStateModel')
    @patch('simulation.kalman_filter.extended.MultNoiseStateModel')
    def test_from_config_sine_accel(self, mock_mult_noise, mock_sine_accel):
        """Test creation from config with sine acceleration model."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.SINE_ACCEL_NONLINEAR
        config.trajectory.sine_accel_omega0 = 0.1
        config.trajectory.sine_accel_kappa = 0.5
        config.trajectory.sine_accel_noise_std = 0.2
        config.kalman_filter.measurement_noise_std_dev = 0.1
        config.kalman_filter.initial_covariance = 2.0
        
        # Create mock SineAccelStateModel instance
        mock_model = MagicMock()
        mock_sine_accel.return_value = mock_model
        
        # Call from_config
        filter = ExtendedKalmanFilter1D.from_config(config)
        
        # Check that SineAccelStateModel was created with correct parameters
        mock_sine_accel.assert_called_once_with(
            omega0=0.1, 
            kappa=0.5, 
            noise_std=0.2
        )
        
        # Check that filter was created with correct parameters
        self.assertEqual(filter.state_model, mock_model)
        self.assertEqual(filter.R, 0.01)  # 0.1^2
        self.assertEqual(filter.P0, 2.0)
    
    @patch('simulation.kalman_filter.extended.SineAccelStateModel')
    @patch('simulation.kalman_filter.extended.MultNoiseStateModel')
    def test_from_config_mult_noise(self, mock_mult_noise, mock_sine_accel):
        """Test creation from config with multiplicative noise model."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.MULT_NOISE_NONLINEAR
        config.trajectory.mult_noise_omega0 = 0.2
        config.trajectory.mult_noise_amp = 0.5
        config.trajectory.mult_noise_base_std = 0.3
        config.kalman_filter.measurement_noise_std_dev = 0.1
        config.kalman_filter.initial_covariance = 2.0
        
        # Create mock MultNoiseStateModel instance
        mock_model = MagicMock()
        mock_mult_noise.return_value = mock_model
        
        # Call from_config
        filter = ExtendedKalmanFilter1D.from_config(config)
        
        # Check that MultNoiseStateModel was created with correct parameters
        mock_mult_noise.assert_called_once_with(
            omega0=0.2, 
            amp=0.5, 
            base_std=0.3
        )
        
        # Check that filter was created with correct parameters
        self.assertEqual(filter.state_model, mock_model)
        self.assertEqual(filter.R, 0.01)  # 0.1^2
        self.assertEqual(filter.P0, 2.0)
    
    @patch('simulation.kalman_filter.extended.SineAccelStateModel')
    def test_from_config_random_walk(self, mock_sine_accel):
        """Test creation from config with random walk model."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.RANDOM_WALK
        config.trajectory.random_walk_std_dev = 0.5
        config.kalman_filter.measurement_noise_std_dev = 0.1
        config.kalman_filter.initial_covariance = 2.0
        
        # Create mock SineAccelStateModel instance (used for random walk)
        mock_model = MagicMock()
        mock_sine_accel.return_value = mock_model
        
        # Call from_config
        filter = ExtendedKalmanFilter1D.from_config(config)
        
        # Check that SineAccelStateModel was created with correct parameters
        mock_sine_accel.assert_called_once_with(
            omega0=0.0, 
            kappa=0.0, 
            noise_std=0.5
        )
        
        # Check that filter was created with correct parameters
        self.assertEqual(filter.state_model, mock_model)
        self.assertEqual(filter.R, 0.01)  # 0.1^2
        self.assertEqual(filter.P0, 2.0)


if __name__ == '__main__':
    unittest.main() 