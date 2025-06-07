"""
Unit tests for helper functions in the kalman_filter package.
"""

import unittest
from unittest.mock import MagicMock, patch
from config.schema import TrajectoryType
from simulation.kalman_filter import get_kalman_filter
from simulation.kalman_filter.base import KalmanFilter1D
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D

class TestHelperFunctions(unittest.TestCase):
    """Test the helper functions in the kalman_filter package."""
    
    @patch('simulation.kalman_filter.ExtendedKalmanFilter1D.from_config')
    @patch('simulation.kalman_filter.KalmanFilter1D.from_config')
    def test_get_kalman_filter_standard(self, mock_kf_from_config, mock_ekf_from_config):
        """Test get_kalman_filter with standard filter type."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.RANDOM_WALK
        config.kalman_filter.filter_type = "standard"
        
        # Create mock filter instances
        mock_kf = MagicMock(spec=KalmanFilter1D)
        mock_kf_from_config.return_value = mock_kf
        
        # Call get_kalman_filter
        filter = get_kalman_filter(config)
        
        # Check that standard KF was created
        mock_kf_from_config.assert_called_once_with(config)
        mock_ekf_from_config.assert_not_called()
        self.assertEqual(filter, mock_kf)
    
    @patch('simulation.kalman_filter.ExtendedKalmanFilter1D.from_config')
    @patch('simulation.kalman_filter.KalmanFilter1D.from_config')
    def test_get_kalman_filter_extended(self, mock_kf_from_config, mock_ekf_from_config):
        """Test get_kalman_filter with extended filter type."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.RANDOM_WALK
        config.kalman_filter.filter_type = "extended"
        
        # Create mock filter instances
        mock_ekf = MagicMock(spec=ExtendedKalmanFilter1D)
        mock_ekf_from_config.return_value = mock_ekf
        
        # Call get_kalman_filter
        filter = get_kalman_filter(config)
        
        # Check that extended KF was created
        mock_ekf_from_config.assert_called_once_with(config, TrajectoryType.RANDOM_WALK)
        mock_kf_from_config.assert_not_called()
        self.assertEqual(filter, mock_ekf)
    
    @patch('simulation.kalman_filter.ExtendedKalmanFilter1D.from_config')
    @patch('simulation.kalman_filter.KalmanFilter1D.from_config')
    def test_get_kalman_filter_nonlinear_trajectory(self, mock_kf_from_config, mock_ekf_from_config):
        """Test get_kalman_filter with non-linear trajectory type."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.SINE_ACCEL_NONLINEAR
        # No filter_type specified, should use extended based on trajectory type
        
        # Create mock filter instances
        mock_ekf = MagicMock(spec=ExtendedKalmanFilter1D)
        mock_ekf_from_config.return_value = mock_ekf
        
        # Call get_kalman_filter
        filter = get_kalman_filter(config)
        
        # Check that extended KF was created
        mock_ekf_from_config.assert_called_once_with(config, TrajectoryType.SINE_ACCEL_NONLINEAR)
        mock_kf_from_config.assert_not_called()
        self.assertEqual(filter, mock_ekf)
    
    @patch('simulation.kalman_filter.ExtendedKalmanFilter1D.from_config')
    @patch('simulation.kalman_filter.KalmanFilter1D.from_config')
    def test_get_kalman_filter_with_trajectory_type_param(self, mock_kf_from_config, mock_ekf_from_config):
        """Test get_kalman_filter with trajectory_type parameter."""
        # Create mock config
        config = MagicMock()
        config.trajectory.trajectory_type = TrajectoryType.RANDOM_WALK
        
        # Create mock filter instances
        mock_ekf = MagicMock(spec=ExtendedKalmanFilter1D)
        mock_ekf_from_config.return_value = mock_ekf
        
        # Call get_kalman_filter with explicit trajectory type
        filter = get_kalman_filter(config, TrajectoryType.SINE_ACCEL_NONLINEAR)
        
        # Check that extended KF was created with the provided trajectory type
        mock_ekf_from_config.assert_called_once_with(config, TrajectoryType.SINE_ACCEL_NONLINEAR)
        mock_kf_from_config.assert_not_called()
        self.assertEqual(filter, mock_ekf)


if __name__ == '__main__':
    unittest.main() 