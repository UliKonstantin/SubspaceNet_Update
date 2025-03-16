#!/usr/bin/env python3
"""
Test script for validating the SubspaceNet factory module.

This script tests that the factory correctly generates objects from the DCD-MUSIC module
and that the configurations align with expectations.
"""

import os
import sys
import unittest
import yaml
import torch
import warnings

# Add the root directory to the path so we can import our modules
sys.path.append('.')

try:
    from config.schema import Config
    from config.loader import load_config
    from config.factory import create_components_from_config
    print("Successfully imported SubspaceNet config modules")
except ImportError as e:
    print(f"Error importing SubspaceNet modules: {e}")
    sys.exit(1)

class TestFactory(unittest.TestCase):
    """Test cases for the factory module."""

    def setUp(self):
        """Set up the test environment."""
        print("\nSetting up test environment...")
        
        # Check if default config exists, create if not
        self.config_path = "configs/default_config.yaml"
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        # Load default configuration
        try:
            self.config = load_config(self.config_path)
            print(f"Successfully loaded configuration from {self.config_path}")
        except Exception as e:
            self.fail(f"Failed to load configuration: {e}")
        
        # Create a modified config for SubspaceNet
        self.subspacenet_config = load_config(self.config_path)
        self.subspacenet_config.model.type = "SubspaceNet"
        self.subspacenet_config.model.params.diff_method = "music_2D"
        self.subspacenet_config.model.params.train_loss_type = "music_spectrum"
        self.subspacenet_config.model.params.tau = 8
        self.subspacenet_config.model.params.field_type = "Near"
        
        # Create a modified config for DCD-MUSIC
        self.dcdmusic_config = load_config(self.config_path)
        self.dcdmusic_config.model.type = "DCD-MUSIC"
        self.dcdmusic_config.model.params.diff_method = ("esprit", "music_1d")
        self.dcdmusic_config.model.params.train_loss_type = ("rmspe", "rmspe")
        self.dcdmusic_config.model.params.tau = 8
        self.dcdmusic_config.model.params.variant = "small"
        
        print("Test environment setup complete")

    def _create_default_config(self):
        """Create a default configuration file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        default_config = {
            "system_model": {
                "N": 127,
                "M": 2,
                "T": 100,
                "field_type": "near",
                "signal_type": "narrowband",
                "signal_nature": "non-coherent",
                "snr": 10,
                "wavelength": 0.06
            },
            "model": {
                "type": "SubspaceNet",
                "params": {
                    "tau": 8,
                    "diff_method": "music_2D",
                    "train_loss_type": "music_spectrum",
                    "field_type": "Near",
                    "regularization": None,
                    "variant": "small",
                    "norm_layer": False
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        print(f"Created default configuration at {self.config_path}")

    def test_create_system_model(self):
        """Test creating a system model from configuration."""
        print("\nTesting system model creation...")
        components = create_components_from_config(self.config)
        
        # Check if components were created successfully
        self.assertIn("system_model", components, "System model was not created")
        system_model = components["system_model"]
        
        # Verify system_model has the correct parameters
        self.assertEqual(system_model.params.N, self.config.system_model.N, 
                         "System model has incorrect N parameter")
        self.assertEqual(system_model.params.M, self.config.system_model.M,
                         "System model has incorrect M parameter")
        self.assertEqual(system_model.params.T, self.config.system_model.T,
                         "System model has incorrect T parameter")
        self.assertEqual(system_model.params.snr, self.config.system_model.snr,
                         "System model has incorrect SNR parameter")
        
        # Check if the system model has expected methods
        self.assertTrue(hasattr(system_model, "steering_vec"), 
                        "System model does not have steering_vec method")
        self.assertTrue(hasattr(system_model, "create_array"), 
                        "System model does not have create_array method")
        
        print("System model creation test passed")

    def test_create_subspacenet_model(self):
        """Test creating a SubspaceNet model from configuration."""
        print("\nTesting SubspaceNet model creation...")
        components = create_components_from_config(self.subspacenet_config)
        
        # Verify model was created
        self.assertIn("model", components, "Model was not created")
        model = components["model"]
        
        # Verify model has the correct parameters
        self.assertEqual(model.tau, self.subspacenet_config.model.params.tau,
                         "Model has incorrect tau parameter")
        
        # Check if model name contains "SubspaceNet" substring
        model_name = model.get_model_name()
        self.assertIn("SubspaceNet", model_name, 
                      f"Model name '{model_name}' does not contain 'SubspaceNet'")
        
        # Check model structure and methods
        self.assertTrue(hasattr(model, "forward"), "Model does not have forward method")
        self.assertTrue(hasattr(model, "training_step"), "Model does not have training_step method")
        
        print("SubspaceNet model creation test passed")

    def test_create_dcdmusic_model(self):
        """Test creating a DCD-MUSIC model from configuration."""
        print("\nTesting DCD-MUSIC model creation...")
        components = create_components_from_config(self.dcdmusic_config)
        
        # Verify model was created
        self.assertIn("model", components, "Model was not created")
        model = components["model"]
        
        # Verify model has the correct parameters
        self.assertEqual(model.tau, self.dcdmusic_config.model.params.tau,
                         "Model has incorrect tau parameter")
        
        # Check if model name contains "DCDMUSIC" substring
        model_name = model.get_model_name()
        self.assertIn("DCDMUSIC", model_name, 
                      f"Model name '{model_name}' does not contain 'DCDMUSIC'")
        
        # Check model structure and methods
        self.assertTrue(hasattr(model, "forward"), "Model does not have forward method")
        self.assertTrue(hasattr(model, "training_step"), "Model does not have training_step method")
        
        print("DCD-MUSIC model creation test passed")

    def test_model_factory_with_different_configs(self):
        """Test that the factory creates different models based on configuration."""
        print("\nTesting model factory with different configurations...")
        
        # Create components with SubspaceNet config
        subspacenet_components = create_components_from_config(self.subspacenet_config)
        
        # Create components with DCD-MUSIC config
        dcdmusic_components = create_components_from_config(self.dcdmusic_config)
        
        # Verify different models were created with correct names
        subspacenet_model_name = subspacenet_components["model"].get_model_name()
        dcdmusic_model_name = dcdmusic_components["model"].get_model_name()
        
        self.assertIn("SubspaceNet", subspacenet_model_name,
                      f"SubspaceNet model name '{subspacenet_model_name}' does not contain 'SubspaceNet'")
        self.assertIn("DCDMUSIC", dcdmusic_model_name,
                      f"DCD-MUSIC model name '{dcdmusic_model_name}' does not contain 'DCDMUSIC'")
        
        # Verify the models are different
        self.assertNotEqual(subspacenet_model_name, dcdmusic_model_name,
                           "SubspaceNet and DCD-MUSIC models have identical names")
        
        print("Model factory test with different configurations passed")


if __name__ == "__main__":
    unittest.main(verbosity=2) 