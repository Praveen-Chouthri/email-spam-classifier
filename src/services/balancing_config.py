"""
Configuration management system for class balancing operations.

This module provides configuration management with environment variable support,
validation, and default value handling for class balancing operations.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


@dataclass
class BalancingConfig:
    """
    Configuration for class balancing operations with environment variable support.
    
    This class provides comprehensive configuration management for class balancing,
    including SMOTE parameters, class weighting options, and validation settings.
    """
    
    # Core balancing settings
    enabled: bool = field(default_factory=lambda: _get_env_bool('ENABLE_CLASS_BALANCING', True))
    method: str = field(default_factory=lambda: _get_env_str('BALANCING_METHOD', 'both'))  # Use both SMOTE and class weights
    target_spam_ratio: float = field(default_factory=lambda: _get_env_float('TARGET_SPAM_RATIO', 0.48))  # More aggressive ratio
    
    # SMOTE-specific settings
    smote_k_neighbors: int = field(default_factory=lambda: _get_env_int('SMOTE_K_NEIGHBORS', 3))  # Reduce k for more diverse samples
    smote_random_state: int = field(default_factory=lambda: _get_env_int('SMOTE_RANDOM_STATE', 42))
    
    # Class weighting settings
    class_weight_strategy: str = field(default_factory=lambda: _get_env_str('CLASS_WEIGHT_STRATEGY', 'balanced'))
    custom_weights: Optional[Dict[int, float]] = None
    
    # Advanced settings
    fallback_to_class_weights: bool = field(default_factory=lambda: _get_env_bool('FALLBACK_TO_CLASS_WEIGHTS', True))
    validate_synthetic_samples: bool = field(default_factory=lambda: _get_env_bool('VALIDATE_SYNTHETIC_SAMPLES', True))
    min_samples_for_smote: int = field(default_factory=lambda: _get_env_int('MIN_SAMPLES_FOR_SMOTE', 10))
    
    # Performance settings
    batch_size_for_large_datasets: int = field(default_factory=lambda: _get_env_int('BALANCING_BATCH_SIZE', 10000))
    memory_limit_mb: int = field(default_factory=lambda: _get_env_int('BALANCING_MEMORY_LIMIT_MB', 1024))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        logger = logging.getLogger(__name__)
        
        # Validate method
        valid_methods = ['smote', 'class_weights', 'both']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid balancing method '{self.method}'. Must be one of: {valid_methods}")
        
        # Validate target spam ratio
        if not isinstance(self.target_spam_ratio, (int, float)):
            raise ValueError("target_spam_ratio must be a number")
        
        if not (0.3 <= self.target_spam_ratio <= 0.5):
            raise ValueError("target_spam_ratio must be between 0.3 and 0.5")
        
        # Validate SMOTE parameters
        if self.smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors must be >= 1")
        
        if self.smote_k_neighbors > 20:
            logger.warning(f"smote_k_neighbors is high ({self.smote_k_neighbors}), may cause performance issues")
        
        # Validate class weight strategy
        valid_strategies = ['balanced', 'custom']
        if self.class_weight_strategy not in valid_strategies:
            raise ValueError(f"Invalid class_weight_strategy '{self.class_weight_strategy}'. Must be one of: {valid_strategies}")
        
        # Validate custom weights if provided
        if self.class_weight_strategy == 'custom':
            if not self.custom_weights:
                raise ValueError("custom_weights must be provided when class_weight_strategy is 'custom'")
            
            if not isinstance(self.custom_weights, dict):
                raise ValueError("custom_weights must be a dictionary")
            
            # Check that weights are for classes 0 and 1
            required_classes = {0, 1}
            provided_classes = set(self.custom_weights.keys())
            if provided_classes != required_classes:
                raise ValueError(f"custom_weights must contain weights for classes {required_classes}")
        
        # Validate performance settings
        if self.min_samples_for_smote < 5:
            raise ValueError("min_samples_for_smote must be >= 5")
        
        if self.batch_size_for_large_datasets < 1000:
            logger.warning(f"batch_size_for_large_datasets is low ({self.batch_size_for_large_datasets})")
        
        if self.memory_limit_mb < 256:
            logger.warning(f"memory_limit_mb is low ({self.memory_limit_mb}MB)")
        
        logger.info("BalancingConfig validation completed successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.info(f"Class balancing configuration loaded:")
        logger.info(f"  - Enabled: {self.enabled}")
        logger.info(f"  - Method: {self.method}")
        logger.info(f"  - Target spam ratio: {self.target_spam_ratio}")
        logger.info(f"  - SMOTE k-neighbors: {self.smote_k_neighbors}")
        logger.info(f"  - Class weight strategy: {self.class_weight_strategy}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'enabled': self.enabled,
            'method': self.method,
            'target_spam_ratio': self.target_spam_ratio,
            'smote_k_neighbors': self.smote_k_neighbors,
            'smote_random_state': self.smote_random_state,
            'class_weight_strategy': self.class_weight_strategy,
            'custom_weights': self.custom_weights,
            'fallback_to_class_weights': self.fallback_to_class_weights,
            'validate_synthetic_samples': self.validate_synthetic_samples,
            'min_samples_for_smote': self.min_samples_for_smote,
            'batch_size_for_large_datasets': self.batch_size_for_large_datasets,
            'memory_limit_mb': self.memory_limit_mb
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BalancingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            BalancingConfig instance
        """
        # Filter out None values and unknown keys
        valid_keys = {
            'enabled', 'method', 'target_spam_ratio', 'smote_k_neighbors', 
            'smote_random_state', 'class_weight_strategy', 'custom_weights',
            'fallback_to_class_weights', 'validate_synthetic_samples',
            'min_samples_for_smote', 'batch_size_for_large_datasets', 'memory_limit_mb'
        }
        
        filtered_dict = {k: v for k, v in config_dict.items() 
                        if k in valid_keys and v is not None}
        
        return cls(**filtered_dict)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        logger = logging.getLogger(__name__)
        logger.info("Updating configuration from environment variables...")
        
        # Update each field from environment if available
        self.enabled = _get_env_bool('ENABLE_CLASS_BALANCING', self.enabled)
        self.method = _get_env_str('BALANCING_METHOD', self.method)
        self.target_spam_ratio = _get_env_float('TARGET_SPAM_RATIO', self.target_spam_ratio)
        self.smote_k_neighbors = _get_env_int('SMOTE_K_NEIGHBORS', self.smote_k_neighbors)
        self.smote_random_state = _get_env_int('SMOTE_RANDOM_STATE', self.smote_random_state)
        self.class_weight_strategy = _get_env_str('CLASS_WEIGHT_STRATEGY', self.class_weight_strategy)
        self.fallback_to_class_weights = _get_env_bool('FALLBACK_TO_CLASS_WEIGHTS', self.fallback_to_class_weights)
        self.validate_synthetic_samples = _get_env_bool('VALIDATE_SYNTHETIC_SAMPLES', self.validate_synthetic_samples)
        self.min_samples_for_smote = _get_env_int('MIN_SAMPLES_FOR_SMOTE', self.min_samples_for_smote)
        self.batch_size_for_large_datasets = _get_env_int('BALANCING_BATCH_SIZE', self.batch_size_for_large_datasets)
        self.memory_limit_mb = _get_env_int('BALANCING_MEMORY_LIMIT_MB', self.memory_limit_mb)
        
        # Re-validate after update
        self._validate_config()


class BalancingConfigManager:
    """
    Manager class for handling balancing configuration across the application.
    
    This class provides centralized configuration management with support for
    different configuration sources and runtime updates.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.logger = logging.getLogger(__name__)
        self._config: Optional[BalancingConfig] = None
        self._config_file_path: Optional[str] = None
    
    def get_config(self) -> BalancingConfig:
        """
        Get the current balancing configuration.
        
        Returns:
            BalancingConfig instance
        """
        if self._config is None:
            self._config = BalancingConfig()
        
        return self._config
    
    def set_config(self, config: BalancingConfig) -> None:
        """
        Set a new balancing configuration.
        
        Args:
            config: New BalancingConfig instance
        """
        self._config = config
        self.logger.info("Balancing configuration updated")
    
    def load_from_file(self, file_path: str) -> BalancingConfig:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            BalancingConfig instance
        """
        import json
        
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            self._config = BalancingConfig.from_dict(config_dict)
            self._config_file_path = file_path
            
            self.logger.info(f"Configuration loaded from {file_path}")
            return self._config
            
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {file_path}")
            self._config = BalancingConfig()
            return self._config
        
        except Exception as e:
            self.logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            self._config = BalancingConfig()
            return self._config
    
    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration (optional)
        """
        import json
        
        if self._config is None:
            self.logger.warning("No configuration to save")
            return
        
        save_path = file_path or self._config_file_path
        if not save_path:
            raise ValueError("No file path specified for saving configuration")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {save_path}: {str(e)}")
    
    def reload_from_env(self) -> None:
        """Reload configuration from environment variables."""
        if self._config is None:
            self._config = BalancingConfig()
        else:
            self._config.update_from_env()
        
        self.logger.info("Configuration reloaded from environment variables")


# Global configuration manager instance
_config_manager = BalancingConfigManager()


def get_balancing_config() -> BalancingConfig:
    """
    Get the global balancing configuration.
    
    Returns:
        BalancingConfig instance
    """
    return _config_manager.get_config()


def set_balancing_config(config: BalancingConfig) -> None:
    """
    Set the global balancing configuration.
    
    Args:
        config: New BalancingConfig instance
    """
    _config_manager.set_config(config)


def load_balancing_config_from_file(file_path: str) -> BalancingConfig:
    """
    Load balancing configuration from file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        BalancingConfig instance
    """
    return _config_manager.load_from_file(file_path)


# Helper functions for environment variable parsing
def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    return value.lower() in ('true', '1', 'yes', 'on')


def _get_env_str(key: str, default: str) -> str:
    """Get string value from environment variable."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return int(value)
    except ValueError:
        logging.getLogger(__name__).warning(f"Invalid integer value for {key}: {value}, using default: {default}")
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return float(value)
    except ValueError:
        logging.getLogger(__name__).warning(f"Invalid float value for {key}: {value}, using default: {default}")
        return default