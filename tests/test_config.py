import pytest
from pydantic import ValidationError
from src.core.config import TrainingConfig, SyntheticDataConfig

def test_training_config_defaults() -> None:
    """
    Tests if the TrainingConfig initializes with correct default values.
    """
    config = TrainingConfig()
    assert config.model_id == "unsloth/llama-3-8b-bnb-4bit"
    assert config.load_in_4bit is True
    assert config.lora_r == 16

def test_synthetic_data_config_requires_api_key() -> None:
    """
    Tests that initializing SyntheticDataConfig without an API key raises a validation error.
    """
    with pytest.raises(ValidationError):
        # API key is missing, should raise error
        SyntheticDataConfig() 

def test_synthetic_data_config_valid_key() -> None:
    """
    Tests successful initialization when a valid API key is provided.
    """
    config = SyntheticDataConfig(api_key="fake_test_key_123")
    assert config.api_key.get_secret_value() == "fake_test_key_123"
