"""
Unit tests for NeMo Gym integration.
"""

import pytest
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

from slime.rollout.nemo_gym import (
    NemoGymConfig,
    NemoGymEnvironmentConfig,
    NemoGymEnvironment,
    sample_to_openai_format,
    openai_response_to_sample,
)
from slime.training.unified_rlvr import (
    UnifiedRLVRConfig,
    UnifiedRLVRTrainer,
    create_unified_rlvr_trainer,
)
from slime.utils.types import Sample


class TestNemoGymConfig:
    """Tests for NemoGymConfig."""

    def test_environment_config_creation(self):
        """Test creating an environment config."""
        config = NemoGymEnvironmentConfig(
            name="math",
            config_path="/path/to/config.yaml",
            weight=0.5,
        )
        assert config.name == "math"
        assert config.weight == 0.5

    def test_environment_config_negative_weight_raises(self):
        """Test that negative weights raise an error."""
        with pytest.raises(ValueError):
            NemoGymEnvironmentConfig(
                name="math",
                config_path="/path/to/config.yaml",
                weight=-0.5,
            )

    def test_nemo_gym_config_creation(self):
        """Test creating a NemoGymConfig."""
        config = NemoGymConfig(
            resource_server_url="http://localhost:8080",
            environments=[
                NemoGymEnvironmentConfig("math", "/path/math.yaml", 0.5),
                NemoGymEnvironmentConfig("coding", "/path/coding.yaml", 0.5),
            ],
        )
        assert config.resource_server_url == "http://localhost:8080"
        assert len(config.environments) == 2

    def test_get_environment_weights(self):
        """Test getting normalized environment weights."""
        config = NemoGymConfig(
            resource_server_url="http://localhost:8080",
            environments=[
                NemoGymEnvironmentConfig("math", "/path/math.yaml", 0.4),
                NemoGymEnvironmentConfig("coding", "/path/coding.yaml", 0.6),
            ],
        )
        weights = config.get_environment_weights()
        assert abs(weights["math"] - 0.4) < 1e-6
        assert abs(weights["coding"] - 0.6) < 1e-6

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            nemo_gym_config=None,
            nemo_gym_resource_server_url="http://localhost:9000",
            nemo_gym_policy_model_url="http://localhost:30000",
            nemo_gym_on_policy_fix=True,
        )
        config = NemoGymConfig.from_args(args)
        assert config.resource_server_url == "http://localhost:9000"
        assert config.policy_model_url == "http://localhost:30000"


class TestSampleConversion:
    """Tests for sample conversion utilities."""

    def test_sample_to_openai_format(self):
        """Test converting sample to OpenAI format."""
        sample = Sample(
            prompt="Hello, how are you?",
            response="I'm doing well, thank you!",
            metadata={"key": "value"},
        )
        result = sample_to_openai_format(sample)
        
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_openai_response_to_sample(self):
        """Test converting OpenAI response back to sample."""
        original = Sample(prompt="Hello")
        response = {
            "choices": [
                {"message": {"content": "Hi there!"}}
            ],
            "reward": 1.0,
        }
        
        result = openai_response_to_sample(response, original)
        assert result.response == "Hi there!"
        assert result.reward == 1.0


class TestUnifiedRLVRConfig:
    """Tests for UnifiedRLVRConfig."""

    def test_config_creation(self):
        """Test creating UnifiedRLVRConfig."""
        config = UnifiedRLVRConfig(
            environments=["math", "coding"],
            environment_weights={"math": 0.6, "coding": 0.4},
        )
        assert len(config.environments) == 2
        assert config.environment_weights["math"] == 0.6

    def test_get_normalized_weights(self):
        """Test getting normalized weights."""
        config = UnifiedRLVRConfig(
            environments=["math", "coding"],
            environment_weights={"math": 2.0, "coding": 3.0},
        )
        weights = config.get_normalized_weights()
        assert abs(weights["math"] - 0.4) < 1e-6
        assert abs(weights["coding"] - 0.6) < 1e-6

    def test_from_args(self):
        """Test creating from args."""
        args = Namespace(
            unified_rlvr_environments=["math", "coding"],
            unified_rlvr_weights=[0.7, 0.3],
        )
        config = UnifiedRLVRConfig.from_args(args)
        assert config.environment_weights["math"] == 0.7
        assert config.environment_weights["coding"] == 0.3


class TestUnifiedRLVRTrainer:
    """Tests for UnifiedRLVRTrainer."""

    def test_trainer_creation(self):
        """Test creating trainer."""
        config = UnifiedRLVRConfig(
            environments=["math", "coding"],
            environment_weights={"math": 0.5, "coding": 0.5},
        )
        args = Namespace()
        trainer = UnifiedRLVRTrainer(config, args)
        
        assert "math" in trainer.env_reward_stats
        assert "coding" in trainer.env_reward_stats

    def test_sample_from_environments(self):
        """Test sampling from environments."""
        config = UnifiedRLVRConfig(
            environments=["math", "coding"],
            environment_weights={"math": 0.5, "coding": 0.5},
        )
        args = Namespace()
        trainer = UnifiedRLVRTrainer(config, args)
        
        # Create mock samples
        math_samples = [
            [Sample(prompt=f"math_{i}")] for i in range(10)
        ]
        coding_samples = [
            [Sample(prompt=f"coding_{i}")] for i in range(10)
        ]
        
        samples_by_env = {
            "math": math_samples,
            "coding": coding_samples,
        }
        
        result = trainer.sample_from_environments(samples_by_env, batch_size=10)
        assert len(result) > 0
        assert len(result) <= 10

    def test_compute_unified_rewards(self):
        """Test reward normalization."""
        config = UnifiedRLVRConfig(
            environments=["math"],
            environment_weights={"math": 1.0},
            normalize_rewards_across_envs=True,
        )
        args = Namespace()
        trainer = UnifiedRLVRTrainer(config, args)
        
        # Create samples with rewards
        samples = [
            [Sample(
                prompt="test",
                reward=1.0,
                metadata={"unified_rlvr_environment": "math"},
            )],
            [Sample(
                prompt="test2",
                reward=2.0,
                metadata={"unified_rlvr_environment": "math"},
            )],
        ]
        
        result = trainer.compute_unified_rewards(samples)
        assert len(result) == 2

    def test_create_unified_rlvr_trainer_disabled(self):
        """Test that trainer is None when disabled."""
        args = Namespace(unified_rlvr=False)
        trainer = create_unified_rlvr_trainer(args)
        assert trainer is None

    def test_create_unified_rlvr_trainer_enabled(self):
        """Test that trainer is created when enabled."""
        args = Namespace(
            unified_rlvr=True,
            unified_rlvr_environments=["math"],
            unified_rlvr_weights=[1.0],
        )
        trainer = create_unified_rlvr_trainer(args)
        assert trainer is not None
        assert isinstance(trainer, UnifiedRLVRTrainer)
