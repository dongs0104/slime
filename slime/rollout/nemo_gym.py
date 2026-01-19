"""
NeMo Gym integration module for Slime.

This module provides the integration layer between Slime and NeMo Gym,
enabling Unified RLVR training with multiple environments.
"""

from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from slime.utils.types import Sample

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


@dataclass
class NemoGymEnvironmentConfig:
    """Configuration for a single NeMo Gym environment."""

    name: str
    config_path: str
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


@dataclass
class NemoGymConfig:
    """
    NeMo Gym integration configuration.

    Supports multiple environments for Unified RLVR training.
    """

    resource_server_url: str
    environments: list[NemoGymEnvironmentConfig] = field(default_factory=list)
    policy_model_url: str = ""
    policy_model_name: str = "slime"
    max_concurrent_rollouts: int = 64
    timeout_seconds: float = 300.0
    enable_on_policy_fix: bool = True
    health_check_interval: float = 30.0

    @classmethod
    def from_yaml(cls, path: str) -> "NemoGymConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        nemo_gym_data = data.get("nemo_gym", data)

        # Parse environments
        environments = []
        for env_data in nemo_gym_data.get("environments", []):
            environments.append(
                NemoGymEnvironmentConfig(
                    name=env_data["name"],
                    config_path=env_data["config"],
                    weight=env_data.get("weight", 1.0),
                )
            )

        return cls(
            resource_server_url=nemo_gym_data.get("resource_server_url", ""),
            environments=environments,
            policy_model_url=nemo_gym_data.get("policy_model_url", ""),
            policy_model_name=nemo_gym_data.get("policy_model_name", "slime"),
            max_concurrent_rollouts=nemo_gym_data.get("max_concurrent_rollouts", 64),
            timeout_seconds=nemo_gym_data.get("timeout_seconds", 300.0),
            enable_on_policy_fix=nemo_gym_data.get("enable_on_policy_fix", True),
            health_check_interval=nemo_gym_data.get("health_check_interval", 30.0),
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "NemoGymConfig":
        """Create configuration from command-line arguments."""
        if args.nemo_gym_config:
            config = cls.from_yaml(args.nemo_gym_config)
        else:
            config = cls(
                resource_server_url=getattr(args, "nemo_gym_resource_server_url", ""),
            )

        # Override with command-line arguments
        if hasattr(args, "nemo_gym_policy_model_url") and args.nemo_gym_policy_model_url:
            config.policy_model_url = args.nemo_gym_policy_model_url

        if hasattr(args, "nemo_gym_on_policy_fix"):
            config.enable_on_policy_fix = args.nemo_gym_on_policy_fix

        return config

    def get_environment_weights(self) -> dict[str, float]:
        """Get normalized environment weights for sampling."""
        total = sum(env.weight for env in self.environments)
        if total == 0:
            return {env.name: 1.0 / len(self.environments) for env in self.environments}
        return {env.name: env.weight / total for env in self.environments}


class NemoGymEnvironment:
    """
    NeMo Gym environment wrapper.

    Provides connection to NeMo Gym Resource Server and supports:
    - Multi-environment rollout collection
    - Unified RLVR training
    - On-policy token ID correction
    """

    def __init__(self, config: NemoGymConfig):
        self.config = config
        self.client: Any = None  # httpx.AsyncClient, lazy loaded
        self.environments: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._health_check_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the environment and connect to Resource Server."""
        if self._initialized:
            return

        # Lazy import httpx
        import httpx

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            limits=httpx.Limits(max_connections=self.config.max_concurrent_rollouts),
        )

        # Verify Resource Server connection
        await self._check_health()

        # Load all configured environments
        for env_config in self.config.environments:
            await self._load_environment(env_config)

        self._initialized = True
        logger.info(
            f"NeMo Gym initialized with {len(self.environments)} environments: "
            f"{list(self.environments.keys())}"
        )

    async def _check_health(self) -> bool:
        """Check Resource Server health."""
        if not self.client:
            return False

        try:
            response = await self.client.get(f"{self.config.resource_server_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def _load_environment(self, env_config: NemoGymEnvironmentConfig) -> None:
        """Load a single environment configuration."""
        self.environments[env_config.name] = {
            "config": env_config,
            "loaded": True,
        }
        logger.info(f"Loaded environment: {env_config.name}")

    async def collect_rollout(
        self,
        prompts: list[dict[str, Any]],
        environment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect rollouts from NeMo Gym environment.

        Args:
            prompts: List of prompt dictionaries in OpenAI format.
            environment_name: Optional specific environment to use.

        Returns:
            List of rollout results with verification scores.
        """
        if not self._initialized:
            await self.initialize()

        # Use default environment if not specified
        if environment_name is None and self.environments:
            environment_name = list(self.environments.keys())[0]

        # Prepare request payload
        payload = {
            "prompts": prompts,
            "environment": environment_name,
            "policy_model_url": self.config.policy_model_url,
            "policy_model_name": self.config.policy_model_name,
        }

        try:
            response = await self.client.post(
                f"{self.config.resource_server_url}/collect_rollouts",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Rollout collection failed: {e}")
            raise

    async def collect_unified_rollouts(
        self,
        prompts: list[dict[str, Any]],
        environment_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect rollouts from multiple environments (Unified RLVR).

        Args:
            prompts: List of prompt dictionaries.
            environment_weights: Optional custom weights for environment sampling.

        Returns:
            List of rollout results from multiple environments.
        """
        if not self._initialized:
            await self.initialize()

        weights = environment_weights or self.config.get_environment_weights()

        # Distribute prompts across environments based on weights
        results = []
        prompt_idx = 0

        for env_name, weight in weights.items():
            env_prompts_count = max(1, int(len(prompts) * weight))
            env_prompts = prompts[prompt_idx : prompt_idx + env_prompts_count]
            prompt_idx += env_prompts_count

            if env_prompts:
                env_results = await self.collect_rollout(env_prompts, env_name)
                for result in env_results:
                    result["environment"] = env_name
                results.extend(env_results)

        return results

    async def get_verification_result(
        self,
        rollout: dict[str, Any],
        environment_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get verification result and reward for a rollout.

        Args:
            rollout: The rollout data to verify.
            environment_name: Optional specific environment for verification.

        Returns:
            Dictionary with verification result and reward.
        """
        if not self._initialized:
            await self.initialize()

        payload = {
            "rollout": rollout,
            "environment": environment_name,
        }

        try:
            response = await self.client.post(
                f"{self.config.resource_server_url}/verify",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self.client:
            await self.client.aclose()

        self._initialized = False
        logger.info("NeMo Gym environment shut down")


def create_nemo_gym_environment(args: Namespace) -> NemoGymEnvironment:
    """
    Create a NeMo Gym environment from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured NemoGymEnvironment instance.
    """
    config = NemoGymConfig.from_args(args)
    return NemoGymEnvironment(config)


def sample_to_openai_format(sample: Sample) -> dict[str, Any]:
    """
    Convert a Slime Sample to OpenAI API format.

    Args:
        sample: Slime Sample object.

    Returns:
        Dictionary in OpenAI chat completion format.
    """
    messages = []

    # Parse prompt if it's a string (may contain chat history)
    if isinstance(sample.prompt, str):
        messages.append({"role": "user", "content": sample.prompt})
    elif isinstance(sample.prompt, list):
        messages = sample.prompt

    # Add response if available
    if sample.response:
        messages.append({"role": "assistant", "content": sample.response})

    return {
        "messages": messages,
        "metadata": sample.metadata or {},
    }


def openai_response_to_sample(
    response: dict[str, Any],
    original_sample: Sample,
) -> Sample:
    """
    Convert an OpenAI API response back to a Slime Sample.

    Args:
        response: OpenAI API response dictionary.
        original_sample: Original sample to update.

    Returns:
        Updated Sample object.
    """
    sample = original_sample

    # Extract response text
    if "choices" in response:
        choice = response["choices"][0]
        if "message" in choice:
            sample.response = choice["message"].get("content", "")
        elif "text" in choice:
            sample.response = choice["text"]

    # Extract reward/verification if available
    if "reward" in response:
        sample.reward = response["reward"]
    elif "verification" in response:
        sample.reward = response["verification"].get("score", 0.0)

    # Update metadata
    if "metadata" in response:
        sample.metadata = sample.metadata or {}
        sample.metadata.update(response["metadata"])

    return sample
