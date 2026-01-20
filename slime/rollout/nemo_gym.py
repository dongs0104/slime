"""
NeMo Gym integration module for Slime.

This module provides the integration layer between Slime and NeMo Gym,
enabling Unified RLVR training with multiple environments.

The integration follows NeMo Gym's API structure:
- ServerClient: Connects to head server and manages server discovery
- Agent servers: Handle rollout collection via /run endpoint
- Resources servers: Handle verification via /verify endpoint
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
    from nemo_gym.server_utils import ServerClient

logger = logging.getLogger(__name__)


@dataclass
class NemoGymEnvironmentConfig:
    """Configuration for a single NeMo Gym environment."""

    name: str  # Resource server name
    agent_name: str = ""  # Agent server name (defaults to {name}_simple_agent)
    config_path: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if not self.agent_name:
            self.agent_name = f"{self.name}_simple_agent"


@dataclass
class NemoGymConfig:
    """
    NeMo Gym integration configuration.

    Supports multiple environments for Unified RLVR training.
    """

    head_server_host: str = "localhost"
    head_server_port: int = 11000
    environments: list[NemoGymEnvironmentConfig] = field(default_factory=list)
    max_concurrent_rollouts: int = 64
    enable_on_policy_fix: bool = True

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
                    agent_name=env_data.get("agent_name", ""),
                    config_path=env_data.get("config", ""),
                    weight=env_data.get("weight", 1.0),
                )
            )

        return cls(
            head_server_host=nemo_gym_data.get("head_server_host", "localhost"),
            head_server_port=nemo_gym_data.get("head_server_port", 11000),
            environments=environments,
            max_concurrent_rollouts=nemo_gym_data.get("max_concurrent_rollouts", 64),
            enable_on_policy_fix=nemo_gym_data.get("enable_on_policy_fix", True),
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "NemoGymConfig":
        """Create configuration from command-line arguments."""
        if getattr(args, "nemo_gym_config", None):
            config = cls.from_yaml(args.nemo_gym_config)
        else:
            config = cls(
                head_server_host=getattr(args, "nemo_gym_head_server_host", "localhost"),
                head_server_port=getattr(args, "nemo_gym_head_server_port", 11000),
            )

        if hasattr(args, "nemo_gym_on_policy_fix"):
            config.enable_on_policy_fix = args.nemo_gym_on_policy_fix

        return config

    def get_environment_weights(self) -> dict[str, float]:
        """Get normalized environment weights for sampling."""
        if not self.environments:
            return {}
        total = sum(env.weight for env in self.environments)
        if total == 0:
            return {env.name: 1.0 / len(self.environments) for env in self.environments}
        return {env.name: env.weight / total for env in self.environments}


class NemoGymEnvironment:
    """
    NeMo Gym environment wrapper.

    Uses NeMo Gym's ServerClient API to:
    - Connect to head server for service discovery
    - Call agent servers via /run for rollout collection
    - Call resources servers via /verify for reward calculation
    """

    def __init__(self, config: NemoGymConfig):
        self.config = config
        self.server_client: Any = None  # ServerClient from nemo_gym
        self.environments: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._semaphore: asyncio.Semaphore | None = None

    async def initialize(self) -> None:
        """Initialize the environment and connect to NeMo Gym head server."""
        if self._initialized:
            return

        # Import nemo_gym components
        from nemo_gym.server_utils import ServerClient
        from nemo_gym.config_types import BaseServerConfig

        # Create head server config
        head_server_config = BaseServerConfig(
            host=self.config.head_server_host,
            port=self.config.head_server_port,
        )

        # Load server client from head server
        self.server_client = ServerClient.load_from_global_config(head_server_config)

        # Create semaphore for concurrent request control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_rollouts)

        # Register all configured environments
        for env_config in self.config.environments:
            self.environments[env_config.name] = {
                "config": env_config,
                "agent_name": env_config.agent_name,
            }

        self._initialized = True
        logger.info(
            f"NeMo Gym initialized with {len(self.environments)} environments: "
            f"{list(self.environments.keys())}"
        )

    async def run_agent(
        self,
        agent_name: str,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call an agent's /run endpoint for rollout collection.

        This is the primary method for generating rollouts through NeMo Gym.

        Args:
            agent_name: Name of the agent server to call.
            request_data: Request payload containing prompt and parameters.

        Returns:
            Agent response with generation and optional reward.
        """
        if not self._initialized:
            await self.initialize()

        from nemo_gym.server_utils import raise_for_status, get_response_json

        async with self._semaphore:
            response = await self.server_client.post(
                server_name=agent_name,
                url_path="/run",
                json=request_data,
            )
            await raise_for_status(response)
            return await get_response_json(response)

    async def verify(
        self,
        resource_server_name: str,
        response_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call a resource server's /verify endpoint for reward calculation.

        Args:
            resource_server_name: Name of the resource server.
            response_data: The response to verify.

        Returns:
            Verification result including reward.
        """
        if not self._initialized:
            await self.initialize()

        from nemo_gym.server_utils import raise_for_status, get_response_json

        async with self._semaphore:
            response = await self.server_client.post(
                server_name=resource_server_name,
                url_path="/verify",
                json=response_data,
            )
            await raise_for_status(response)
            return await get_response_json(response)

    async def collect_rollout(
        self,
        prompts: list[dict[str, Any]],
        environment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect rollouts from NeMo Gym environment.

        Args:
            prompts: List of prompt dictionaries with responses_create_params.
            environment_name: Specific environment to use.

        Returns:
            List of rollout results with rewards.
        """
        if not self._initialized:
            await self.initialize()

        # Use default environment if not specified
        if environment_name is None and self.environments:
            environment_name = list(self.environments.keys())[0]

        env_info = self.environments.get(environment_name, {})
        agent_name = env_info.get("agent_name", f"{environment_name}_simple_agent")

        # Collect rollouts in parallel
        tasks = []
        for prompt in prompts:
            # Format request for NeMo Gym agent
            request_data = self._format_agent_request(prompt, environment_name)
            tasks.append(self.run_agent(agent_name, request_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Rollout {i} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

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

    def _format_agent_request(
        self,
        prompt: dict[str, Any],
        environment_name: str | None = None,
    ) -> dict[str, Any]:
        """Format a prompt for NeMo Gym agent request."""
        # NeMo Gym expects responses_create_params with messages
        if "responses_create_params" not in prompt:
            prompt = {
                "responses_create_params": {
                    "messages": prompt.get("messages", []),
                    **{k: v for k, v in prompt.items() if k != "messages"},
                }
            }

        # Add environment reference if specified
        if environment_name:
            prompt["resource_ref"] = {"name": environment_name}

        return prompt

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self.server_client = None
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


def sample_to_nemo_gym_request(sample: Sample) -> dict[str, Any]:
    """
    Convert a Slime Sample to NeMo Gym request format.

    Args:
        sample: Slime Sample object.

    Returns:
        Dictionary in NeMo Gym agent request format.
    """
    messages = []

    # Parse prompt
    if isinstance(sample.prompt, str):
        messages.append({"role": "user", "content": sample.prompt})
    elif isinstance(sample.prompt, list):
        messages = list(sample.prompt)

    # Add response if available (for verification)
    if sample.response:
        messages.append({"role": "assistant", "content": sample.response})

    return {
        "responses_create_params": {
            "messages": messages,
        },
        "metadata": sample.metadata or {},
    }


def nemo_gym_response_to_sample(
    response: dict[str, Any],
    original_sample: Sample,
) -> Sample:
    """
    Convert a NeMo Gym response back to a Slime Sample.

    Args:
        response: NeMo Gym agent response dictionary.
        original_sample: Original sample to update.

    Returns:
        Updated Sample object.
    """
    sample = original_sample

    # Extract response from NeMo Gym format
    if "response" in response:
        nemo_response = response["response"]
        if "output" in nemo_response:
            # Extract last assistant message
            output = nemo_response["output"]
            if isinstance(output, list) and output:
                last_msg = output[-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    sample.response = last_msg["content"]
            elif isinstance(output, str):
                sample.response = output

    # Extract reward
    if "reward" in response:
        sample.reward = response["reward"]

    # Update metadata
    if "metadata" in response:
        sample.metadata = sample.metadata or {}
        sample.metadata.update(response["metadata"])

    return sample
