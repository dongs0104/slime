"""
NeMo Gym based rollout generation for Slime.

This module provides rollout generation using NeMo Gym environments,
supporting Unified RLVR training with multiple environments.

Supports two data modes:
1. Slime data source (default) - use Slime's HuggingFace datasets
2. NeMo Gym env data - use each environment's jsonl_fpath
"""

from __future__ import annotations

import asyncio
import logging
import random
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput
from slime.rollout.nemo_gym import (
    NemoGymConfig,
    NemoGymEnvironment,
    create_nemo_gym_environment,
    sample_to_nemo_gym_request,
    nemo_gym_response_to_sample,
)
from slime.utils.async_utils import run
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class NemoGymRolloutState:
    """State management for NeMo Gym rollout generation."""

    _instance: "NemoGymRolloutState | None" = None

    def __new__(cls, args: Namespace):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, args: Namespace):
        if self._initialized:
            return

        self.args = args
        self.environment: NemoGymEnvironment | None = None
        self.env_data: dict[str, list[dict]] = {}  # Environment-specific data
        self.env_data_indices: dict[str, int] = {}  # Current index per environment
        self._initialized = True

    async def initialize(self) -> None:
        """Initialize the NeMo Gym environment."""
        if self.environment is None:
            self.environment = create_nemo_gym_environment(self.args)
            await self.environment.initialize()
            
            # Load environment-specific data if configured
            config = self.environment.config
            if getattr(self.args, "nemo_gym_use_env_data", False) or \
               any(env.jsonl_fpath for env in config.environments):
                self.env_data = config.load_environment_data()
                self.env_data_indices = {name: 0 for name in self.env_data}
                logger.info(f"Loaded environment data for: {list(self.env_data.keys())}")

    def get_env_samples(
        self,
        env_name: str,
        count: int,
    ) -> list[dict]:
        """
        Get samples from environment-specific data.
        
        Args:
            env_name: Environment name.
            count: Number of samples to get.
            
        Returns:
            List of data rows in NeMo Gym format.
        """
        if env_name not in self.env_data:
            raise ValueError(
                f"No data loaded for environment '{env_name}'. "
                f"Available: {list(self.env_data.keys())}"
            )
        
        data = self.env_data[env_name]
        idx = self.env_data_indices[env_name]
        
        # Wrap around if needed
        if idx + count > len(data):
            # Shuffle and reset
            random.shuffle(data)
            idx = 0
        
        samples = data[idx:idx + count]
        self.env_data_indices[env_name] = idx + count
        
        return samples

    async def shutdown(self) -> None:
        """Shutdown the NeMo Gym environment."""
        if self.environment:
            await self.environment.shutdown()
            self.environment = None


def convert_nemo_gym_row_to_sample(row: dict, group_index: int = 0) -> Sample:
    """
    Convert a NeMo Gym data row to Slime Sample.
    
    Args:
        row: NeMo Gym jsonl row with responses_create_params.
        group_index: Group index for the sample.
        
    Returns:
        Slime Sample object.
    """
    # Extract prompt from responses_create_params
    responses_params = row.get("responses_create_params", {})
    messages = responses_params.get("messages", responses_params.get("input", []))
    
    if isinstance(messages, list) and messages:
        # Get the user message as prompt
        if isinstance(messages[0], dict):
            prompt = messages
        else:
            prompt = messages
    else:
        prompt = str(messages)
    
    return Sample(
        prompt=prompt,
        index=group_index,
        group_index=group_index,
        metadata=row.get("metadata", {}),
    )


async def generate_nemo_gym_sample(
    args: Namespace,
    sample: Sample,
    environment_name: str | None = None,
) -> Sample:
    """Generate a single sample using NeMo Gym environment."""
    state = NemoGymRolloutState(args)
    await state.initialize()

    request_data = sample_to_nemo_gym_request(sample)
    
    results = await state.environment.collect_rollout(
        prompts=[request_data],
        environment_name=environment_name,
    )

    if results:
        sample = nemo_gym_response_to_sample(results[0], sample)
        sample.status = Sample.Status.COMPLETED

    return sample


async def generate_nemo_gym_row(
    args: Namespace,
    row: dict,
    environment_name: str | None = None,
    group_index: int = 0,
) -> Sample:
    """
    Generate rollout from a NeMo Gym data row directly.
    
    Args:
        args: Command-line arguments.
        row: NeMo Gym jsonl row.
        environment_name: Environment to use.
        group_index: Group index for the sample.
        
    Returns:
        Completed Sample with response and reward.
    """
    state = NemoGymRolloutState(args)
    await state.initialize()
    
    # Get agent name for this environment
    env_info = state.environment.environments.get(environment_name, {})
    agent_name = env_info.get("agent_name", f"{environment_name}_simple_agent")
    
    # Call agent /run directly with the row
    result = await state.environment.run_agent(agent_name, row)
    
    # Convert result to Sample
    sample = convert_nemo_gym_row_to_sample(row, group_index)
    sample = nemo_gym_response_to_sample(result, sample)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = sample.metadata or {}
    sample.metadata["nemo_gym_environment"] = environment_name
    
    return sample


async def generate_nemo_gym_group(
    args: Namespace,
    group: list[Sample],
    environment_name: str | None = None,
) -> list[Sample]:
    """Generate a group of samples using NeMo Gym environment."""
    tasks = [
        generate_nemo_gym_sample(args, sample, environment_name) for sample in group
    ]
    return await asyncio.gather(*tasks)


async def generate_nemo_gym_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source,
) -> tuple[RolloutFnTrainOutput, list]:
    """
    Generate rollouts using NeMo Gym environments.

    Supports two modes:
    1. use_env_data=True: Use each environment's jsonl data
    2. use_env_data=False: Use Slime's data_source
    """
    state = NemoGymRolloutState(args)
    await state.initialize()

    target_batch_size = args.rollout_batch_size
    config = state.environment.config
    
    # Get environment weights for Unified RLVR
    env_weights = config.get_environment_weights()
    use_env_data = bool(state.env_data)
    
    logger.info(
        f"Starting NeMo Gym rollout {rollout_id}, "
        f"environments: {list(env_weights.keys())}, "
        f"use_env_data: {use_env_data}"
    )
    
    data = []
    group_index = 0
    
    if use_env_data:
        # Mode 1: Use environment-specific data
        while len(data) < target_batch_size:
            for env_name, weight in env_weights.items():
                if len(data) >= target_batch_size:
                    break
                
                # Calculate samples for this environment
                env_batch_size = max(1, int(args.n_samples_per_prompt * weight))
                
                # Get data rows for this environment
                try:
                    rows = state.get_env_samples(env_name, env_batch_size)
                except ValueError as e:
                    logger.warning(f"Skipping environment {env_name}: {e}")
                    continue
                
                # Generate rollouts from rows
                tasks = [
                    generate_nemo_gym_row(args, row, env_name, group_index + i)
                    for i, row in enumerate(rows)
                ]
                samples = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter successful samples
                for sample in samples:
                    if isinstance(sample, Exception):
                        logger.error(f"Rollout failed: {sample}")
                    else:
                        data.append(sample)
                        group_index += 1
    else:
        # Mode 2: Use Slime data source
        while len(data) < target_batch_size:
            samples = data_source.get_samples(args.over_sampling_batch_size)

            sample_idx = 0
            for env_name, weight in (env_weights.items() if env_weights else [(None, 1.0)]):
                env_count = max(1, int(len(samples) * weight))
                env_samples = samples[sample_idx : sample_idx + env_count]
                sample_idx += env_count

                if not env_samples:
                    continue

                for group in env_samples:
                    result_group = await generate_nemo_gym_group(args, group, env_name)

                    for sample in result_group:
                        if sample.metadata is None:
                            sample.metadata = {}
                        sample.metadata["nemo_gym_environment"] = env_name

                    if len(data) < target_batch_size:
                        data.extend(result_group)

    # Log first sample for debugging
    if data:
        first_sample = data[0] if not isinstance(data[0], list) else data[0][0]
        logger.info(
            f"NeMo Gym rollout sample: prompt={str(first_sample.prompt)[:100]}, "
            f"reward={first_sample.reward}"
        )

    return RolloutFnTrainOutput(samples=data, metrics={}), []


async def eval_nemo_gym_rollout_async(
    args: Namespace,
    rollout_id: int,
) -> RolloutFnEvalOutput:
    """Evaluate using NeMo Gym environments."""
    state = NemoGymRolloutState(args)
    await state.initialize()

    results = {}

    for env_config in state.environment.config.environments:
        env_name = env_config.name
        results[env_name] = {
            "rewards": [],
            "truncated": [],
            "samples": [],
        }

    return RolloutFnEvalOutput(data=results)


def generate_nemo_gym_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Main entry point for NeMo Gym rollout generation."""
    if evaluation:
        output = run(eval_nemo_gym_rollout_async(args, rollout_id))
        return output

    output, aborted = run(
        generate_nemo_gym_rollout_async(args, rollout_id, data_source)
    )
    return output
