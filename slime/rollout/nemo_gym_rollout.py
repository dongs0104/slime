"""
NeMo Gym based rollout generation for Slime.

This module provides rollout generation using NeMo Gym environments,
supporting Unified RLVR training with multiple environments.
"""

from __future__ import annotations

import asyncio
import logging
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
        self._initialized = True

    async def initialize(self) -> None:
        """Initialize the NeMo Gym environment."""
        if self.environment is None:
            self.environment = create_nemo_gym_environment(self.args)
            await self.environment.initialize()

    async def shutdown(self) -> None:
        """Shutdown the NeMo Gym environment."""
        if self.environment:
            await self.environment.shutdown()
            self.environment = None


async def generate_nemo_gym_sample(
    args: Namespace,
    sample: Sample,
    environment_name: str | None = None,
) -> Sample:
    """
    Generate a single sample using NeMo Gym environment.

    Args:
        args: Command-line arguments.
        sample: Input sample with prompt.
        environment_name: Optional specific environment to use.

    Returns:
        Updated sample with response and reward.
    """
    state = NemoGymRolloutState(args)
    await state.initialize()

    # Convert sample to NeMo Gym request format
    request_data = sample_to_nemo_gym_request(sample)

    # Collect rollout from NeMo Gym
    results = await state.environment.collect_rollout(
        prompts=[request_data],
        environment_name=environment_name,
    )

    if results:
        # Convert response back to sample
        sample = nemo_gym_response_to_sample(results[0], sample)
        sample.status = Sample.Status.COMPLETED

    return sample


async def generate_nemo_gym_group(
    args: Namespace,
    group: list[Sample],
    environment_name: str | None = None,
) -> list[Sample]:
    """
    Generate a group of samples using NeMo Gym environment.

    Args:
        args: Command-line arguments.
        group: List of samples to process.
        environment_name: Optional specific environment to use.

    Returns:
        List of updated samples.
    """
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

    Supports Unified RLVR with multiple environments.

    Args:
        args: Command-line arguments.
        rollout_id: Current rollout iteration ID.
        data_source: Data source providing samples.

    Returns:
        Tuple of (RolloutFnTrainOutput, aborted_samples).
    """
    state = NemoGymRolloutState(args)
    await state.initialize()

    target_batch_size = args.rollout_batch_size
    data = []

    # Get environment weights for Unified RLVR
    env_weights = state.environment.config.get_environment_weights()
    env_names = list(env_weights.keys()) if env_weights else [None]

    logger.info(f"Starting NeMo Gym rollout {rollout_id} with environments: {env_names}")

    while len(data) < target_batch_size:
        # Get samples from data source
        samples = data_source.get_samples(args.over_sampling_batch_size)

        # Distribute samples across environments (Unified RLVR)
        sample_idx = 0
        for env_name, weight in (env_weights.items() if env_weights else [(None, 1.0)]):
            env_count = max(1, int(len(samples) * weight))
            env_samples = samples[sample_idx : sample_idx + env_count]
            sample_idx += env_count

            if not env_samples:
                continue

            # Process each group
            for group in env_samples:
                result_group = await generate_nemo_gym_group(args, group, env_name)

                # Add environment info to metadata
                for sample in result_group:
                    if sample.metadata is None:
                        sample.metadata = {}
                    sample.metadata["nemo_gym_environment"] = env_name

                if len(data) < target_batch_size:
                    data.append(result_group)

    # Log first sample for debugging
    if data:
        first_sample = data[0][0] if isinstance(data[0], list) else data[0]
        logger.info(
            f"NeMo Gym rollout sample: prompt={str(first_sample.prompt)[:100]}, "
            f"reward={first_sample.reward}"
        )

    return RolloutFnTrainOutput(samples=data, metrics={}), []


async def eval_nemo_gym_rollout_async(
    args: Namespace,
    rollout_id: int,
) -> RolloutFnEvalOutput:
    """
    Evaluate using NeMo Gym environments.

    Args:
        args: Command-line arguments.
        rollout_id: Current rollout iteration ID.

    Returns:
        Evaluation results.
    """
    state = NemoGymRolloutState(args)
    await state.initialize()

    results = {}

    # Evaluate on each configured environment
    for env_config in state.environment.config.environments:
        env_name = env_config.name

        # TODO: Load evaluation dataset for this environment
        # For now, return empty results
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
    """
    Main entry point for NeMo Gym rollout generation.

    Args:
        args: Command-line arguments.
        rollout_id: Current rollout iteration ID.
        data_source: Data source providing samples.
        evaluation: Whether this is an evaluation run.

    Returns:
        Rollout output (train or eval).
    """
    if evaluation:
        output = run(eval_nemo_gym_rollout_async(args, rollout_id))
        return output

    output, aborted = run(
        generate_nemo_gym_rollout_async(args, rollout_id, data_source)
    )
    return output
