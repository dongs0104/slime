"""
Unified RLVR (Reinforcement Learning from Verifiable Rewards) Trainer.

This module provides training utilities for multi-environment RLVR,
as used in Nemotron-3-Nano and similar models.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRLVRConfig:
    """
    Configuration for Unified RLVR training.
    
    Unified RLVR trains the model simultaneously across diverse environments
    (math, coding, QA, instruction following, tool use, multi-turn conversations)
    to achieve uniform improvement across domains.
    """
    
    environments: list[str] = field(default_factory=list)
    environment_weights: dict[str, float] = field(default_factory=dict)
    normalize_rewards_across_envs: bool = True
    combine_rewards: bool = True
    reward_temperature: float = 1.0
    min_samples_per_env: int = 1

    @classmethod
    def from_args(cls, args: Namespace) -> "UnifiedRLVRConfig":
        """Create configuration from command-line arguments."""
        environments = getattr(args, "unified_rlvr_environments", None) or []
        weights = getattr(args, "unified_rlvr_weights", None) or []
        
        # Create weight mapping
        if environments and weights:
            if len(weights) != len(environments):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of environments ({len(environments)})"
                )
            environment_weights = dict(zip(environments, weights))
        else:
            # Equal weights by default
            environment_weights = {env: 1.0 for env in environments}
        
        return cls(
            environments=environments,
            environment_weights=environment_weights,
            normalize_rewards_across_envs=True,
        )

    def get_normalized_weights(self) -> dict[str, float]:
        """Get normalized environment weights that sum to 1."""
        total = sum(self.environment_weights.values())
        if total == 0:
            n = len(self.environments)
            return {env: 1.0 / n for env in self.environments}
        return {
            env: weight / total 
            for env, weight in self.environment_weights.items()
        }


class UnifiedRLVRTrainer:
    """
    Trainer for Unified RLVR across multiple environments.
    
    Key features:
    - Multi-environment sampling with configurable weights
    - Cross-environment reward normalization
    - Environment-specific reward tracking
    - Prevents overfitting to any single domain
    """
    
    def __init__(self, config: UnifiedRLVRConfig, args: Namespace):
        self.config = config
        self.args = args
        
        # Track per-environment statistics
        self.env_reward_stats: dict[str, dict[str, float]] = {}
        self.env_sample_counts: dict[str, int] = {}
        
        for env in config.environments:
            self.env_reward_stats[env] = {
                "mean": 0.0,
                "std": 1.0,
                "count": 0,
            }
            self.env_sample_counts[env] = 0

    def sample_from_environments(
        self,
        samples_by_env: dict[str, list[list[Sample]]],
        batch_size: int,
    ) -> list[list[Sample]]:
        """
        Sample from multiple environments based on configured weights.
        
        Args:
            samples_by_env: Dictionary mapping environment names to sample lists.
            batch_size: Total number of sample groups to select.
            
        Returns:
            List of sample groups balanced across environments.
        """
        weights = self.config.get_normalized_weights()
        selected = []
        
        for env, weight in weights.items():
            if env not in samples_by_env:
                continue
                
            env_samples = samples_by_env[env]
            n_select = max(
                self.config.min_samples_per_env,
                int(batch_size * weight)
            )
            n_select = min(n_select, len(env_samples))
            
            if n_select > 0:
                indices = np.random.choice(
                    len(env_samples), 
                    size=n_select, 
                    replace=False
                )
                for idx in indices:
                    group = env_samples[idx]
                    # Tag samples with environment info
                    for sample in group:
                        if sample.metadata is None:
                            sample.metadata = {}
                        sample.metadata["unified_rlvr_environment"] = env
                    selected.append(group)
                    
        return selected

    def compute_unified_rewards(
        self,
        samples: list[list[Sample]],
    ) -> list[list[Sample]]:
        """
        Normalize rewards across environments for stable training.
        
        This prevents any single environment from dominating the gradients
        and helps achieve uniform improvement across domains.
        
        Args:
            samples: List of sample groups with rewards.
            
        Returns:
            Samples with normalized rewards.
        """
        if not self.config.normalize_rewards_across_envs:
            return samples
        
        # Collect rewards by environment
        rewards_by_env: dict[str, list[float]] = {}
        
        for group in samples:
            for sample in group:
                if sample.reward is None:
                    continue
                    
                env = (sample.metadata or {}).get("unified_rlvr_environment", "default")
                if env not in rewards_by_env:
                    rewards_by_env[env] = []
                    
                reward = sample.reward
                if isinstance(reward, dict):
                    reward = reward.get("score", 0.0)
                rewards_by_env[env].append(float(reward))
        
        # Compute per-environment statistics
        env_stats: dict[str, tuple[float, float]] = {}
        for env, rewards in rewards_by_env.items():
            if rewards:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-8
                env_stats[env] = (mean, std)
                
                # Update running statistics
                old = self.env_reward_stats.get(env, {"mean": 0, "std": 1, "count": 0})
                new_count = old["count"] + len(rewards)
                alpha = len(rewards) / new_count
                self.env_reward_stats[env] = {
                    "mean": (1 - alpha) * old["mean"] + alpha * mean,
                    "std": (1 - alpha) * old["std"] + alpha * std,
                    "count": new_count,
                }
        
        # Normalize rewards
        for group in samples:
            for sample in group:
                if sample.reward is None:
                    continue
                    
                env = (sample.metadata or {}).get("unified_rlvr_environment", "default")
                if env in env_stats:
                    mean, std = env_stats[env]
                    reward = sample.reward
                    if isinstance(reward, dict):
                        reward = reward.get("score", 0.0)
                    
                    # Apply temperature scaling
                    normalized = (float(reward) - mean) / std
                    normalized *= self.config.reward_temperature
                    
                    if isinstance(sample.reward, dict):
                        sample.reward["normalized_score"] = normalized
                        sample.reward["original_score"] = sample.reward.get("score", 0.0)
                        sample.reward["score"] = normalized
                    else:
                        sample.reward = normalized
        
        return samples

    def get_environment_metrics(self) -> dict[str, Any]:
        """Get per-environment training metrics."""
        metrics = {}
        
        for env, stats in self.env_reward_stats.items():
            metrics[f"{env}/reward_mean"] = stats["mean"]
            metrics[f"{env}/reward_std"] = stats["std"]
            metrics[f"{env}/sample_count"] = stats["count"]
            
        return metrics

    def log_environment_balance(self, samples: list[list[Sample]]) -> None:
        """Log the distribution of samples across environments."""
        env_counts: dict[str, int] = {}
        
        for group in samples:
            for sample in group:
                env = (sample.metadata or {}).get("unified_rlvr_environment", "default")
                env_counts[env] = env_counts.get(env, 0) + 1
        
        total = sum(env_counts.values())
        if total > 0:
            logger.info(
                f"Unified RLVR environment distribution: "
                f"{', '.join(f'{k}: {v} ({v/total*100:.1f}%)' for k, v in env_counts.items())}"
            )


def create_unified_rlvr_trainer(args: Namespace) -> UnifiedRLVRTrainer | None:
    """
    Create a Unified RLVR trainer if enabled.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        UnifiedRLVRTrainer instance if unified_rlvr is enabled, None otherwise.
    """
    if not getattr(args, "unified_rlvr", False):
        return None
    
    config = UnifiedRLVRConfig.from_args(args)
    return UnifiedRLVRTrainer(config, args)
