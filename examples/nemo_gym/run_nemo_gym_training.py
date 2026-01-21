#!/usr/bin/env python3
"""
Example script for running Slime training with NeMo Gym environments.

This script demonstrates:
1. Auto-starting NeMo Gym servers
2. Configuring environments via YAML
3. Running Unified RLVR training
"""

import argparse
import logging
import sys
from pathlib import Path

# Add slime to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NeMo Gym + Slime Training Example")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nemo_gym_unified.yaml",
        help="Path to NeMo Gym configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rollout batch size",
    )
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=100,
        help="Number of rollouts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test NeMo Gym connection without training",
    )
    args = parser.parse_args()

    if args.dry_run:
        test_nemo_gym_connection(args.config)
    else:
        run_training(args)


def test_nemo_gym_connection(config_path: str):
    """Test NeMo Gym server connection without running full training."""
    from slime.rollout.nemo_gym import (
        NemoGymConfig,
        NemoGymServerManager,
        get_server_manager,
    )
    
    logger.info(f"Loading config from {config_path}")
    config = NemoGymConfig.from_yaml(config_path)
    
    logger.info(f"Config loaded: {config}")
    logger.info(f"Environments: {[e.name for e in config.environments]}")
    logger.info(f"Config paths: {config.get_config_paths()}")
    
    if config.auto_start_servers:
        logger.info("Testing auto-start servers...")
        manager = get_server_manager()
        try:
            host, port = manager.start(config)
            logger.info(f"✅ NeMo Gym servers started at {host}:{port}")
            
            # Test connection
            import requests
            response = requests.get(f"http://{host}:{port}/server_instances", timeout=5)
            if response.ok:
                instances = response.json()
                logger.info(f"✅ Connected! Found {len(instances)} server instances:")
                for inst in instances:
                    logger.info(f"   - {inst.get('name', 'unknown')}: {inst.get('url', 'no url')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start servers: {e}")
        finally:
            manager.shutdown()
            logger.info("Servers shut down")
    else:
        logger.info("auto_start_servers is False, skipping server startup test")


def run_training(args):
    """Run full training with NeMo Gym."""
    import subprocess
    
    cmd = [
        "python", "train.py",
        "--use-nemo-gym",
        "--nemo-gym-config", args.config,
        "--hf-checkpoint", args.model,
        "--rollout-batch-size", str(args.batch_size),
        "--num-rollout", str(args.num_rollout),
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)


if __name__ == "__main__":
    main()
