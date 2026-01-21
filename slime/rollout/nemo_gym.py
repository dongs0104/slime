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
import atexit
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
    config_path: str = ""  # NeMo Gym YAML config path (for ng_run)
    jsonl_fpath: str = ""  # Data source JSONL file path (relative to nemo-gym root)
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

    # Auto-start configuration
    auto_start_servers: bool = False
    config_paths: list[str] = field(default_factory=list)

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
                    jsonl_fpath=env_data.get("jsonl_fpath", ""),
                    weight=env_data.get("weight", 1.0),
                )
            )

        # Get config_paths for auto-start
        config_paths = nemo_gym_data.get("config_paths", [])

        return cls(
            head_server_host=nemo_gym_data.get("head_server_host", "localhost"),
            head_server_port=nemo_gym_data.get("head_server_port", 11000),
            environments=environments,
            max_concurrent_rollouts=nemo_gym_data.get("max_concurrent_rollouts", 64),
            enable_on_policy_fix=nemo_gym_data.get("enable_on_policy_fix", True),
            auto_start_servers=nemo_gym_data.get("auto_start_servers", False),
            config_paths=config_paths,
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

        # Override with command-line arguments
        if hasattr(args, "nemo_gym_on_policy_fix"):
            config.enable_on_policy_fix = args.nemo_gym_on_policy_fix

        if hasattr(args, "nemo_gym_auto_start") and args.nemo_gym_auto_start:
            config.auto_start_servers = True

        return config

    def get_environment_weights(self) -> dict[str, float]:
        """Get normalized environment weights for sampling."""
        if not self.environments:
            return {}
        total = sum(env.weight for env in self.environments)
        if total == 0:
            return {env.name: 1.0 / len(self.environments) for env in self.environments}
        return {env.name: env.weight / total for env in self.environments}

    def get_config_paths(self) -> list[str]:
        """Get all config paths for server startup."""
        if self.config_paths:
            return self.config_paths
        # Collect from environments
        return [env.config_path for env in self.environments if env.config_path]

    def load_environment_data(self, nemo_gym_root: str | None = None) -> dict[str, list[dict]]:
        """
        Load data from each environment's jsonl file.
        
        Args:
            nemo_gym_root: Root directory of NeMo Gym (default: 3rdparty/nemo-gym).
        
        Returns:
            Dictionary mapping environment name to list of data rows.
        """
        import json
        from pathlib import Path
        
        if nemo_gym_root is None:
            # Default to 3rdparty/nemo-gym relative to slime root
            import os
            slime_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            nemo_gym_root = slime_root / "3rdparty" / "nemo-gym"
        else:
            nemo_gym_root = Path(nemo_gym_root)
        
        env_data = {}
        
        for env_config in self.environments:
            if not env_config.jsonl_fpath:
                logger.warning(
                    f"Environment '{env_config.name}' has no jsonl_fpath, "
                    "will use Slime data source for this environment"
                )
                continue
            
            jsonl_path = nemo_gym_root / env_config.jsonl_fpath
            
            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"Data file not found for environment '{env_config.name}': {jsonl_path}"
                )
            
            with open(jsonl_path) as f:
                rows = [json.loads(line) for line in f]
            
            logger.info(f"Loaded {len(rows)} samples for environment '{env_config.name}'")
            env_data[env_config.name] = rows
        
        return env_data


class NemoGymServerManager:
    """
    Manages NeMo Gym server lifecycle within Slime.

    Automatically starts NeMo Gym servers (head, agents, resources) when
    --use-nemo-gym with --nemo-gym-auto-start is enabled.
    """

    _instance: "NemoGymServerManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_done = False
        return cls._instance

    def __init__(self):
        if self._init_done:
            return
        self.run_helper: Any = None
        self.head_server_host: str = "localhost"
        self.head_server_port: int = 11000
        self._started = False
        self._init_done = True

    def start(self, config: NemoGymConfig) -> tuple[str, int]:
        """
        Start NeMo Gym servers from configuration.

        Args:
            config: NemoGymConfig with config_paths for environments.

        Returns:
            Tuple of (head_server_host, head_server_port).
        """
        if self._started:
            logger.info("NeMo Gym servers already started")
            return self.head_server_host, self.head_server_port

        config_paths = config.get_config_paths()

        if not config_paths:
            raise ValueError(
                "No config_paths specified for NeMo Gym auto-start. "
                "Either set config_paths or provide environments with config_path."
            )

        logger.info(f"Starting NeMo Gym servers with configs: {config_paths}")

        try:
            # Import NeMo Gym CLI components
            from nemo_gym.cli import RunHelper
            from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
            from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME
            from nemo_gym import PARENT_DIR as NEMO_GYM_ROOT
            from pathlib import Path

            # Determine env.yaml path - check Slime root first, then nemo-gym root
            import os
            slime_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            slime_env_yaml = slime_root / "env.yaml"
            nemo_gym_env_yaml = NEMO_GYM_ROOT / "env.yaml"
            
            if slime_env_yaml.exists():
                dotenv_path = slime_env_yaml
                logger.info(f"Using env.yaml from Slime root: {dotenv_path}")
            elif nemo_gym_env_yaml.exists():
                dotenv_path = nemo_gym_env_yaml
                logger.info(f"Using env.yaml from NeMo Gym: {dotenv_path}")
            else:
                dotenv_path = None
                logger.info("No env.yaml found, using config_paths only")

            # Create config parser with specified paths and dotenv
            parser_config = GlobalConfigDictParserConfig(
                config_paths=config_paths,
                dotenv_path=dotenv_path,
            )

            # Start servers using RunHelper
            self.run_helper = RunHelper()
            self.run_helper.start(parser_config)

            # Get head server info from global config
            global_config = get_global_config_dict(global_config_dict_parser_config=parser_config)

            head_config = global_config.get(HEAD_SERVER_KEY_NAME, {})
            self.head_server_host = head_config.get("host", "localhost")
            self.head_server_port = head_config.get("port", 11000)

            self._started = True
            logger.info(
                f"NeMo Gym servers started. Head server at "
                f"{self.head_server_host}:{self.head_server_port}"
            )

            # Register shutdown handler
            atexit.register(self.shutdown)

            return self.head_server_host, self.head_server_port

        except ImportError as e:
            raise ImportError(
                f"Failed to import nemo_gym. Make sure it's installed: "
                f"pip install -e 3rdparty/nemo-gym\n{e}"
            )

    def shutdown(self) -> None:
        """Shutdown all NeMo Gym servers."""
        if not self._started:
            return

        if self.run_helper:
            logger.info("Shutting down NeMo Gym servers...")
            try:
                self.run_helper.shutdown()
            except Exception as e:
                logger.warning(f"Error during NeMo Gym shutdown: {e}")
            self.run_helper = None

        self._started = False
        logger.info("NeMo Gym servers shut down")

    def is_running(self) -> bool:
        """Check if servers are running."""
        return self._started


# Global server manager
_server_manager: NemoGymServerManager | None = None


def get_server_manager() -> NemoGymServerManager:
    """Get the global NeMo Gym server manager."""
    global _server_manager
    if _server_manager is None:
        _server_manager = NemoGymServerManager()
    return _server_manager


def start_nemo_gym_servers(args: Namespace) -> tuple[str, int]:
    """
    Start NeMo Gym servers from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (head_server_host, head_server_port).
    """
    config = NemoGymConfig.from_args(args)

    if not config.auto_start_servers:
        # Servers not auto-started, just return configured values
        return config.head_server_host, config.head_server_port

    manager = get_server_manager()
    return manager.start(config)


def shutdown_nemo_gym_servers() -> None:
    """Shutdown NeMo Gym servers if running."""
    manager = get_server_manager()
    manager.shutdown()


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
        self.server_client: Any = None
        self.environments: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._semaphore: asyncio.Semaphore | None = None

    async def initialize(self) -> None:
        """Initialize the environment and connect to NeMo Gym head server."""
        if self._initialized:
            return

        from nemo_gym.server_utils import ServerClient
        from nemo_gym.config_types import BaseServerConfig

        head_server_config = BaseServerConfig(
            host=self.config.head_server_host,
            port=self.config.head_server_port,
        )

        self.server_client = ServerClient.load_from_global_config(head_server_config)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_rollouts)

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
        """Call an agent's /run endpoint for rollout collection."""
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
        """Call a resource server's /verify endpoint for reward calculation."""
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
        """Collect rollouts from NeMo Gym environment."""
        if not self._initialized:
            await self.initialize()

        if environment_name is None and self.environments:
            environment_name = list(self.environments.keys())[0]

        env_info = self.environments.get(environment_name, {})
        agent_name = env_info.get("agent_name", f"{environment_name}_simple_agent")

        tasks = []
        for prompt in prompts:
            request_data = self._format_agent_request(prompt, environment_name)
            tasks.append(self.run_agent(agent_name, request_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Rollout {i} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    def _format_agent_request(
        self,
        prompt: dict[str, Any],
        environment_name: str | None = None,
    ) -> dict[str, Any]:
        """Format a prompt for NeMo Gym agent request."""
        if "responses_create_params" not in prompt:
            prompt = {
                "responses_create_params": {
                    "messages": prompt.get("messages", []),
                    **{k: v for k, v in prompt.items() if k != "messages"},
                }
            }

        if environment_name:
            prompt["resource_ref"] = {"name": environment_name}

        return prompt

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self.server_client = None
        logger.info("NeMo Gym environment shut down")


def create_nemo_gym_environment(args: Namespace) -> NemoGymEnvironment:
    """Create a NeMo Gym environment from command-line arguments."""
    config = NemoGymConfig.from_args(args)
    return NemoGymEnvironment(config)


def sample_to_nemo_gym_request(sample: Sample) -> dict[str, Any]:
    """Convert a Slime Sample to NeMo Gym request format."""
    messages = []

    if isinstance(sample.prompt, str):
        messages.append({"role": "user", "content": sample.prompt})
    elif isinstance(sample.prompt, list):
        messages = list(sample.prompt)

    if sample.response:
        messages.append({"role": "assistant", "content": sample.response})

    return {
        "responses_create_params": {"messages": messages},
        "metadata": sample.metadata or {},
    }


def nemo_gym_response_to_sample(
    response: dict[str, Any],
    original_sample: Sample,
) -> Sample:
    """Convert a NeMo Gym response back to a Slime Sample."""
    sample = original_sample

    if "response" in response:
        nemo_response = response["response"]
        if "output" in nemo_response:
            output = nemo_response["output"]
            if isinstance(output, list) and output:
                last_msg = output[-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    sample.response = last_msg["content"]
            elif isinstance(output, str):
                sample.response = output

    if "reward" in response:
        sample.reward = response["reward"]

    if "metadata" in response:
        sample.metadata = sample.metadata or {}
        sample.metadata.update(response["metadata"])

    return sample
