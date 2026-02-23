# CLAUDE.md

This file provides guidance for AI assistants working in the slime codebase.

## Project Overview

**slime** (v0.2.2) is an LLM post-training framework for Reinforcement Learning (RL) scaling. It powers production models including GLM-4.7, GLM-4.6, and GLM-4.5. The framework connects **Megatron-LM** (training) with **SGLang** (inference/rollout) via **Ray** (distributed orchestration).

**Key capabilities:**
- High-performance RL training (GRPO, PPO, REINFORCE++, SFT, and custom algorithms)
- Flexible data generation via custom rollout/reward interfaces
- Support for Qwen3, DeepSeek V3, Llama 3, GLM-4 model families
- Synchronous (`train.py`) and asynchronous (`train_async.py`) training loops
- Colocated and separated training/inference GPU topologies

## Repository Structure

```
slime/
├── train.py                  # Synchronous training entry point
├── train_async.py            # Asynchronous training entry point (overlaps rollout + train)
├── slime/                    # Main Python package
│   ├── ray/                  # Ray-based distributed orchestration
│   │   ├── placement_group.py  # GPU placement group allocation, factory functions
│   │   ├── actor_group.py      # RayTrainGroup - manages Megatron training actors
│   │   ├── rollout.py          # RolloutManager - manages SGLang inference engines
│   │   ├── train_actor.py      # Individual Ray training actor
│   │   └── ray_actor.py        # Base Ray actor utilities
│   ├── rollout/              # Data generation components
│   │   ├── sglang_rollout.py   # Default rollout function (generate_rollout)
│   │   ├── sft_rollout.py      # SFT rollout (no reward computation)
│   │   ├── sleep_rollout.py    # Debug rollout (sleeps instead of generating)
│   │   ├── base_types.py       # RolloutFnTrainOutput, RolloutFnEvalOutput
│   │   ├── data_source.py      # RolloutDataSourceWithBuffer
│   │   ├── rm_hub/             # Built-in reward models (math, GPQA, F1, etc.)
│   │   ├── filter_hub/         # Dynamic sampling filters
│   │   └── generate_hub/       # Benchmark generation utilities
│   ├── backends/             # Training backends
│   │   ├── megatron_utils/     # Megatron-LM backend (recommended)
│   │   │   ├── actor.py        # MegatronActor - main training logic
│   │   │   ├── model.py        # Megatron model definitions
│   │   │   ├── model_provider.py
│   │   │   ├── loss.py         # Policy/SFT/custom loss functions
│   │   │   ├── arguments.py    # Megatron-specific argument parsing
│   │   │   ├── initialize.py   # Megatron initialization
│   │   │   ├── checkpoint.py   # Checkpoint save/load
│   │   │   ├── sglang.py       # Weight sync from Megatron to SGLang
│   │   │   ├── update_weight/  # Weight update utilities
│   │   │   └── megatron_to_hf/ # Megatron→HuggingFace weight converters
│   │   ├── fsdp_utils/         # FSDP backend (being rewritten, use Megatron)
│   │   └── sglang_utils/       # SGLang engine management
│   ├── router/               # Custom HTTP request router
│   │   └── middleware_hub/     # Radix-tree prefix caching middleware
│   └── utils/                # Utilities
│       ├── arguments.py        # All slime-specific argument definitions
│       ├── types.py            # Core data types (Sample, RolloutBatch, etc.)
│       ├── logging_utils.py    # Logger configuration, W&B/TensorBoard init
│       ├── data.py             # Dataset loading
│       ├── mask_utils.py       # Loss mask generation (MultiTurnLossMaskGenerator)
│       ├── misc.py             # load_function, should_run_periodic_action, etc.
│       ├── metric_utils.py     # Metric computation
│       ├── processing_utils.py # Tokenizer/processor loading
│       └── ...
├── slime_plugins/            # Extension plugin package
│   ├── rollout_buffer/       # External rollout buffer (for async agentic training)
│   │   ├── buffer.py           # RolloutBuffer server
│   │   ├── generator/          # BaseGenerator for external data generation
│   │   └── rollout_buffer_example.py
│   ├── mbridge/              # Model bridge (custom model architectures)
│   │   ├── glm4.py, glm4moe.py, qwen3_next.py, mimo.py
│   ├── models/               # Custom HuggingFace model definitions
│   └── megatron_bridge/      # Megatron bridge utilities
├── scripts/                  # Training launch scripts
│   ├── models/               # Model-specific argument configs (source these)
│   ├── run-qwen3-4B.sh       # Example: Qwen3-4B training
│   ├── run-deepseek-r1.sh    # Example: DeepSeek R1 training
│   └── ...
├── tools/                    # Utility scripts
│   ├── convert_hf_to_torch_dist.py   # HF → Megatron format
│   └── convert_torch_dist_to_hf.py   # Megatron → HF format
├── tests/                    # Test suite
│   ├── ci/                   # CI-specific test helpers
│   └── test_*.py / test_*.sh # Integration and system tests
├── examples/                 # Use-case examples
│   ├── fully_async/          # Fully asynchronous training
│   ├── multi_agent/          # Multi-agent RL
│   ├── true_on_policy/       # True on-policy training
│   ├── search-r1/            # Search-augmented RL
│   └── ...
├── docs/                     # Sphinx documentation (English + Chinese)
├── docker/                   # Docker build configs
├── pyproject.toml            # Build config, Black/isort/ruff/pytest settings
├── setup.py                  # Package setup (version 0.2.2)
└── requirements.txt          # Runtime dependencies
```

## Architecture

The training loop follows a three-module design:

```
┌──────────────┐     rollout data      ┌──────────────┐
│   Training   │ ←──────────────────── │    Rollout   │
│  (Megatron)  │                       │   (SGLang)   │
│              │ ──── weight sync ───→ │              │
└──────────────┘                       └──────────────┘
       ↑↓ data
┌──────────────┐
│ Data Buffer  │
│  (RolloutMgr)│
└──────────────┘
```

- **Training module**: Megatron-based distributed training (actor/critic)
- **Rollout module**: SGLang inference servers managed by `RolloutManager`, produces samples with rewards
- **Data buffer**: `RolloutDataSourceWithBuffer` manages prompt pool, sampling, filtering

**Ray actors involved:**
- `RayTrainGroup` → N×`RayTrainActor` (one per GPU) → Megatron process groups
- `RolloutManager` → M×SGLang engine instances

## Core Data Types

All located in `slime/utils/types.py`:

### `Sample` (dataclass)
The fundamental unit passing through rollout → reward → training:
```python
@dataclass
class Sample:
    group_index: int | None       # group of samples from same prompt
    index: int | None             # position within group
    prompt: str | list[dict]      # raw prompt text or chat messages
    tokens: list[int]             # full sequence tokens (prompt + response)
    response: str                 # generated response text
    response_length: int          # number of response tokens
    label: str | None             # ground truth label
    reward: float | dict | None   # reward signal (scalar or dict)
    loss_mask: list[int] | None   # per-token loss mask
    status: Sample.Status         # PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED
    remove_sample: bool           # exclude from loss calculation
    metadata: dict                # arbitrary per-sample metadata
    train_metadata: dict | None   # training-specific metadata
    rollout_log_probs: list[float] | None  # log probs from rollout engine
```

### `RolloutBatch`
A dict of tensor lists produced from a group of `Sample`s, consumed by Megatron data iterators.

### `RolloutFnTrainOutput` / `RolloutFnEvalOutput`
Return types for rollout functions (in `slime/rollout/base_types.py`).

## Argument System

Arguments are divided into three namespaces (all parsed into one flat `args` object):

1. **Megatron arguments** — standard Megatron-LM flags (e.g., `--tensor-model-parallel-size`)
2. **SGLang arguments** — prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
3. **slime arguments** — defined in `slime/utils/arguments.py`

### Critical slime Arguments

| Argument | Required | Description |
|---|---|---|
| `--train-backend` | No (default: `megatron`) | `megatron` or `fsdp` |
| `--hf-checkpoint` | Yes | Path to HuggingFace model checkpoint |
| `--prompt-data` | Yes* | Path to JSONL training data |
| `--rollout-batch-size` | **Required** | Prompts per rollout step |
| `--n-samples-per-prompt` | No (default: 1) | Responses per prompt |
| `--num-rollout` | Yes* | Total rollout steps (`--num-epoch` alternative) |
| `--actor-num-nodes` | No (default: 1) | Training nodes |
| `--actor-num-gpus-per-node` | No (default: 8) | GPUs per training node |
| `--rollout-num-gpus` | No | Total inference GPUs |
| `--rollout-num-gpus-per-engine` | No (default: 1) | TP size for each SGLang engine |
| `--colocate` | No | Share GPUs between training and inference |
| `--advantage-estimator` | No (default: `grpo`) | `grpo`, `gspo`, `reinforce_plus_plus`, `ppo`, `on_policy_distillation` |
| `--loss-type` | No (default: `policy_loss`) | `policy_loss`, `sft_loss`, `custom_loss` |
| `--kl-coef` | No (default: 0.0) | KL penalty coefficient (reward shaping) |
| `--save` | No | Checkpoint save directory |
| `--load` | No | Checkpoint to load from |

### Customization Function Paths
slime uses a plugin pattern where any function can be replaced by path string:

| Argument | Default | Signature |
|---|---|---|
| `--rollout-function-path` | `slime.rollout.sglang_rollout.generate_rollout` | `fn(args, rollout_id, *, evaluation=False) -> RolloutFnTrainOutput\|RolloutFnEvalOutput` |
| `--custom-generate-function-path` | None | `fn(args, sample, sampling_params)` |
| `--custom-rm-path` | None | `fn(args, sample) -> float` |
| `--custom-loss-function-path` | None | Custom loss (when `--loss-type custom_loss`) |
| `--dynamic-sampling-filter-path` | None | `fn(args, samples) -> bool` |
| `--buffer-filter-path` | None | `fn(list[list[Sample]]) -> list[list[Sample]]` |
| `--rollout-data-postprocess-path` | None | Post-process rollout data |
| `--data-source-path` | `slime.rollout.data_source.RolloutDataSourceWithBuffer` | Custom data source class |

Use `slime.utils.misc.load_function` to dynamically load these functions.

## Development Workflow

### Environment Setup

**Recommended (Docker):**
```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash

# Update slime in container
cd /root/slime && git pull && pip install -e . --no-deps
```

**Conda alternative:**
```bash
bash build_conda.sh
```

### Code Style

Install and use pre-commit hooks:
```bash
apt install pre-commit -y
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

The hooks enforce (in order):
1. **ruff** — linting with auto-fix (`E`, `F`, `B`, `UP` rules; `E402`, `E501` ignored)
2. **autoflake** — removes unused imports
3. **isort** — import sorting (black-compatible profile, line_length=119)
4. **black** — code formatting (line_length=119)

Config is in `pyproject.toml`. Do not add `# noqa` comments unless absolutely necessary — prefer fixing the issue.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test markers
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m "not skipduringci"

# Run a specific test file
pytest tests/test_qwen2.5_0.5B_gsm8k.py
```

Test markers defined in `pyproject.toml`:
- `unit` — isolated unit tests
- `integration` — subsystem integration tests
- `system` — full system tests
- `acceptance` — user acceptance criteria
- `skipduringci` — skipped in CI (requires special hardware/setup)
- `pleasefixme` — broken tests needing attention

Many tests require actual GPU hardware and model weights — they are integration/system tests that spin up Ray clusters and SGLang servers.

### Model Weight Conversion

**HuggingFace → Megatron (required before Megatron backend training):**
```bash
source scripts/models/<model>.sh   # loads MODEL_ARGS
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/hf_model \
    --save /path/to/megatron_ckpt
```

**Megatron → HuggingFace:**
```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    ${MODEL_ARGS[@]} \
    --load /path/to/megatron_ckpt \
    --save /path/to/hf_output
```

### Launching Training

```bash
# Using a script from scripts/
bash scripts/run-qwen3-4B.sh

# Or directly
python train.py \
    --train-backend megatron \
    --hf-checkpoint /path/to/model \
    --prompt-data /path/to/data.jsonl \
    --rollout-batch-size 64 \
    --n-samples-per-prompt 8 \
    --num-rollout 1000 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    --advantage-estimator grpo \
    --tensor-model-parallel-size 4 \
    # ... more Megatron args
```

**Asynchronous training** (overlaps rollout and training for better throughput):
```bash
python train_async.py [same args]  # note: --colocate not supported
```

## Key Conventions

### Adding a Custom Rollout Function

Create a Python module with the function and pass its dotted path:
```python
# my_project/my_rollout.py
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample

def generate_rollout(args, rollout_id, *, evaluation=False):
    samples: list[list[Sample]] = []
    # ... generate and score samples
    return RolloutFnTrainOutput(samples=samples)
```
```bash
python train.py --rollout-function-path my_project.my_rollout.generate_rollout ...
```

### Adding a Custom Reward Model

```python
# my_project/my_rm.py
def custom_rm(args, sample):
    # sample.response contains the generated text
    # sample.label contains ground truth
    return 1.0 if correct(sample.response, sample.label) else 0.0
```
```bash
python train.py --custom-rm-path my_project.my_rm.custom_rm --rm-type custom ...
```

### Adding a Custom Loss Function

```python
# my_project/my_loss.py
def custom_loss(args, ...):
    # See slime/backends/megatron_utils/loss.py for signature
    ...
```
```bash
python train.py --loss-type custom_loss --custom-loss-function-path my_project.my_loss.custom_loss ...
```

### Data Format

Training data is JSONL format. Each line should have:
```json
{"input": "question text", "label": "expected answer"}
```
Or with chat template format:
```json
{"input": [{"role": "user", "content": "question"}], "label": "expected answer"}
```

Configure with `--input-key` and `--label-key` (defaults: `input`, `None`).
Enable chat template with `--apply-chat-template`.

### External Rollout Buffer (Agentic/Async)

For agentic training where external processes generate rollout data:
1. Start `slime_plugins/rollout_buffer/buffer.py` as a server
2. Use `--rollout-buffer-url` to connect slime to the buffer
3. See `slime_plugins/rollout_buffer/rollout_buffer_example.py` for implementation

### Logging and Tracking

- **W&B**: `--use-wandb --wandb-project <name> [--wandb-team <team>]`
- **TensorBoard**: `--use-tensorboard --tb-project-name <dir>`
- Both can be used simultaneously.

### Debugging

```bash
# Debug rollout only (no training)
python train.py --debug-rollout-only ...

# Debug training only (load pre-saved rollout data)
python train.py --load-debug-rollout-data /path/to/rollout_{rollout_id}.pt ...

# Save rollout data for later debugging
python train.py --save-debug-rollout-data /path/to/rollout_{rollout_id}.pt ...

# Dump all details for post-hoc analysis
python train.py --dump-details /path/to/output_dir ...
```

Refer to `docs/en/developer_guide/debug.md` for full debugging guide.

## Important Notes for AI Assistants

1. **FSDP backend is being rewritten** — always prefer Megatron backend for new work. The warning `🚧 FSDP backend is being rewritten` appears at runtime.

2. **SGLang arguments must use `--sglang-` prefix** — e.g., `--mem-fraction-static` from SGLang becomes `--sglang-mem-fraction-static`.

3. **Megatron arguments are passed directly** — standard Megatron flags work without prefix.

4. **`variable_seq_lengths=True` is always set** — slime forces variable-length sequences in Megatron.

5. **`SLIME_BACKEND` env var is deprecated** — use `--train-backend` argument instead.

6. **Weight sync happens before training** — `actor_model.update_weights()` is always called first so SGLang has the latest weights.

7. **The `Sample.remove_sample` flag** — setting this to `True` excludes a sample from loss calculation but does NOT exclude it from advantage normalization.

8. **Custom config YAML** — use `--custom-config-path` to pass a YAML file with additional `args` overrides, useful for complex configurations.

9. **Placement groups use PACK strategy** — all GPUs for a component are placed on the same node(s) for efficiency.

10. **`--rollout-num-gpus-per-engine` = SGLang tensor-parallel size** — this determines how many GPUs each SGLang instance uses.

## Dependencies

Core runtime dependencies (`requirements.txt`):
- `ray[default]` — distributed computing framework
- `sglang-router>=0.2.3` — SGLang router for load balancing
- `transformers` — HuggingFace model loading and tokenization
- `wandb` — experiment tracking
- `tensorboard` — alternative experiment tracking
- `omegaconf` — YAML config loading for eval configs
- `datasets` — dataset loading
- `ring_flash_attn` — efficient attention for long sequences
- `accelerate` — FSDP backend support
- `pillow`, `qwen_vl_utils` — vision-language model support
- `memray` — memory profiling

External dependencies (not in pip, must be installed separately):
- **Megatron-LM** — must be on `PYTHONPATH`
- **SGLang** — inference engine

Python >= 3.10 required.
