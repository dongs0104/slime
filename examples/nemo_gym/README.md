# NeMo Gym Integration Example for Slime

This directory contains examples for using NeMo Gym environments with Slime.

## Quick Start

### 1. Install Dependencies

```bash
# Install Slime
pip install -e .

# Install NeMo Gym from 3rdparty
pip install -e 3rdparty/nemo-gym
```

### 2. Create env.yaml (Optional)

Create `env.yaml` in Slime root or `3rdparty/nemo-gym/` with API keys:

```yaml
# env.yaml
google_api_key: "your-google-api-key"
google_cx: "your-custom-search-engine-id"
hf_token: "your-huggingface-token"
```

### 3. Run Training

**Option A: Auto-start servers (Recommended)**

```bash
python train.py \
  --use-nemo-gym \
  --nemo-gym-config configs/nemo_gym_unified.yaml \
  --hf-checkpoint Qwen/Qwen3-4B-Instruct-2507 \
  --rollout-batch-size 32 \
  --num-rollout 100
```

**Option B: Manual server start**

```bash
# Terminal 1: Start NeMo Gym servers
cd 3rdparty/nemo-gym
ng_run "+config_paths=[resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml]"

# Terminal 2: Run Slime training
python train.py \
  --use-nemo-gym \
  --nemo-gym-config configs/nemo_gym_unified.yaml \
  --hf-checkpoint Qwen/Qwen3-4B-Instruct-2507 \
  --num-rollout 100
```

## Configuration

See `configs/nemo_gym_unified.yaml` for full configuration options:

```yaml
nemo_gym:
  # Auto-start NeMo Gym servers
  auto_start_servers: true
  
  # Config paths (relative to 3rdparty/nemo-gym)
  config_paths:
    - "resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
  
  # Environments for Unified RLVR
  environments:
    - name: "math_with_judge"
      agent_name: "math_with_judge_simple_agent"
      weight: 1.0
```

## NeMo Gym Environment Types

Different environments use different endpoints:

| Environment | Endpoints | Description |
|-------------|-----------|-------------|
| math_with_judge | `/run`, `/verify` | Math problem solving with verification |
| google_search | `/run`, `/search`, `/browse`, `/verify` | Web search with tool calls |
| coding | `/run`, `/verify` | Code generation and verification |
| if_eval | `/run`, `/verify` | Instruction following evaluation |

## Unified RLVR (Multi-Environment Training)

Train on multiple environments simultaneously:

```yaml
nemo_gym:
  environments:
    - name: "math_with_judge"
      weight: 0.4  # 40% of samples
    - name: "coding"
      weight: 0.3  # 30% of samples
    - name: "if_eval"
      weight: 0.3  # 30% of samples
```

```bash
python train.py \
  --use-nemo-gym \
  --nemo-gym-config configs/nemo_gym_unified.yaml \
  --unified-rlvr \
  --hf-checkpoint Qwen/Qwen3-4B-Instruct-2507
```

## Troubleshooting

### "Could not find policy_model"

This error means NeMo Gym can't find the policy model server. When running with Slime, Slime's SGLang server acts as the policy model. Make sure:

1. Slime's server is running
2. Environment config references the correct model

### "No config_paths specified"

Add `config_paths` to your YAML or provide environments with `config` field:

```yaml
nemo_gym:
  config_paths:
    - "resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
```

### Import errors

Make sure nemo-gym is installed:

```bash
pip install -e 3rdparty/nemo-gym
```
