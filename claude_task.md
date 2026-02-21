# Task: Integrate CoMT-VSP with Self-Hosted Qwen for Hidden States Extraction

## Goal

Run the CoMT-VSP agent pipeline against self-hosted open-source VL models on AutoDL, capturing the **last-token hidden state** at each inference call. The API server is **model-agnostic** — works with any HuggingFace VL model by just changing the model path.

## Current State

### Mediator → VisualSketchPad (existing flow)
```
Mediator (provider.py ComtVspProvider)
  → subprocess: VisualSketchPad/agent/main.py run_agent(..., model="xxx")
    → config.py: base_url configurable via LLM_BASE_URL env var (default: OpenRouter)
      → AutoGen OpenAIWrapper → HTTP POST to LLM endpoint
    → Vision experts (SOM/GroundingDINO/DepthAnything) → HTTP to remote servers
```

### AutoDL Instance (`ssh seetacloud`)
- **Services running**:
  - **SOM** (Semantic-SAM) — port 7862
  - **GroundingDINO** — port 7860
  - **Depth Anything** — port 7861
  - **Qwen3-VL-8B-Instruct** (hidden states server) — port 8000
- **Key constraint**: AutoDL ports are **NOT publicly accessible**. Must use SSH port forwarding to access from Mac.

### Port Forwarding (required)

AutoDL doesn't expose ports externally. Use SSH tunnels to map remote ports to localhost:

```bash
# Forward all 4 services in one command (background):
ssh -L 17860:localhost:7860 \
    -L 17861:localhost:7861 \
    -L 17862:localhost:7862 \
    -L 18000:localhost:8000 \
    seetacloud -N &

# Then access services via localhost:
#   GroundingDINO  → http://localhost:17860
#   Depth Anything → http://localhost:17861
#   SOM            → http://localhost:17862
#   Qwen LLM      → http://localhost:18000
```

### Config Changes Needed

**`~/code/VisualSketchPad/agent/config.py`** — Vision expert addresses currently point to old AWS IP:
```python
# OLD (AWS):
SOM_ADDRESS = "http://34.210.214.193:7862"
GROUNDING_DINO_ADDRESS = "http://34.210.214.193:7860"
DEPTH_ANYTHING_ADDRESS = "http://34.210.214.193:7861"

# NEW (via SSH tunnel):
SOM_ADDRESS = "http://localhost:17862"
GROUNDING_DINO_ADDRESS = "http://localhost:17860"
DEPTH_ANYTHING_ADDRESS = "http://localhost:17861"
```

**Mediator LLM endpoint** — pass `--llm_base_url http://localhost:18000/v1` instead of the old AWS IP.

## Approach: Model-Agnostic API Server on AutoDL

Server serves OpenAI-compatible `/v1/chat/completions` with hidden state extraction.
AutoGen ignores the extra `hidden_state` field (it only reads `choices`).

### API Response

```json
{
    "choices": [{"message": {"content": "Hello"}, ...}],
    "hidden_state": {
        "last_token": [0.123, -0.456, ...],
        "layer": -1,
        "hidden_dim": 4096,
        "model": "Qwen3-VL-8B-Instruct"
    }
}
```

## Implementation Plan

### Step 1: Build the model-agnostic API server — DONE
**Location**: AutoDL server (hidden_states project)

- FastAPI + uvicorn server, accepts `--model_path`, `--device_map`, `--host`, `--port`
- Endpoints: `/v1/chat/completions`, `/v1/models`, `/health`
- Generates response via `model.generate(output_hidden_states=True, return_dict_in_generate=True)`
- Returns OpenAI-compatible response + `hidden_state` with last-token vector
- Handles multimodal inputs (base64 images decoded to PIL)
- Migrated from AWS to AutoDL

### Step 2: Make VisualSketchPad's LLM config overridable — DONE
**File**: `~/code/VisualSketchPad/agent/config.py`

Made `base_url` and `api_key` configurable via `LLM_BASE_URL` and `LLM_API_KEY` environment variables,
falling back to OpenRouter defaults (`OPENROUTER_API_KEY` / `https://openrouter.ai/api/v1`).

### Step 3: Pass endpoint config from Mediator to VSP subprocess — DONE
**File**: `~/code/Mediator/provider.py` — `VSPProvider._call_vsp()`

Passes `LLM_BASE_URL` and `LLM_API_KEY` as environment variables to the subprocess.
When `llm_base_url` is set but `llm_api_key` is not, defaults to `"not-needed"`.

### Step 4: Add CLI arguments in Mediator — DONE
**File**: `~/code/Mediator/request.py`

Added `--llm_base_url` and `--llm_api_key` CLI arguments + `RunConfig` fields.

### Step 5: Migrate services from AWS to AutoDL — DONE
All 4 services (SOM, GroundingDINO, Depth Anything, Qwen3-VL-8B) migrated to AutoDL.

### Step 6: Update vision expert addresses for SSH tunneling — DONE
**File**: `~/code/VisualSketchPad/agent/config.py`

Update `SOM_ADDRESS`, `GROUNDING_DINO_ADDRESS`, `DEPTH_ANYTHING_ADDRESS` to use localhost
with forwarded ports (17862, 17860, 17861). Ideally make these configurable via environment
variables like the LLM endpoint.

### Step 7: Capture hidden states in Mediator — DONE
**Files**:
- `~/code/VisualSketchPad/agent/multimodal_conversable_agent.py` — capture `hidden_state` from API response in `generate_oai_reply()`
- `~/code/VisualSketchPad/agent/main.py` — save captured hidden states to `hidden_states.json` in task output directory
- `~/code/Mediator/provider.py` — `_save_hidden_states()` reads JSON, saves per-turn `.npy` files + `_turns.json` metadata to `{job_folder}/hidden_states/`

Also slimmed down `_save_vsp_metadata()` to remove redundant `prompt_struct` (base64 images) and `vsp_result` (duplicate of output.json), reducing per-task metadata from ~160KB to ~2KB.

## Files Modified/Created

| Location | File | Status |
|----------|------|--------|
| AutoDL | hidden_states `server.py` | Done (migrated from AWS) |
| AutoDL | SOM server (port 7862) | Done |
| AutoDL | GroundingDINO server (port 7860) | Done |
| AutoDL | Depth Anything server (port 7861) | Done |
| Local | `~/code/VisualSketchPad/agent/config.py` | Done (Step 2), TODO (Step 6: update addresses) |
| Local | `~/code/VisualSketchPad/agent/multimodal_conversable_agent.py` | Done (Step 7: hidden state capture) |
| Local | `~/code/VisualSketchPad/agent/main.py` | Done (Step 7: hidden state save to JSON) |
| Local | `~/code/Mediator/provider.py` | Done (Step 3, 7: hidden states + metadata slimming) |
| Local | `~/code/Mediator/request.py` | Done (Step 4) |

## Usage

```bash
# 1. Start SSH port forwarding to AutoDL
ssh -L 17860:localhost:7860 \
    -L 17861:localhost:7861 \
    -L 17862:localhost:7862 \
    -L 18000:localhost:8000 \
    seetacloud -N &

# 2. Verify services are accessible
curl http://localhost:18000/health
curl http://localhost:17862  # SOM
curl http://localhost:17860  # GroundingDINO
curl http://localhost:17861  # Depth Anything

# 3. Test Qwen LLM
curl -s http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-8B-Instruct","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0.7}'

# 4. Run Mediator pointing to AutoDL (via SSH tunnel)
python request.py --mode comt_vsp --model "Qwen3-VL-8B-Instruct" \
  --llm_base_url "http://localhost:18000/v1" \
  --comt_sample_id "deletion-0107" --max_tasks 10
```

## SSH Access

```
Host seetacloud
    HostName connect.nma1.seetacloud.com
    User root
    Port 13132
    IdentityFile ~/.ssh/seetacloud_key
```

## Open Issues

- **Vision expert addresses**: `config.py` still hardcodes old AWS IP (34.210.214.193). Need to update
  to localhost tunnel ports (Step 6). Best approach: make them env-var configurable like the LLM endpoint.
- **Larger models**: Qwen3-VL-30B-A3B or larger models may need more VRAM depending on AutoDL GPU tier.
