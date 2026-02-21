# VSP Stable Diffusion Post-Processor Guide

## Overview

The VSP post-processor now supports Stable Diffusion via Replicate API for intelligent inpainting of detected objects.

## Prerequisites

1. **Replicate API Token**: Set in your environment
   ```bash
   export REPLICATE_API_TOKEN="your_token_here"
   ```
   Or add to `~/.zshrc`:
   ```bash
   echo 'export REPLICATE_API_TOKEN="your_token_here"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Install replicate library** (already done in VSP environment):
   ```bash
   cd /Users/yuantian/code/VisualSketchpad
   source sketchpad_env/bin/activate
   pip install replicate
   ```

## Quick Start

### Basic Usage

```bash
python request.py \
  --mode comt_vsp \
  --model "qwen/qwen3-vl-235b-a22b-instruct" \
  --comt_sample_id deletion-0107 \
  --max_tasks 1 \
  --vsp_postproc \
  --vsp_postproc_backend sd
```

### With Custom Prompt

```bash
python request.py \
  --mode comt_vsp \
  --max_tasks 5 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --vsp_postproc_sd_prompt "intelligently remove the detected objects and fill with contextually appropriate background" \
  --vsp_postproc_sd_negative_prompt "blurry, distorted, artifacts, unrealistic"
```

### Using Different SD Model

```bash
python request.py \
  --mode vsp \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --vsp_postproc_sd_model "stability-ai/stable-diffusion-inpainting"
```

### Fine-tuning Generation Parameters

```bash
python request.py \
  --mode comt_vsp \
  --max_tasks 5 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --vsp_postproc_sd_num_steps 50 \
  --vsp_postproc_sd_guidance_scale 9.0 \
  --vsp_postproc_sd_prompt "seamlessly remove objects, natural lighting"
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vsp_postproc_backend` | `ask` | Use `sd` for Stable Diffusion |
| `--vsp_postproc_sd_model` | `stability-ai/stable-diffusion-inpainting` | Replicate model ID |
| `--vsp_postproc_sd_prompt` | `remove the objects, fill with natural background` | Inpainting instruction |
| `--vsp_postproc_sd_negative_prompt` | `blurry, distorted, artifacts` | What to avoid |
| `--vsp_postproc_sd_num_steps` | `50` | Inference steps (higher = better quality, slower) |
| `--vsp_postproc_sd_guidance_scale` | `7.5` | How closely to follow prompt (1-20) |

## How It Works

1. **Detection**: VSP's vision tools detect objects and return bounding boxes
2. **Mask Creation**: SD processor creates a white mask over detected regions
3. **Replicate API Call**: Sends image + mask + prompt to Stable Diffusion
4. **Inpainting**: SD fills masked regions intelligently based on prompt
5. **Return**: LLM receives the inpainted image (transparently)

## Output Structure

When SD post-processing is enabled:

```
output/job_XXX/details/vsp_*/category/task_id/output/input/
├── image_0.jpg                          # Original input
├── before_postproc_detection_*.png      # VSP-annotated (before SD)
└── <hash>.png                           # SD-inpainted image (what LLM sees)
```

## Comparison: ASK vs SD

| Feature | ASK (Agent-ScanKit) | SD (Stable Diffusion) |
|---------|---------------------|----------------------|
| **Speed** | ⚡ Very fast (<1s) | 🐢 Slower (10-30s per image) |
| **Quality** | Simple masking/inpainting | 🎨 Intelligent, context-aware |
| **Cost** | 💰 Free (local) | 💳 Paid (Replicate API ~$0.01-0.10/image) |
| **Internet** | ❌ Not required | ✅ Required |
| **Use Case** | Quick testing, masking | Production, high-quality removal |

## Troubleshooting

### Error: REPLICATE_API_TOKEN not set

```bash
export REPLICATE_API_TOKEN="your_token_here"
```

### Error: replicate library not installed

```bash
cd /Users/yuantian/code/VisualSketchpad
source sketchpad_env/bin/activate
pip install replicate
```

### Error: Rate limit (429) - "Request was throttled"

**Cause**: Account has less than $5 credit, limited to 6 requests/minute with burst=1

**Solutions**:
1. **Wait a few seconds** between requests (automatic retry built-in)
2. **Add $5+ credit** to your Replicate account: https://replicate.com/account/billing
   - Removes rate limit restrictions
   - Recommended for batch processing

**Note**: The code now automatically retries with exponential backoff when rate limited.

### SD API call takes too long

- Reduce `--vsp_postproc_sd_num_steps` (e.g., to 20)
- Use a faster model (if available on Replicate)

### Poor inpainting quality

- Adjust `--vsp_postproc_sd_prompt` with more specific instructions
- Increase `--vsp_postproc_sd_num_steps` (e.g., to 50)
- Adjust `--vsp_postproc_sd_guidance_scale` (try 8-10)

## Debug Logs

Check VSP debug logs to see SD processing:

```bash
grep "POST_PROCESSOR" output/job_*/details/vsp_*/*/*/output/vsp_debug.log
```

Expected output:
```
[POST_PROCESSOR] SD:inpaint via Replicate
[POST_PROCESSOR] Using Replicate model: stability-ai/stable-diffusion-inpainting
[POST_PROCESSOR] Prompt: remove the objects, fill with natural background
[POST_PROCESSOR] Calling Replicate API...
[POST_PROCESSOR] Replicate API call successful
```

## Cost Estimation

Replicate SD 3.5 Large pricing (approximate):
- ~$0.055 per image for default settings (28 steps)
- 100 images ≈ $5.50
- 1000 images ≈ $55

For testing, use `--max_tasks 5` to limit costs.

## Examples

### Test with CoMT

```bash
python request.py \
  --mode comt_vsp \
  --model "qwen/qwen3-vl-235b-a22b-instruct" \
  --comt_sample_id deletion-0107 \
  --max_tasks 1 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --skip_eval
```

### Batch Processing with SD

```bash
python request.py \
  --mode vsp \
  --categories "01-Illegal_Activity" "08-Political_Lobbying" \
  --max_tasks 20 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --vsp_postproc_sd_prompt "remove detected objects naturally"
```

### Compare ASK vs SD

Run two jobs and compare:

```bash
# ASK (fast)
python request.py --mode vsp --max_tasks 5 --vsp_postproc --vsp_postproc_backend ask --vsp_postproc_method visual_edit

# SD (high quality)
python request.py --mode vsp --max_tasks 5 --vsp_postproc --vsp_postproc_backend sd
```

## Notes

- SD post-processing is **transparent to the LLM** - the model sees the inpainted image but doesn't know it was processed
- "Before" images are automatically saved for comparison
- First API call may be slower due to model cold start on Replicate
- Consider using ASK for quick tests, SD for production runs
