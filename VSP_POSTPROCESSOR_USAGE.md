# VSP Post-Processor Control from Mediator

## Overview

You can now control VSP's post-processing framework directly from Mediator's `request.py` command line. This allows you to dynamically enable/disable post-processing and choose different methods without modifying VSP's config files.

## Command-Line Arguments

### `--vsp_postproc`
Enable VSP post-processing (default: False)

### `--vsp_postproc_backend`
Choose post-processing backend (default: "ask")
- `ask`: Agent-ScanKit (local image manipulation)
- `sd`: Stable Diffusion (future, not yet implemented)

### `--vsp_postproc_method`
Choose ASK processing method (default: None, uses VSP's config.py default)
- `visual_mask`: Draw black rectangles over detected regions
- `visual_edit`: Inpaint detected regions using OpenCV
- `zoom_in`: Crop and zoom into detected regions

## Usage Examples

### Example 1: Enable Visual Masking

Mask detected objects with black rectangles:

```bash
python request.py \
  --mode vsp \
  --model "gpt-5" \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method visual_mask
```

### Example 2: Enable Inpainting

Remove detected objects using OpenCV inpainting:

```bash
python request.py \
  --mode vsp \
  --model "gpt-5" \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method visual_edit
```

### Example 3: Enable Zoom-In

Zoom into the first detected region:

```bash
python request.py \
  --mode vsp \
  --model "gpt-5" \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method zoom_in
```

### Example 4: Use with CoMT-VSP

Post-processing also works with CoMT-VSP provider:

```bash
python request.py \
  --mode comt_vsp \
  --model "gpt-5" \
  --comt_sample_id "creation-10003" \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method visual_mask
```

### Example 5: Disable Post-Processing

Run without post-processing (default behavior):

```bash
python request.py \
  --mode vsp \
  --model "gpt-5" \
  --max_tasks 10
# No --vsp_postproc flag = post-processing disabled
```

## How It Works

### Data Flow

```
Mediator (request.py)
    ↓ (command-line args)
RunConfig
    ↓ (vsp_postproc_* fields)
VSPProvider
    ↓ (environment variables)
VSP config.py
    ↓ (POST_PROCESSOR_CONFIG)
VSP vision tools (detection, segment_and_mark)
    ↓ (apply_postprocess)
Modified images
```

### Environment Variables

Mediator passes settings to VSP via environment variables:
- `VSP_POSTPROC_ENABLED`: "1" (enabled) or "0" (disabled)
- `VSP_POSTPROC_BACKEND`: "ask" or "sd"
- `VSP_POSTPROC_METHOD`: "visual_mask", "visual_edit", or "zoom_in"

VSP's `config.py` reads these environment variables and configures the post-processor accordingly.

## Technical Details

### Modified Files

**Mediator:**
- `request.py`: Added command-line arguments and RunConfig fields
- `provider.py`: Modified VSPProvider._call_vsp() to pass environment variables

**VSP:**
- `agent/config.py`: Modified POST_PROCESSOR_CONFIG to read from environment variables

### Transparency to LLM

Post-processing is completely transparent to the LLM:
- ✅ LLM **sees** the modified images
- ✅ LLM **reasons** about modifications
- ✅ LLM **doesn't need to know** about post-processing
- ✅ No prompt changes required

### When to Use Each Method

**visual_mask** (Black Rectangles):
- Good for: Testing detection accuracy
- Use case: Verify that detected regions are correctly identified
- Effect: Draws black rectangles over detected objects

**visual_edit** (Inpainting):
- Good for: Content-aware removal
- Use case: Remove detected objects while preserving background
- Effect: Intelligently fills in detected regions

**zoom_in** (Crop & Zoom):
- Good for: Focusing on specific regions
- Use case: Analyze detected objects in detail
- Effect: Crops image to first detected region with padding

## Validation

### Test the Setup

1. Run a simple test with masking:
```bash
python request.py \
  --mode vsp \
  --max_tasks 2 \
  --vsp_postproc \
  --vsp_postproc_method visual_mask
```

2. Check the output images in the job folder:
```bash
# Images should show detected objects masked with black rectangles
ls output/job_*/details/vsp_*/*/*/output/*.png
```

3. Verify in VSP debug logs:
```bash
# Should see "[POST_PROCESSOR] ASK:visual_mask"
grep "POST_PROCESSOR" output/job_*/details/vsp_*/*/*/output/vsp_debug.log
```

## Troubleshooting

### Post-processing not working

1. **Check provider**: Only works with `vsp` and `comt_vsp` providers
2. **Check flag**: Make sure `--vsp_postproc` is set
3. **Check logs**: Look for `[POST_PROCESSOR]` in VSP debug logs
4. **Check environment**: Run `env | grep VSP_POSTPROC` in the shell

### Images unchanged

1. **Vision tools not called**: Post-processing only applies when VSP uses vision tools (detection, segment_and_mark)
2. **No objects detected**: If no bboxes are found, some methods (like zoom_in) may not modify the image
3. **Method not specified**: If no method is specified, VSP uses the default from config.py

## Examples in Practice

### Scenario 1: Safety Evaluation with Object Removal

Remove detected harmful objects before showing to LLM:

```bash
python request.py \
  --mode vsp \
  --categories "01-Illegal_Activity" \
  --max_tasks 20 \
  --vsp_postproc \
  --vsp_postproc_method visual_edit
```

### Scenario 2: Compare With/Without Post-Processing

Run two jobs to compare:

```bash
# Without post-processing
python request.py --mode vsp --max_tasks 10

# With visual masking
python request.py --mode vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_mask
```

Then compare the results in the eval.csv files.

## Before/After Image Saving

When post-processing is enabled, **two versions** of the image are automatically saved:

### 1. Before Post-Processing Image

**Filename format**: `before_postproc_{tool_name}_{timestamp}.png`

This is the annotated image from VSP's vision tool (detection/segment_and_mark):
- ✅ Shows bounding boxes, labels, masks from VSP
- ✅ Vision analysis complete
- ❌ NOT yet modified by post-processor

### 2. After Post-Processing Image

**Filename format**: `{hash}.png` (e.g., `5c87f9fe552b46d088ac1f1f27fa9af2.png`)

This is the final image after ASK post-processing:
- ✅ Shows VSP's vision analysis
- ✅ Modified by post-processor (masked, inpainted, or zoomed)
- ✅ This is what the LLM sees

### Example Directory Structure

```
output/job_111_tasks_1_ComtVsp_.../details/vsp_.../08-Political_Lobbying/0/output/input/
├── before_postproc_detection_123456.png  ← Before: VSP annotated only
└── 5c87f9fe552b46d088ac1f1f27fa9af2.png  ← After: Post-processed
```

### Use Cases

**Debugging**: Compare what VSP detected vs. what was post-processed

**Analysis**: Verify post-processing correctness

**Research**: Study the impact of post-processing on LLM reasoning

## Benefits

1. ✅ **Dynamic control**: No need to edit VSP config files
2. ✅ **Reproducible**: Command-line args are logged in metadata.yaml
3. ✅ **Flexible**: Easy to run experiments with different settings
4. ✅ **Clean**: VSP codebase unchanged, settings passed via environment
5. ✅ **Documented**: All settings visible in command history
6. ✅ **Before/After comparison**: Both images saved automatically

## Future Enhancements

- [ ] Support Stable Diffusion backend (--vsp_postproc_backend sd)
- [ ] Add method-specific parameters (e.g., --vsp_mask_padding 30)
- [ ] Support chaining multiple post-processors
- [ ] Add per-category post-processing rules
