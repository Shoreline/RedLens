#!/bin/bash
# Test Stable Diffusion Post-Processor

echo "================================"
echo "Testing SD Post-Processor"
echo "================================"
echo ""

# Source .zshrc to get REPLICATE_API_TOKEN
source ~/.zshrc

# Check if token is set
if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "❌ ERROR: REPLICATE_API_TOKEN not set"
    echo "Please add to ~/.zshrc:"
    echo "  export REPLICATE_API_TOKEN='your_token_here'"
    exit 1
else
    echo "✅ REPLICATE_API_TOKEN is set"
    echo "Token: ${REPLICATE_API_TOKEN:0:10}..."
fi

echo ""
echo "Running test with 1 task..."
echo ""

cd /Users/yuantian/code/RedLens

# Activate venv
source venv/bin/activate

# Run test
python request.py \
  --provider comt_vsp \
  --model "qwen/qwen3-vl-235b-a22b-instruct" \
  --comt_sample_id deletion-0107 \
  --max_tasks 1 \
  --vsp_postproc \
  --vsp_postproc_backend sd \
  --vsp_postproc_sd_prompt "remove the detected people, fill with natural background" \
  --skip_eval

echo ""
echo "================================"
echo "Test Complete!"
echo "================================"
echo ""
echo "Check output in: output/job_*/"
echo "Look for SD processing logs:"
echo "  grep 'POST_PROCESSOR.*SD' output/job_*/details/vsp_*/*/*/output/vsp_debug.log"
