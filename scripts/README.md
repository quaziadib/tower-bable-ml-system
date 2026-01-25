# TensorRT-LLM Setup Scripts

This directory contains scripts for integrating TensorRT-LLM into the translation API to achieve **5-10x speed improvement**.

## Quick Start Guide

### Prerequisites

1. **TensorRT-LLM installed:**
   ```bash
   source venv/bin/activate
   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
   pip install nvidia-modelopt polygraphy onnx
   ```

2. **GPU Requirements:**
   - NVIDIA GPU with 16GB+ VRAM (tested on RTX 4090)
   - CUDA 12.x or 13.x
   - ~60GB free disk space

### Step-by-Step Setup

#### Step 1: Merge LoRA Adapter (10-15 minutes)

```bash
python scripts/merge_lora.py
```

This merges the 188MB QLoRA adapter (final_model/) into the 17GB base model (Babel-9B-Chat), creating `Babel-9B-Chat-Merged/`.

**Output:** Babel-9B-Chat-Merged/ (~17GB)

#### Step 2: Convert to TensorRT Format (15-20 minutes)

```bash
python -m tensorrt_llm.models.qwen.convert \
    --model_dir Babel-9B-Chat-Merged \
    --output_dir trt_checkpoints/babel-9b-int8 \
    --dtype bfloat16 \
    --tp_size 1 \
    --pp_size 1
```

This converts the HuggingFace model to TensorRT-LLM checkpoint format.

**Output:** trt_checkpoints/babel-9b-int8/ (~17GB)

#### Step 3: Build TensorRT Engine (20-30 minutes)

**For INT8 (2-3x speedup):**
```bash
trtllm-build \
    --checkpoint_dir trt_checkpoints/babel-9b-int8 \
    --output_dir trt_engines/babel-9b-int8 \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --max_batch_size 8 \
    --max_input_len 512 \
    --max_seq_len 1024 \
    --max_beam_width 1 \
    --strongly_typed
```

**For INT4 (5-10x speedup):**
```bash
trtllm-build \
    --checkpoint_dir trt_checkpoints/babel-9b-int8 \
    --output_dir trt_engines/babel-9b-int4 \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --use_weight_only \
    --weight_only_precision int4 \
    --max_batch_size 8 \
    --max_input_len 512 \
    --max_seq_len 1024 \
    --max_beam_width 1 \
    --strongly_typed
```

**Output:** trt_engines/babel-9b-int8/ (~8GB) or trt_engines/babel-9b-int4/ (~4GB)

#### Step 4: Run API with TensorRT

```bash
# Enable TensorRT-LLM
export USE_TENSORRT=1

# Start API
python api.py
```

The API will automatically use the TensorRT engine for 5-10x faster inference!

### Switching Between Backends

```bash
# Use TensorRT-LLM (fast)
export USE_TENSORRT=1
python api.py

# Use PyTorch (original)
export USE_TENSORRT=0
python api.py
```

## Performance Benchmarking

```bash
python scripts/benchmark.py
```

Expected results:
- **PyTorch FP16**: 1500-2500ms per translation (baseline)
- **TensorRT INT8**: 500-800ms (2-3x faster)
- **TensorRT INT4**: 200-400ms (5-10x faster) ✓ TARGET

## Quality Validation

```bash
python scripts/validate_quality.py
```

Compares PyTorch vs TensorRT outputs on test cases.

**Acceptance criteria:**
- INT8: BLEU degradation < 2%
- INT4: BLEU degradation < 5%

## Troubleshooting

### Issue: TensorRT-LLM installation fails

**Solution:**
```bash
# Try latest version
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Or check available versions
pip index versions tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### Issue: Engine build fails with OOM

**Solution:**
```bash
# Reduce parallelism
trtllm-build --workers 1 ...

# Or clear GPU memory
pkill python
nvidia-smi
```

### Issue: Quality degradation > 5%

**Solution:**
- Use INT8 instead of INT4
- Or use mixed precision: INT8 weights, FP16 activations

### Issue: Slower than expected

**Solution:**
- Check GPU utilization: `nvidia-smi dmon`
- Verify plugins enabled in build command
- Profile with: `nsys profile python api.py`

## File Sizes

| Item | Size | Description |
|------|------|-------------|
| Babel-9B-Chat/ | 17GB | Original base model |
| final_model/ | 188MB | QLoRA adapter |
| Babel-9B-Chat-Merged/ | 17GB | Merged model |
| trt_checkpoints/ | 17GB | TensorRT checkpoint |
| trt_engines/int8/ | 8GB | INT8 engine |
| trt_engines/int4/ | 4GB | INT4 engine |
| **Total** | **~60GB** | Full setup |

## Rollback to PyTorch

If TensorRT has issues:

```bash
# Immediate fallback
export USE_TENSORRT=0

# Clean removal
rm -rf Babel-9B-Chat-Merged/ trt_checkpoints/ trt_engines/
pip uninstall tensorrt-llm nvidia-modelopt
```

Original models (Babel-9B-Chat/, final_model/) remain untouched!

## Additional Scripts

- **merge_lora.py** - Merge QLoRA adapter into base model
- **benchmark.py** - Performance testing
- **validate_quality.py** - Quality validation (TODO)

## Support

For issues, refer to:
- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [Implementation Plan](/home/adib/.claude/plans/magical-wobbling-petal.md)
