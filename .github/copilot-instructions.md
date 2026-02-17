# Shadertools Project - AI Agent Instructions

## Project Overview

Neural video compression for Shadertoy using PyTorch → GLSL code generation. Compresses Bad Apple video (1.08 GB) into a tiny neural network (~17 KB) that runs entirely in GLSL shader code, with NO custom textures (Shadertoy constraint).

**Core Concept**: Train `f(frame, x, y) → pixel_value` then embed the NN weights directly in GLSL code for real-time playback.

## Architecture & Data Flow

```
video.webm (480×360, 6572 frames)
    ↓ [extract_pixels.py → video.py]
video_pixels.parquet (1.08 GB Polars DataFrame)
    ↓ [train_nn.py → nn.py: TinyVideoNet]
nn_weights.npz + metadata.json (4353 params, ~17 KB)
    ↓ [generate_shaders.py + Jinja2 templates]
shadertoy_buffer_a.fs + shadertoy_image.fs
    ↓ [Manual upload to Shadertoy.com]
Multi-pass shader (Buffer A stores weights, Image runs NN inference)
```

## Critical Constraints

1. **Shadertoy 65K character limit**: Total GLSL code must fit in ~65K chars
2. **No custom textures**: Must embed all data (NN weights) as GLSL `const float[]` arrays
3. **Multi-pass architecture**: Buffer A stores weights as texture, Image reads via `iChannel0`
4. **Normalization**: All inputs/outputs normalized to [0, 1] for both PyTorch and GLSL

## Key Components

### 1. Video Extraction (`src/shadertools/video.py`)
- Uses OpenCV to read video frames
- Converts to grayscale, extracts all pixels
- Pre-allocates NumPy arrays for vectorized operations (fast!)
- Outputs Polars DataFrame with columns: `frame`, `x`, `y`, `pixel_value`

### 2. Neural Network (`src/shadertools/nn.py`)

**TinyVideoNet Architecture**: `[3] → [32] → [64] → [32] → [1]`
- Input: `(frame_norm, x_norm, y_norm)` normalized to [0, 1]
- Hidden: Fully-connected layers with ReLU
- Output: Single pixel value via Sigmoid → [0, 1]
- Training: MSE loss, Adam optimizer, 5% sample rate

**Critical Pattern**: `VideoDataset` normalizes coordinates by dividing by max values:
```python
self.frames /= total_frames
self.xs /= width
self.ys /= height
self.pixels /= 255.0  # Grayscale normalization
```

### 3. GLSL Generation (`src/shadertools/bin/generate_shaders.py`)

**Jinja2 Templates** (`src/shadertools/templates/`):
- `buffer_a.fs`: Embeds weights as `const float NN_WEIGHTS[N]`, packs 4 per RGBA pixel
- `image.fs`: Reads weights via `texelFetch()`, implements NN forward pass

**Weight Encoding**:
1. Flatten all PyTorch parameters into single array
2. Generate GLSL array literal: `float[4353](w1, w2, ...)`
3. Pack into texture: 4 values per RGBA pixel
4. Store denormalization: `weight * 0.5 + 0.5` → `pixel * 2.0 - 1.0`

**NN Inference in GLSL**:
- Loop-based matrix multiplication (no GLSL matrix helpers)
- Manual ReLU: `max(0.0, x)`, Sigmoid: `1.0 / (1.0 + exp(-x))`
- Reads weights sequentially using offset tracking

## Console Scripts (Entry Points)

Install with `uv sync` or `pip install -e .` to get:

1. **`shadertools_extract_pixels`** → `src/shadertools/bin/extract_pixels.py`
   - Default: `-i video.webm -o video_pixels.parquet`
   
2. **`shadertools_train_nn`** → `src/shadertools/bin/train_nn.py`
   - Default: `-i video_pixels.parquet -o nn_weights.json`
   - Outputs: `.json`, `.npz`, `_metadata.json`
   
3. **`shadertools_generate_shaders`** → `src/shadertools/bin/generate_shaders.py`
   - Default: `-i nn_weights_tiny.npz`
   - Outputs: `shadertoy_buffer_a.fs`, `shadertoy_image.fs`

## Development Workflows

### Full Pipeline (from `bad_apple/` directory)
```bash
cd bad_apple
shadertools_extract_pixels  # 5-10 min
shadertools_train_nn        # 1-2 hours CPU, 10-15 min GPU
shadertools_generate_shaders nn_weights_tiny.npz  # instant
```

### Quick Testing (Reduce Training Time)
In `train_nn.py`, modify architecture config:
```python
"sample_rate": 0.02,  # 2% instead of 5%
"epochs": 10,         # 10 instead of 30
```

Or filter data for fewer frames:
```python
df = df.filter(pl.col("frame") < 100)  # First 100 frames only
```

### Debugging GLSL Output

**Check buffer sizes**:
```python
# In generate_shaders.py
total_weights = sum(p.size for p in weights_dict.values())
tex_size = int(np.ceil(np.sqrt(total_weights / 4)))  # 4 weights per pixel
```

**Verify normalization match**: PyTorch training uses same ranges as GLSL inference:
- Python: `frames /= total_frames` ↔ GLSL: `float(frame) / float(TOTAL_FRAMES)`
- Python: `pixels /= 255.0` ↔ GLSL: `sigmoid()` output already in [0, 1]

## Project Conventions

1. **Polars over Pandas**: All DataFrame operations use Polars for speed
2. **Type hints everywhere**: Python 3.13+ with modern type annotations
3. **Console scripts in pyproject.toml**: All CLIs defined as `project.scripts`
4. **Working directory matters**: Run from `bad_apple/` for default paths to work
5. **Jinja2 for code generation**: Never manually write GLSL arrays
6. **NumPy intermediate format**: `.npz` files bridge PyTorch ↔ GLSL

## Common Pitfalls

❌ **Don't** manually edit generated `.fs` files (regenerate from templates)  
❌ **Don't** forget to connect Buffer A to iChannel0 in Shadertoy Image tab  
❌ **Don't** exceed 65K chars (monitor with `len(buffer_a) + len(image_shader)`)  
❌ **Don't** use different normalization in Python vs GLSL (causes black screens)  
✅ **Do** run commands from `bad_apple/` directory (or specify full paths)  
✅ **Do** check GPU availability: `torch.cuda.is_available()` (10x speedup)  
✅ **Do** verify metadata file exists alongside `.npz` for shader generation  

## Quick Reference

| File | Purpose |
|------|---------|
| `src/shadertools/nn.py` | PyTorch model definition + training logic |
| `src/shadertools/video.py` | OpenCV video → Polars DataFrame |
| `src/shadertools/templates/*.fs` | Jinja2 templates for GLSL generation |
| `bad_apple/video_pixels.parquet` | Extracted pixel data (generated) |
| `bad_apple/nn_weights_tiny.npz` | Trained weights (generated) |
| `bad_apple/shadertoy_*.fs` | Final GLSL shaders (generated) |

## External Dependencies

- **Shadertoy.com**: Final deployment target (multi-pass shader setup required)
- **PyTorch**: Training infrastructure (GPU optional but recommended)
- **opencv-python-headless**: Video I/O (headless = no GUI dependencies)
- **Polars**: Fast DataFrame operations (replaces Pandas)
- **Jinja2**: Template engine for GLSL code generation
