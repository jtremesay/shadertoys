# ShaderToys

A collection of demos intended for publication on Shadertoy.com, plus supporting tools for procedural and meta generation.

## Demos

### Bad Apple

Encode a video into a neural network embedded in a GLSL shader. This demo compresses the Bad Apple video into a tiny neural network that runs entirely in shader code, with no custom textures required.

The `bad_apple/` folder contains the full pipeline: video extraction, neural network training, and GLSL shader generation. Anyone can use these tools to create their own neural network video shaders.

See [bad_apple/README.md](bad_apple/README.md) for details.

### Doom

Replay a Doom demo recording entirely within a shader. This recreates the Doom engine's game logic and rendering in GLSL to playback recorded gameplay.

The `doom/` folder contains the demo files and shader implementation.

## Tools

The `src/shadertoys/` package provides reusable utilities for building Shadertoy demos:

- Video frame extraction and data preprocessing
- Neural network training and PyTorch to GLSL conversion
- Shader code generation using Jinja2 templates
- General procedural generation helpers

Install with `uv sync` or `pip install -e .` to access the command-line tools.

More demos coming soon.