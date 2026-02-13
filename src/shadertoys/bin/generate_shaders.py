import json
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader("shadertoys"))


def load_weights(weights_path):
    """Load NN weights from npz file."""
    data = np.load(weights_path)
    weights = {key: data[key] for key in data.files}
    return weights


def generate_buffer_a(weights_dict, metadata):
    """Generate Buffer A shader that encodes weights as a texture."""

    # Linearize all weights
    all_weights = []
    offsets = {}
    current_offset = 0

    sorted_keys = sorted(weights_dict.keys())
    for key in sorted_keys:
        param = weights_dict[key]
        flat = param.flatten()
        all_weights.extend(flat)
        offsets[key] = {
            "offset": current_offset,
            "size": len(flat),
            "shape": list(param.shape),
        }
        current_offset += len(flat)

    total_weights = len(all_weights)
    print(f"Total weights: {total_weights:,}")

    # Calculate texture size needed
    values_per_pixel = 4  # RGBA
    num_pixels = (total_weights + values_per_pixel - 1) // values_per_pixel
    tex_size = int(np.ceil(np.sqrt(num_pixels)))

    tpl = env.get_template("buffer_a.fs")
    return (
        tpl.render(total_weights=total_weights, tex_size=tex_size, weights=all_weights),
        offsets,
        tex_size,
    )


def generate_image_shader(metadata, offsets, tex_size):
    """Generate main Image shader that performs NN inference."""

    # Extract architecture info
    hidden_sizes = metadata.get("hidden_sizes", [32, 64, 32])
    width = metadata.get("width", 480)
    height = metadata.get("height", 360)
    total_frames = metadata.get("total_frames", 6572)

    # Calculate layer info
    layer_info = []
    input_size = 3
    for i, hidden_size in enumerate(hidden_sizes):
        weight_size = input_size * hidden_size
        bias_size = hidden_size
        layer_info.append(
            {
                "index": i,
                "input": input_size,
                "output": hidden_size,
                "weight_size": weight_size,
                "bias_size": bias_size,
            }
        )
        input_size = hidden_size

    # Final layer
    layer_info.append(
        {
            "index": len(hidden_sizes),
            "input": input_size,
            "output": 1,
            "weight_size": input_size * 1,
            "bias_size": 1,
        }
    )

    shader_code = f"""// Bad Apple NN - Image: Neural Network Inference
// Reads weights from Buffer A (iChannel0) and performs forward pass

const int VIDEO_WIDTH = {width};
const int VIDEO_HEIGHT = {height};
const int TOTAL_FRAMES = {total_frames};
const int TEXTURE_SIZE = {tex_size};

// Read weight from Buffer A texture
float readWeight(int index) {{
    int pixel_idx = index / 4;
    int channel = index % 4;
    
    int tex_y = pixel_idx / TEXTURE_SIZE;
    int tex_x = pixel_idx % TEXTURE_SIZE;
    
    vec2 uv = (vec2(tex_x, tex_y) + 0.5) / float(TEXTURE_SIZE);
    vec4 pixel = texelFetch(iChannel0, ivec2(tex_x, tex_y), 0);
    
    // Denormalize from [0, 1] to [-1, 1]
    float value = pixel[channel] * 2.0 - 1.0;
    return value;
}}

// Activation functions
float relu(float x) {{
    return max(0.0, x);
}}

float sigmoid(float x) {{
    return 1.0 / (1.0 + exp(-x));
}}

// Neural network forward pass
float neuralNetwork(vec3 input) {{
    // Architecture: {hidden_sizes} -> 1
    
    int offset = 0;
"""

    # Generate forward pass code for each layer
    prev_layer_var = "input"
    prev_layer_size = 3

    for i, layer in enumerate(layer_info[:-1]):  # All hidden layers
        next_size = layer["output"]
        layer_var = f"hidden{i + 1}"

        shader_code += f"""
    // Layer {i}: [{layer["input"]}] -> [{layer["output"]}]
    float {layer_var}[{next_size}];
    for (int i = 0; i < {next_size}; i++) {{
        float sum = 0.0;
        for (int j = 0; j < {layer["input"]}; j++) {{
            sum += {prev_layer_var}[j] * readWeight(offset + i * {layer["input"]} + j);
        }}
        sum += readWeight(offset + {layer["weight_size"]} + i);
        {layer_var}[i] = relu(sum);
    }}
    offset += {layer["weight_size"] + layer["bias_size"]};
"""
        prev_layer_var = layer_var
        prev_layer_size = next_size

    # Final layer
    final_layer = layer_info[-1]
    shader_code += f"""
    // Output layer: [{final_layer["input"]}] -> [1]
    float output = 0.0;
    for (int i = 0; i < {final_layer["input"]}; i++) {{
        output += {prev_layer_var}[i] * readWeight(offset + i);
    }}
    output += readWeight(offset + {final_layer["weight_size"]});
    output = sigmoid(output);
    
    return output;
}}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    // Calculate current frame based on time (30 fps)
    int frame = int(iTime * 30.0) % TOTAL_FRAMES;
    
    // Get pixel coordinates
    vec2 uv = fragCoord / iResolution.xy;
    int px = int(uv.x * float(VIDEO_WIDTH));
    int py = int((1.0 - uv.y) * float(VIDEO_HEIGHT));
    
    // Boundary check
    if (px >= VIDEO_WIDTH || py >= VIDEO_HEIGHT) {{
        fragColor = vec4(0.0);
        return;
    }}
    
    // Normalize inputs to [0, 1]
    float frame_norm = float(frame) / float(TOTAL_FRAMES);
    float x_norm = float(px) / float(VIDEO_WIDTH);
    float y_norm = float(py) / float(VIDEO_HEIGHT);
    
    vec3 input = vec3(frame_norm, x_norm, y_norm);
    
    // Run neural network
    float gray = neuralNetwork(input);
    
    fragColor = vec4(vec3(gray), 1.0);
}}
"""

    return shader_code


def generate_multipass_shader(weights_path: Path, output_dir: Path):
    """Generate complete multi-pass Shadertoy shader."""

    print(f"Loading weights from: {weights_path}")
    weights_dict = load_weights(weights_path)

    # Load metadata
    metadata_path = weights_path.with_name(weights_path.stem + "_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metadata not found at {metadata_path}")
        metadata = {}

    print("\nWeight layers:")
    total_params = 0
    for key, param in weights_dict.items():
        size = param.size
        total_params += size
        print(f"  {key}: {param.shape} = {size:,} values")

    print(f"\nTotal parameters: {total_params:,}")
    size_kb = total_params * 4 / 1024
    print(f"Size: {size_kb:.2f} KB")

    # Generate shaders
    print("\nGenerating Buffer A (weight storage)...")
    buffer_a, offsets, tex_size = generate_buffer_a(weights_dict, metadata)

    print("Generating Image shader (NN inference)...")
    image_shader = generate_image_shader(metadata, offsets, tex_size)

    # Save shaders to files
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_a_path = output_dir / "shadertoy_buffer_a.fs"
    with open(buffer_a_path, "w") as f:
        f.write(buffer_a)
    print(f"\nSaved Buffer A: {buffer_a_path}")

    image_path = output_dir / "shadertoy_image.fs"
    with open(image_path, "w") as f:
        f.write(image_shader)
    print(f"Saved Image: {image_path}")

    print(f"\n{'=' * 80}")
    print("✅ Multi-pass shader generation complete!")
    print(f"{'=' * 80}")
    print(f"\nCode size: {len(buffer_a) + len(image_shader):,} characters")
    print(f"Buffer A: {len(buffer_a):,} chars")
    print(f"Image: {len(image_shader):,} chars")

    if len(buffer_a) > 65000 or len(image_shader) > 65000:
        print("\n⚠️  WARNING: Code may exceed Shadertoy limits!")
        print("Consider using an even smaller architecture.")

    return buffer_a_path, image_path


def main(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default="nn_weights_tiny.npz",
        help="Path to the input NPZ file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Path to the output directory.",
    )
    args = parser.parse_args(argv)
    generate_multipass_shader(
        args.input, output_dir=args.output_dir or args.input.parent
    )
