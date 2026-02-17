# ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4)
#
# Copyright © 2026 Jonathan Tremesayques
#
# This is anti-capitalist software, released for free use by individuals and
# organizations that do not operate by capitalist principles.
#
# Permission is hereby granted, free of charge, to any person or organization
# (the "User") obtaining a copy of this software and associated documentation
# files (the "Software"), to use, copy, modify, merge, distribute, and/or sell
# copies of the Software, subject to the following conditions:
#
#   1. The above copyright notice and this permission notice shall be included
#      in all copies or modified versions of the Software.
#
#   2. The User is one of the following:
#     a. An individual person, laboring for themselves
#     b. A non-profit organization
#     c. An educational institution
#     d. An organization that seeks shared profit for all of its members, and
#        allows non-members to set the cost of their labor
#
#   3. If the User is an organization with owners, then all owners are workers
#     and all workers are owners with equal equity and/or equal vote.
#
#   4. If the User is an organization, then the User is not law enforcement or
#      military, or working for or under either.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY
# KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import json
from pathlib import Path

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

    tpl = env.get_template("image.fs")
    return tpl.render(
        width=width,
        height=height,
        total_frames=total_frames,
        tex_size=tex_size,
        hidden_sizes=hidden_sizes,
        layer_info=layer_info,
    )


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
