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
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from shadertoys.shader import reconstruct_model


def render_video(
    model: torch.nn.Module,
    output_path: Path,
    width: int,
    height: int,
    total_frames: int,
    fps: int = 30,
    device: str = "cpu",
) -> None:
    """Render neural network back to video using batch processing.

    Args:
        model: Trained TinyVideoNet model
        output_path: Path to output video file
        width: Output video width
        height: Output video height
        total_frames: Number of frames to render
        fps: Frames per second for output video
        device: Device to run inference on ('cpu' or 'cuda')
    """
    model = model.to(device)
    model.eval()

    print("\nRendering video:")
    print(f"  Resolution: {width}×{height}")
    print(f"  Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Device: {device}")
    print(f"  Output: {output_path}")

    # Determine codec based on file extension
    suffix = output_path.suffix.lower()
    if suffix == ".webm":
        # Try VP9 first, fallback to other codecs if not available
        fourcc_options = [
            ("VP90", "VP9"),
            ("VP80", "VP8"),
        ]
    elif suffix == ".mp4":
        fourcc_options = [
            ("mp4v", "MPEG-4"),
            ("avc1", "H.264"),
        ]
    elif suffix == ".avi":
        fourcc_options = [
            ("MJPG", "Motion JPEG"),
            ("XVID", "Xvid"),
        ]
    else:
        # Default fallback
        fourcc_options = [("MJPG", "Motion JPEG")]

    # Try each codec until one works
    out = None
    used_codec = None
    for fourcc_str, codec_name in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
            isColor=False,  # Grayscale output
        )
        if out.isOpened():
            used_codec = codec_name
            print(f"  Codec: {codec_name}")
            break
        out.release()

    if out is None or not out.isOpened():
        raise RuntimeError(
            f"Failed to open video writer for {output_path}. "
            f"Try a different output format (e.g., .mp4 or .avi)"
        )

    # Create coordinate grid for batch processing
    # Generate all (x, y) coordinates for a single frame
    y_coords, x_coords = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    x_coords = x_coords.flatten()  # Shape: [width * height]
    y_coords = y_coords.flatten()  # Shape: [width * height]

    # Normalize spatial coordinates (same as training)
    x_norm = x_coords / width
    y_norm = y_coords / height

    # Render each frame
    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
            # Normalize frame index
            frame_norm = frame_idx / total_frames

            # Create batch input: [num_pixels, 3] where each row is [frame, x, y]
            frame_norms = np.full_like(x_norm, frame_norm)
            inputs = np.stack([frame_norms, x_norm, y_norm], axis=1)

            # Convert to tensor and move to device
            inputs_tensor = torch.from_numpy(inputs).to(device)

            # Batch inference
            outputs = model(inputs_tensor)  # Shape: [num_pixels, 1]

            # Denormalize: [0, 1] -> [0, 255]
            pixels = (outputs.cpu().numpy() * 255).astype(np.uint8)

            # Reshape to frame: [height, width]
            frame = pixels.reshape(height, width)

            # Write frame to video
            out.write(frame)

    # Release video writer
    out.release()

    print(f"\n✅ Video rendered successfully: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")

    # Calculate compression metrics
    original_size = width * height * total_frames  # Bytes (grayscale)
    compressed_size = output_path.stat().st_size
    compression_ratio = original_size / compressed_size
    print(f"   Compression: {compression_ratio:.2f}x")


def main(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser(description="Render trained neural network to video.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input .npz weights file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output video file (default: <input_stem>_rendered.mp4)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output video width (default: use metadata or training dimensions)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output video height (default: use metadata or training dimensions)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate (default: 30)",
    )
    args = parser.parse_args(argv)

    # Validate input file
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Set default output path
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_rendered.mp4"

    # Load model and metadata
    print(f"Loading model from: {args.input}")
    model, metadata = reconstruct_model(args.input)
    print(f"  Architecture: {metadata.get('architecture', 'Unknown')}")
    print(f"  Hidden sizes: {metadata.get('hidden_sizes', 'Unknown')}")
    print(f"  Parameters: {metadata.get('total_parameters', 'Unknown'):,}")

    # Determine video dimensions
    # Priority: CLI args > metadata > error
    width = args.width
    height = args.height
    total_frames = metadata.get("total_frames")

    if width is None:
        width = metadata.get("video_width")
        if width is None:
            raise ValueError(
                "Video width not found in metadata. Please specify --width"
            )

    if height is None:
        height = metadata.get("video_height")
        if height is None:
            raise ValueError(
                "Video height not found in metadata. Please specify --height"
            )

    if total_frames is None:
        raise ValueError(
            "Total frames not found in metadata. "
            "Retrain the model with the updated training script."
        )

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("  Using CPU (GPU not available)")

    # Render video
    render_video(
        model=model,
        output_path=args.output,
        width=width,
        height=height,
        total_frames=total_frames,
        fps=args.fps,
        device=device,
    )


if __name__ == "__main__":
    main()
