from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from cv2 import (
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_RGB2GRAY,
    VideoCapture,
    cvtColor,
)
from tqdm import tqdm


def extract_pixels_from_capture(capture: VideoCapture) -> pl.DataFrame:
    """Extract all pixels from video using vectorized operations."""
    frames_count = int(capture.get(CAP_PROP_FRAME_COUNT))
    height = int(capture.get(CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(CAP_PROP_FRAME_WIDTH))

    # Pre-allocate arrays for all data
    pixels_per_frame = height * width
    total_pixels = frames_count * pixels_per_frame

    frame_indices = np.empty(total_pixels, dtype=np.uint32)
    x_coords = np.empty(total_pixels, dtype=np.uint16)
    y_coords = np.empty(total_pixels, dtype=np.uint16)
    pixel_values = np.empty(total_pixels, dtype=np.uint8)

    # Pre-compute coordinate grids (reused for each frame)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x_flat = x_grid.ravel().astype(np.uint16)
    y_flat = y_grid.ravel().astype(np.uint16)

    # Process frames
    for i in tqdm(range(frames_count), desc="Processing frames"):
        ret, frame = capture.read()
        if not ret:
            # Truncate arrays if video ends early
            total_pixels = i * pixels_per_frame
            frame_indices = frame_indices[:total_pixels]
            x_coords = x_coords[:total_pixels]
            y_coords = y_coords[:total_pixels]
            pixel_values = pixel_values[:total_pixels]
            break

        # Convert to grayscale and flatten
        gray_frame = cvtColor(frame, COLOR_RGB2GRAY)

        # Calculate offset for this frame
        offset = i * pixels_per_frame
        end = offset + pixels_per_frame

        # Fill arrays using vectorized operations
        frame_indices[offset:end] = i
        x_coords[offset:end] = x_flat
        y_coords[offset:end] = y_flat
        pixel_values[offset:end] = gray_frame.ravel()

    # Create DataFrame from numpy arrays (much faster)
    return pl.DataFrame(
        {
            "frame": frame_indices,
            "x": x_coords,
            "y": y_coords,
            "pixel_value": pixel_values,
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, help="output file")
    parser.add_argument("input", type=Path, help="input file")

    args = parser.parse_args(argv)
    input_file = args.input
    output_file = args.output
    if output_file is None:
        output_file = input_file.parent / (input_file.stem + "_pixels.parquet")

    capture = VideoCapture(str(input_file))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_file}")

    df = extract_pixels_from_capture(capture)
    df.write_parquet(output_file)
