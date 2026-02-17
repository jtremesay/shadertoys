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
from pathlib import Path

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


def extract_pixels_from_capture(video_path: Path) -> pl.DataFrame:
    """Extract all pixels from video using vectorized operations."""
    capture = VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

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
