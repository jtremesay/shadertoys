#!/usr/bin/env python3
from argparse import ArgumentParser
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_RGB2GRAY, VideoCapture, cvtColor
from cv2.typing import MatLike
from tqdm import tqdm


@dataclass
class PixelData:
    frame: int
    x: int
    y: int
    pixel_value: int


def yield_pixels_from_frame(frame: MatLike, i) -> Generator[PixelData]:
    height, width = frame.shape
    for y in range(height):
        for x in range(width):
            yield PixelData(i, x, y, frame[y, x])


def yield_pixels_from_capture(
    capture: VideoCapture,
) -> Generator[PixelData]:
    frames_count = int(capture.get(CAP_PROP_FRAME_COUNT))  # Get total number of frames
    for i in tqdm(range(frames_count), desc="Processing frames"):
        ret, frame = capture.read()
        if not ret:
            break

        frame = cvtColor(frame, COLOR_RGB2GRAY)  # Convert to grayscale
        yield from yield_pixels_from_frame(frame, i)


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

    df = pl.DataFrame(yield_pixels_from_capture(capture))
    df.write_parquet(output_file)


if __name__ == "__main__":
    main()
