from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from shadertoys.video import extract_pixels_from_capture


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", default="video.webm", type=Path, help="input file"
    )
    parser.add_argument("-o", "--output", type=Path, help="output file")

    args = parser.parse_args(argv)
    input_file = args.input
    output_file = args.output
    if output_file is None:
        output_file = input_file.parent / (input_file.stem + "_pixels.parquet")

    df = extract_pixels_from_capture(input_file)
    df.write_parquet(output_file)
