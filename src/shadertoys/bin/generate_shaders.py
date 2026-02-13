from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from shadertoys.shader import generate_multipass_shader


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
