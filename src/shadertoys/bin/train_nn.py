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
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from torch.utils.data import DataLoader

from shadertoys.nn import (
    TinyVideoNet,
    VideoDataset,
    evaluate_model,
    load_checkpoint,
    save_model_weights,
    train_model,
)


def main(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser(
        description="Train a tiny neural network to compress video."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("video_pixels.parquet"),
        help="Path to input parquet file with video pixel data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("nn_weights.json"),
        help="Path to output JSON file for model weights",
    )
    parser.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="Force restart training from scratch, ignoring existing checkpoints",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    args = parser.parse_args(argv)

    """Main training pipeline."""
    # Load video data
    print("Loading video data...")
    df = pl.read_parquet(args.input)

    # Get video dimensions
    width = df.select(pl.col("x").max()).item() + 1
    height = df.select(pl.col("y").max()).item() + 1
    total_frames = df.select(pl.col("frame").max()).item() + 1

    print(f"Video: {width}×{height}, {total_frames} frames")
    print(f"Total pixels: {len(df):,}")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Focus on Tiny architecture for Shadertoy (no custom textures)
    # Optimized for best quality within code size constraints
    architectures = [
        {
            "name": "Tiny",
            "hidden": [32, 64, 32],
            "sample_rate": 1,  # 5% of data for better quality
            "batch_size": 8192,
        },
    ]

    for config in architectures:
        print(f"\n{'=' * 80}")
        print(f"Training {config['name']} Network: {config['hidden']}")
        print(f"{'=' * 80}")

        # Create dataset (sample for faster training)
        dataset = VideoDataset(
            df, width, height, total_frames, sample_rate=config["sample_rate"]
        )
        dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
        )

        # Create model
        model = TinyVideoNet(hidden_sizes=config["hidden"])

        # Check for existing checkpoint
        checkpoint_path = (
            args.output.parent
            / f"{args.output.stem}_{config['name'].lower()}_checkpoint.pth"
        )
        start_epoch = 0
        loss_history = []

        if checkpoint_path.exists() and not args.restart:
            print(f"📥 Found checkpoint: {checkpoint_path.name}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            start_epoch, loss_history = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(f"🔄 Resuming from epoch {start_epoch + 1}")
        else:
            if args.restart and checkpoint_path.exists():
                print("🔥 Restarting training from scratch (--restart flag)")
            else:
                print("✨ Starting new training")

        # Train
        model, loss_history = train_model(
            model,
            dataloader,
            epochs=args.epochs,
            lr=0.001,
            device=device,
            checkpoint_path=checkpoint_path,
            start_epoch=start_epoch,
            loss_history=loss_history,
        )

        # Evaluate
        evaluate_model(model, df, width, height, total_frames, device=device)

        # Save weights
        output_path = args.output.with_name(
            f"{args.output.stem}_{config['name'].lower()}{args.output.suffix}"
        )
        weights = save_model_weights(model, output_path)

        # Save model metadata
        metadata = {
            "architecture": config["name"],
            "hidden_sizes": config["hidden"],
            "total_parameters": model.count_parameters(),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }
        metadata_path = output_path.with_name(
            f"{args.output.stem}_{config['name'].lower()}_metadata{args.output.suffix}"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
