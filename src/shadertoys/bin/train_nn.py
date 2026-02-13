#!/usr/bin/env python3
"""Train a tiny neural network to compress Bad Apple video."""

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
            "sample_rate": 0.05,  # 5% of data for better quality
            "epochs": 30,  # More training for better convergence
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

        # Train
        model = train_model(
            model, dataloader, epochs=config["epochs"], lr=0.001, device=device
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
