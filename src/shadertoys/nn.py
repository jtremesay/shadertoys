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
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class VideoDataset(Dataset):
    """Dataset for video pixels."""

    def __init__(
        self,
        df: pl.DataFrame,
        width: int,
        height: int,
        total_frames: int,
        sample_rate: float = 1.0,
    ):
        """
        Args:
            df: Polars DataFrame with pixel data
            width, height: Video dimensions
            total_frames: Number of frames
            sample_rate: Fraction of pixels to use for training (1.0 = all pixels)
        """
        self.width = width
        self.height = height
        self.total_frames = total_frames

        # Sample data if needed
        if sample_rate < 1.0:
            n_samples = int(len(df) * sample_rate)
            df = df.sample(n=n_samples, shuffle=True, seed=42)

        # Convert to numpy for faster access
        self.frames = df.select("frame").to_numpy().flatten().astype(np.float32)
        self.xs = df.select("x").to_numpy().flatten().astype(np.float32)
        self.ys = df.select("y").to_numpy().flatten().astype(np.float32)
        self.pixels = df.select("pixel_value").to_numpy().flatten().astype(np.float32)

        # Normalize inputs to [0, 1]
        self.frames /= total_frames
        self.xs /= width
        self.ys /= height

        # Normalize outputs to [0, 1]
        self.pixels /= 255.0

        print(
            f"Dataset: {len(self.frames):,} pixels ({sample_rate * 100:.1f}% of total)"
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Input: (frame_normalized, x_normalized, y_normalized)
        x = torch.tensor(
            [self.frames[idx], self.xs[idx], self.ys[idx]], dtype=torch.float32
        )
        # Output: pixel_value_normalized
        y = torch.tensor([self.pixels[idx]], dtype=torch.float32)
        return x, y


class TinyVideoNet(nn.Module):
    """Tiny neural network for video compression."""

    def __init__(self, hidden_sizes: list[int] = [32, 64, 32]):
        """
        Args:
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()

        layers = []
        input_size = 3  # (frame, x, y)

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.network = nn.Sequential(*layers)
        self.hidden_sizes = hidden_sizes

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    output_path: Path,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = "cpu",
    start_epoch: int = 0,
    loss_history: list[float] | None = None,
    width: int | None = None,
    height: int | None = None,
    total_frames: int | None = None,
) -> tuple[nn.Module, list[float]]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_history is None:
        loss_history = []

    print(f"\nTraining on {device}...")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Epochs: {start_epoch + 1} → {epochs}, Learning rate: {lr}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Training epochs", leave=False):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        save_checkpoint(
            model,
            optimizer,
            epoch,
            loss_history,
            output_path.with_name(f"{output_path.stem}_checkpoint.pth"),
        )
        save_model(
            model,
            output_path,
            width=width,
            height=height,
            total_frames=total_frames,
        )
    return model, loss_history


def save_model(
    model: nn.Module,
    output_path: Path,
    width: int | None = None,
    height: int | None = None,
    total_frames: int | None = None,
):
    """Save model weights and metadata."""
    save_model_weights(model, output_path)
    save_metadata(
        model,
        output_path.with_name(f"{output_path.stem}_metadata.json"),
        width=width,
        height=height,
        total_frames=total_frames,
    )


def save_model_weights(model: nn.Module, output_path: Path):
    """Save model weights as JSON for easy shader integration."""
    weights_dict = {}

    layer_idx = 0
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
        layer_idx += 1

    with open(output_path, "w") as f:
        json.dump(weights_dict, f)

    # Also save as numpy for texture encoding
    numpy_path = output_path.with_suffix(".npz")
    np.savez(numpy_path, **{k: np.array(v) for k, v in weights_dict.items()})

    return weights_dict


def save_metadata(
    model: nn.Module,
    metadata_path: Path,
    width: int | None = None,
    height: int | None = None,
    total_frames: int | None = None,
):
    """Save model metadata as JSON."""
    metadata = {
        "architecture": "TinyVideoNet",
        "hidden_sizes": model.hidden_sizes,
        "total_parameters": model.count_parameters(),
    }

    # Add video dimensions if provided
    if width is not None:
        metadata["video_width"] = width
    if height is not None:
        metadata["video_height"] = height
    if total_frames is not None:
        metadata["total_frames"] = total_frames

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss_history: list[float],
    checkpoint_path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> tuple[int, list[float]]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss_history = checkpoint["loss_history"]
    return epoch + 1, loss_history  # +1 to start at next epoch


def evaluate_model(
    model: nn.Module,
    df: pl.DataFrame,
    width: int,
    height: int,
    total_frames: int,
    device: str = "cpu",
    num_samples: int = 10000,
) -> None:
    """Evaluate model on random samples."""
    model.eval()
    model = model.to(device)

    # Sample random pixels
    sample_df = df.sample(n=min(num_samples, len(df)), shuffle=True, seed=42)

    frames = (
        sample_df.select("frame").to_numpy().flatten().astype(np.float32) / total_frames
    )
    xs = sample_df.select("x").to_numpy().flatten().astype(np.float32) / width
    ys = sample_df.select("y").to_numpy().flatten().astype(np.float32) / height
    pixels = (
        sample_df.select("pixel_value").to_numpy().flatten().astype(np.float32) / 255.0
    )

    inputs = torch.tensor(np.stack([frames, xs, ys], axis=1), dtype=torch.float32).to(
        device
    )
    targets = torch.tensor(pixels, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions = model(inputs)
        mse = nn.MSELoss()(predictions, targets).item()
        mae = nn.L1Loss()(predictions, targets).item()

    # Calculate PSNR (assuming values in [0, 1])
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    print(f"\nEvaluation on {num_samples:,} samples:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Avg pixel error: {mae * 255:.2f} / 255")
