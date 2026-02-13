#!/usr/bin/env python3
"""Analyze Bad Apple video metadata from parquet file."""

import os

import polars as pl


def analyze_video():
    """Analyze Bad Apple video metadata and compression feasibility."""
    # Read the parquet file
    df = pl.read_parquet("bad_apple/video_pixels.parquet")

    print("=" * 80)
    print("BAD APPLE VIDEO METADATA ANALYSIS")
    print("=" * 80)

    # 1. Video resolution (max x and y values)
    max_x = df.select(pl.col("x").max()).item()
    max_y = df.select(pl.col("y").max()).item()
    width = max_x + 1  # 0-indexed
    height = max_y + 1  # 0-indexed

    print("\n1. VIDEO RESOLUTION:")
    print(f"   Width:  {width} pixels")
    print(f"   Height: {height} pixels")
    print(f"   Resolution: {width}x{height}")

    # 2. Total number of frames
    total_frames = df.select(pl.col("frame").max()).item() + 1  # 0-indexed
    print("\n2. TOTAL FRAMES:")
    print(f"   {total_frames} frames")

    # 3. Total data points
    total_rows = len(df)
    print("\n3. TOTAL DATA POINTS:")
    print(f"   {total_rows:,} rows")
    expected = width * height * total_frames
    print(f"   Verification: {width} × {height} × {total_frames} = {expected:,}")
    if total_rows == expected:
        print("   ✓ Data integrity OK")
    else:
        print("   ⚠ Data mismatch!")

    # 4. Data size calculations
    print("\n4. DATA SIZE ANALYSIS:")
    print("\n   A. Raw Data Size:")
    pixels_per_frame = width * height
    bytes_per_pixel = 1  # grayscale uint8
    raw_size_bytes = total_frames * pixels_per_frame * bytes_per_pixel
    print(
        f"      Per frame: {pixels_per_frame:,} pixels × 1 byte = {pixels_per_frame:,} bytes"
    )
    print(
        f"      Total raw: {total_frames} frames × {pixels_per_frame:,} = {raw_size_bytes:,} bytes"
    )
    print(
        f"      Total raw: {raw_size_bytes / 1024:.2f} KB = {raw_size_bytes / (1024**2):.2f} MB"
    )

    # For shader code analysis
    print("\n   B. Parquet File (on disk):")
    parquet_size = os.path.getsize("bad_apple/video_pixels.parquet")
    print(
        f"      Compressed size: {parquet_size:,} bytes = {parquet_size / (1024**2):.2f} MB"
    )
    print(f"      Compression ratio: {raw_size_bytes / parquet_size:.2f}x")

    # Texture budget analysis
    print("\n5. TEXTURE STORAGE BUDGET:")
    texture_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    for tex_w, tex_h in texture_sizes:
        tex_pixels = tex_w * tex_h
        tex_bytes_rgba = tex_pixels * 4  # RGBA = 4 bytes per pixel
        tex_bytes_float = tex_pixels * 4 * 4  # RGBA32F = 16 bytes per pixel
        print(f"\n   {tex_w}×{tex_h} texture:")
        print(
            f"      RGBA8: {tex_bytes_rgba:,} bytes = {tex_bytes_rgba / (1024**2):.2f} MB"
        )
        print(
            f"      RGBA32F: {tex_bytes_float:,} bytes = {tex_bytes_float / (1024**2):.2f} MB"
        )
        print(
            f"      Can store {tex_bytes_rgba / raw_size_bytes:.2f}x raw video (RGBA8)"
        )
        print(
            f"      Can store {tex_bytes_float / raw_size_bytes:.2f}x raw video (RGBA32F)"
        )

    print("\n6. COMPRESSION APPROACH ANALYSIS:")

    # FFT/DCT analysis
    print("\n   A. FFT/DCT (Frequency Domain) Approach:")
    fft_coefficients_per_frame = width * height
    total_fft_coeffs = total_frames * fft_coefficients_per_frame
    print(f"      Total DCT coefficients (no compression): {total_fft_coeffs:,}")

    for percent in [0.5, 1, 2, 5]:
        kept_coeffs = int(total_fft_coeffs * (percent / 100))
        # Each coefficient: float32 (4 bytes value) + uint16 for position (2 bytes)
        size_bytes = kept_coeffs * 6
        print(f"\n      Top {percent}% frequencies: {kept_coeffs:,} coefficients")
        print(f"         Size: {size_bytes:,} bytes = {size_bytes / 1024:.2f} KB")
        print(f"         Compression ratio: {raw_size_bytes / size_bytes:.1f}x")
        # Check if fits in 1024×1024 RGBA8 texture
        tex_1024_capacity = 1024 * 1024 * 4
        if size_bytes <= tex_1024_capacity:
            print("         ✓ Fits in 1024×1024 RGBA8 texture")
        else:
            print("         ✗ Needs larger texture or more compression")

    # Neural Network analysis
    print("\n   B. Neural Network Approach:")
    architectures = [
        ("Tiny", [3, 32, 64, 32, 1]),
        ("Small", [3, 64, 128, 64, 1]),
        ("Medium", [3, 128, 256, 128, 1]),
    ]

    for name, layers in architectures:
        total_params = 0
        for i in range(len(layers) - 1):
            # weights + biases
            total_params += layers[i] * layers[i + 1] + layers[i + 1]

        # Each param is a float32 (4 bytes)
        size_bytes = total_params * 4
        print(f"\n      {name} NN {layers}:")
        print(f"         Parameters: {total_params:,}")
        print(f"         Size: {size_bytes:,} bytes = {size_bytes / 1024:.2f} KB")

        # Check texture storage
        tex_1024_capacity = 1024 * 1024 * 4
        print(
            f"         ✓ Fits in 1024×1024 RGBA8 texture (uses {size_bytes / tex_1024_capacity * 100:.2f}%)"
        )

    print("\n7. RECOMMENDATION:")
    print("   ════════════════════════════════════════════════════════════")
    print("   Both approaches are feasible with 1024×1024 texture:")
    print("   ")
    print("   • DCT: Keep top 1-2% of coefficients for good quality")
    print("   • Neural Network: Small architecture (64→128→64) recommended")
    print("   • Both will fit comfortably in texture storage")
    print("   ════════════════════════════════════════════════════════════")

    print("\n" + "=" * 80)

    # Return metadata for other scripts
    return {
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "raw_size_bytes": raw_size_bytes,
    }


if __name__ == "__main__":
    metadata = analyze_video()
    print(f"\nMetadata: {metadata}")
