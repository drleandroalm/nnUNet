#!/usr/bin/env python3
"""
Generate per-stage preprocessing fixtures for Swift validation.

Usage:
    python Scripts/generate_fixtures.py \
        --input-nifti /path/to/test_volume.nii.gz \
        --plans-json /path/to/nnUNetPlans.json \
        --dataset-fingerprint /path/to/dataset_fingerprint.json \
        --configuration 3d_fullres \
        --output-dir Tests/nnUNetPreprocessingTests/Fixtures
"""

import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from typing import Tuple, Dict, Any, Optional

# Import nnUNet preprocessing components
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape


def compute_checksum(array: np.ndarray) -> str:
    """Compute MD5 checksum of numpy array."""
    return hashlib.md5(array.tobytes()).hexdigest()


def generate_fixtures(
    input_nifti: Path,
    plans_path: Path,
    fingerprint_path: Path,
    configuration: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate per-stage fixtures for Swift validation."""

    # Load plans and fingerprint
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    with open(fingerprint_path, 'r') as f:
        fingerprint = json.load(f)

    config = plans["configurations"][configuration]

    # Load input volume
    nifti = nib.load(str(input_nifti))
    raw_data = nifti.get_fdata().astype(np.float32)
    spacing = np.array(nifti.header.get_zooms()[:3])

    print(f"Input shape: {raw_data.shape}")
    print(f"Input spacing: {spacing}")

    stages = {}
    metadata = {
        "input_file": str(input_nifti.name),
        "configuration": configuration,
        "stages": {}
    }

    # Stage 1: Raw (after loading, before any processing)
    # Add channel dimension as nnUNet expects (C, Z, Y, X) but we store without channel for simplicity
    current = raw_data.copy()
    stages["01_raw"] = current
    metadata["stages"]["01_raw"] = {
        "shape": list(current.shape),
        "spacing": spacing.tolist(),
        "dtype": str(current.dtype)
    }

    # Stage 2: Transpose
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    current = np.transpose(current, transpose_forward)
    transposed_spacing = spacing[transpose_forward]
    stages["02_transposed"] = current
    metadata["stages"]["02_transposed"] = {
        "shape": list(current.shape),
        "spacing": transposed_spacing.tolist(),
        "transpose_axes": transpose_forward
    }

    # Stage 3: Crop to nonzero
    # nnUNet uses data != 0 for nonzero mask
    nonzero_mask = current != 0
    if nonzero_mask.any():
        bbox = []
        for axis in range(current.ndim):
            axis_mask = nonzero_mask.any(axis=tuple(i for i in range(current.ndim) if i != axis))
            indices = np.where(axis_mask)[0]
            if len(indices) > 0:
                bbox.append((int(indices[0]), int(indices[-1]) + 1))
            else:
                bbox.append((0, current.shape[axis]))

        slices = tuple(slice(b[0], b[1]) for b in bbox)
        current = current[slices].copy()
        bbox_list = bbox
    else:
        bbox_list = [(0, s) for s in current.shape]

    stages["03_cropped"] = current
    metadata["stages"]["03_cropped"] = {
        "shape": list(current.shape),
        "bbox": bbox_list
    }

    # Stage 4: CT Normalization
    if "CTNormalization" in config["normalization_schemes"]:
        props = fingerprint["foreground_intensity_properties_per_channel"]["0"]
        mean = props["mean"]
        std = props["std"]
        lower = props["percentile_00_5"]
        upper = props["percentile_99_5"]

        # Clip and normalize
        current = np.clip(current, lower, upper)
        current = (current - mean) / max(std, 1e-8)

    stages["04_normalized"] = current
    metadata["stages"]["04_normalized"] = {
        "shape": list(current.shape),
        "spacing": transposed_spacing.tolist(),
        "mean": float(np.mean(current)),
        "std": float(np.std(current)),
        "min": float(np.min(current)),
        "max": float(np.max(current))
    }

    # Stage 5: Resample to target spacing
    target_spacing = np.array(config["spacing"])
    current_spacing = transposed_spacing  # After transpose

    # Compute target shape
    scale_factors = current_spacing / target_spacing
    target_shape = np.round(np.array(current.shape) * scale_factors).astype(int)

    # Add channel dimension for nnUNet resampling function
    current_with_channel = current[np.newaxis, ...]

    # Get resampling kwargs
    resample_kwargs = config.get("resampling_fn_data_kwargs", {
        "is_seg": False,
        "order": 3,
        "order_z": 0,
        "force_separate_z": None
    })

    resampled = resample_data_or_seg_to_shape(
        current_with_channel,
        target_shape,
        current_spacing,
        target_spacing,
        **resample_kwargs
    )

    # Remove channel dimension
    current = resampled[0]

    stages["05_resampled"] = current
    metadata["stages"]["05_resampled"] = {
        "shape": list(current.shape),
        "target_spacing": target_spacing.tolist(),
        "original_spacing": current_spacing.tolist(),
        "resample_kwargs": resample_kwargs
    }

    # Save fixtures
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}

    for name, array in stages.items():
        filepath = output_dir / f"{name}.npy"
        np.save(filepath, array.astype(np.float32))
        checksums[name] = compute_checksum(array.astype(np.float32))
        print(f"Saved {name}: shape={array.shape}, checksum={checksums[name][:8]}...")

    # Save metadata with checksums
    metadata["checksums"] = checksums
    with open(output_dir / "fixture_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFixtures saved to {output_dir}")
    return metadata


def generate_synthetic_fixtures(output_dir: Path, params_path: Path) -> Dict[str, Any]:
    """Generate synthetic fixtures when no NIfTI file is available."""

    # Load preprocessing params
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Create synthetic volume (small for testing)
    np.random.seed(42)  # Reproducibility
    raw_shape = (32, 64, 64)  # D, H, W
    raw_spacing = np.array(params.get("original_spacing", [2.5, 0.7, 0.7]))

    # Create synthetic CT-like data with some structure
    raw_data = np.zeros(raw_shape, dtype=np.float32)

    # Add a sphere in the center to simulate anatomy
    center = np.array(raw_shape) // 2
    for z in range(raw_shape[0]):
        for y in range(raw_shape[1]):
            for x in range(raw_shape[2]):
                dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                if dist < 15:
                    # Soft tissue HU values
                    raw_data[z, y, x] = 50 + np.random.randn() * 20
                elif dist < 20:
                    # Bone-like values
                    raw_data[z, y, x] = 500 + np.random.randn() * 50

    # Add some background air
    raw_data[raw_data == 0] = -1000 + np.random.randn(np.sum(raw_data == 0)) * 10

    print(f"Synthetic input shape: {raw_data.shape}")
    print(f"Synthetic input spacing: {raw_spacing}")

    stages = {}
    metadata = {
        "input_file": "synthetic_volume",
        "configuration": params["configuration_name"],
        "stages": {}
    }

    # Stage 1: Raw
    current = raw_data.copy()
    stages["01_raw"] = current
    metadata["stages"]["01_raw"] = {
        "shape": list(current.shape),
        "spacing": raw_spacing.tolist(),
        "dtype": str(current.dtype)
    }

    # Stage 2: Transpose
    transpose_forward = params.get("transpose_forward", [0, 1, 2])
    current = np.transpose(current, transpose_forward)
    transposed_spacing = raw_spacing[transpose_forward]
    stages["02_transposed"] = current
    metadata["stages"]["02_transposed"] = {
        "shape": list(current.shape),
        "spacing": transposed_spacing.tolist(),
        "transpose_axes": transpose_forward
    }

    # Stage 3: Crop to nonzero
    nonzero_mask = current != 0
    if nonzero_mask.any():
        bbox = []
        for axis in range(current.ndim):
            axis_mask = nonzero_mask.any(axis=tuple(i for i in range(current.ndim) if i != axis))
            indices = np.where(axis_mask)[0]
            if len(indices) > 0:
                bbox.append((int(indices[0]), int(indices[-1]) + 1))
            else:
                bbox.append((0, current.shape[axis]))
        slices = tuple(slice(b[0], b[1]) for b in bbox)
        current = current[slices].copy()
        bbox_list = bbox
    else:
        bbox_list = [(0, s) for s in current.shape]

    stages["03_cropped"] = current
    metadata["stages"]["03_cropped"] = {
        "shape": list(current.shape),
        "bbox": bbox_list
    }

    # Stage 4: CT Normalization
    if "CTNormalization" in params.get("normalization_schemes", []):
        props = params["foreground_intensity_properties"].get("0", {})
        mean = props.get("mean", 100.5)
        std = props.get("std", 50.2)
        lower = props.get("percentile_00_5", -1024.0)
        upper = props.get("percentile_99_5", 1500.0)

        current = np.clip(current, lower, upper)
        current = (current - mean) / max(std, 1e-8)

    stages["04_normalized"] = current
    metadata["stages"]["04_normalized"] = {
        "shape": list(current.shape),
        "spacing": transposed_spacing.tolist(),
        "mean": float(np.mean(current)),
        "std": float(np.std(current)),
        "min": float(np.min(current)),
        "max": float(np.max(current))
    }

    # Stage 5: Resample
    target_spacing = np.array(params.get("target_spacing", [1.0, 0.5, 0.5]))
    current_spacing = transposed_spacing

    scale_factors = current_spacing / target_spacing
    target_shape = np.round(np.array(current.shape) * scale_factors).astype(int)

    current_with_channel = current[np.newaxis, ...]

    resample_kwargs = params.get("resampling_fn_data_kwargs", {
        "is_seg": False,
        "order": 3,
        "order_z": 0,
        "force_separate_z": None
    })

    resampled = resample_data_or_seg_to_shape(
        current_with_channel,
        target_shape,
        current_spacing,
        target_spacing,
        **resample_kwargs
    )

    current = resampled[0]

    stages["05_resampled"] = current
    metadata["stages"]["05_resampled"] = {
        "shape": list(current.shape),
        "target_spacing": target_spacing.tolist(),
        "original_spacing": current_spacing.tolist(),
        "resample_kwargs": resample_kwargs
    }

    # Save fixtures
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}

    for name, array in stages.items():
        filepath = output_dir / f"{name}.npy"
        np.save(filepath, array.astype(np.float32))
        checksums[name] = compute_checksum(array.astype(np.float32))
        print(f"Saved {name}: shape={array.shape}, checksum={checksums[name][:8]}...")

    metadata["checksums"] = checksums
    with open(output_dir / "fixture_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSynthetic fixtures saved to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate preprocessing fixtures for Swift validation"
    )
    parser.add_argument(
        "--input-nifti",
        type=Path,
        help="Input NIfTI volume for fixture generation"
    )
    parser.add_argument(
        "--plans-json",
        type=Path,
        help="Path to nnUNetPlans.json"
    )
    parser.add_argument(
        "--dataset-fingerprint",
        type=Path,
        help="Path to dataset_fingerprint.json"
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="Configuration name (default: 3d_fullres)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fixtures"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic fixtures (no NIfTI required)"
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="Path to preprocessing_params.json (for synthetic mode)"
    )

    args = parser.parse_args()

    if args.synthetic:
        if not args.params_json:
            args.params_json = args.output_dir / "preprocessing_params.json"
        generate_synthetic_fixtures(args.output_dir, args.params_json)
    else:
        if not args.input_nifti or not args.plans_json or not args.dataset_fingerprint:
            parser.error("--input-nifti, --plans-json, and --dataset-fingerprint are required unless --synthetic is used")
        generate_fixtures(
            args.input_nifti,
            args.plans_json,
            args.dataset_fingerprint,
            args.configuration,
            args.output_dir
        )


if __name__ == "__main__":
    main()
