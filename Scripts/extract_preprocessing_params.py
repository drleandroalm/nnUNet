#!/usr/bin/env python3
"""
Extract preprocessing parameters from trained nnUNet model.

Usage:
    python Scripts/extract_preprocessing_params.py \
        --plans-json /path/to/nnUNetPlans.json \
        --dataset-fingerprint /path/to/dataset_fingerprint.json \
        --configuration 3d_fullres \
        --output Tests/nnUNetPreprocessingTests/Fixtures/preprocessing_params.json
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def extract_preprocessing_params(
    plans_path: Path,
    fingerprint_path: Path,
    configuration: str
) -> dict:
    """Extract all preprocessing parameters needed for iOS implementation."""

    with open(plans_path, 'r') as f:
        plans = json.load(f)

    with open(fingerprint_path, 'r') as f:
        fingerprint = json.load(f)

    if configuration not in plans["configurations"]:
        available = list(plans["configurations"].keys())
        raise ValueError(f"Configuration '{configuration}' not found. Available: {available}")

    config = plans["configurations"][configuration]

    # Extract parameters (including resampling details v1 missed)
    params = {
        # Basic configuration
        "configuration_name": configuration,
        "target_spacing": config["spacing"],
        "patch_size": config["patch_size"],

        # Transpose axes
        "transpose_forward": plans.get("transpose_forward", [0, 1, 2]),
        "transpose_backward": plans.get("transpose_backward", [0, 1, 2]),

        # Normalization
        "normalization_schemes": config["normalization_schemes"],
        "use_mask_for_norm": config.get("use_mask_for_norm", [False]),

        # Foreground intensity properties (for CT normalization)
        "foreground_intensity_properties": fingerprint.get(
            "foreground_intensity_properties_per_channel", {}
        ),

        # NEW: Full resampling specification (v1 missed these)
        "resampling_fn_data": config.get("resampling_fn_data", "resample_data_or_seg_to_shape"),
        "resampling_fn_data_kwargs": config.get("resampling_fn_data_kwargs", {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None
        }),
        "resampling_fn_seg": config.get("resampling_fn_seg", "resample_data_or_seg_to_shape"),
        "resampling_fn_seg_kwargs": config.get("resampling_fn_seg_kwargs", {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None
        }),

        # NEW: Anisotropy handling
        "anisotropy_threshold": 3.0,  # nnUNet default for separate-Z decision

        # Dataset properties
        "original_spacing": fingerprint.get("spacing", []),
        "original_median_shape": fingerprint.get("shapes_after_crop", [[]])[0] if fingerprint.get("shapes_after_crop") else [],
    }

    # Validate CT normalization parameters if CTNormalization is used
    if "CTNormalization" in config["normalization_schemes"]:
        channel_props = params["foreground_intensity_properties"].get("0", {})
        required_keys = ["mean", "std", "percentile_00_5", "percentile_99_5"]
        missing_keys = [k for k in required_keys if k not in channel_props]
        if missing_keys:
            raise ValueError(
                f"Missing CT normalization parameters: {missing_keys}. "
                f"Available keys: {list(channel_props.keys())}"
            )

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Extract preprocessing parameters from nnUNet plans"
    )
    parser.add_argument(
        "--plans-json",
        type=Path,
        required=True,
        help="Path to nnUNetPlans.json"
    )
    parser.add_argument(
        "--dataset-fingerprint",
        type=Path,
        required=True,
        help="Path to dataset_fingerprint.json"
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="Configuration name (default: 3d_fullres)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    params = extract_preprocessing_params(
        args.plans_json,
        args.dataset_fingerprint,
        args.configuration
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Parameters extracted successfully to {args.output}")
    print(f"Configuration: {params['configuration_name']}")
    print(f"Target spacing: {params['target_spacing']}")
    print(f"Patch size: {params['patch_size']}")
    print(f"Resampling order (data): {params['resampling_fn_data_kwargs'].get('order', 3)}")
    print(f"Resampling order (Z): {params['resampling_fn_data_kwargs'].get('order_z', 0)}")


if __name__ == "__main__":
    main()
