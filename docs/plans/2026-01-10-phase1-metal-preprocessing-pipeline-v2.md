# Phase 1: Metal Preprocessing Pipeline Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Revision:** v2 — Addresses all P0/P1 issues from GPT audit report (2026-01-09)

**Goal:** Implement GPU-accelerated preprocessing pipeline for CT DICOM volumes using Metal, matching nnUNet's preprocessing behavior exactly with fixture-validated correctness.

**Key Changes from v1:**
- Integrates DICOM-Decoder for native DICOM parsing (no more architecture gap)
- Implements full nnUNet pipeline order: transpose → crop-to-nonzero → normalize → resample
- CPU-first implementation strategy with Metal port after correctness is proven
- Python fixture generator for per-stage validation
- Fixes all Swift/Metal code bugs identified in GPT audit

---

## Architecture Overview

### Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    nnUNetPreprocessing                       │
│  (New SPM package - nnUNet-specific preprocessing logic)    │
└─────────────────────────┬───────────────────────────────────┘
                          │ depends on
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    DICOM-Decoder    │         │        MTK          │
│  (DICOM parsing,    │         │  (VolumeDataset,    │
│   HU conversion,    │         │   Metal infra,      │
│   IPP sorting)      │         │   coordinates)      │
└─────────────────────┘         └─────────────────────┘
```

### Package Structure

```
nnUNetPreprocessing/
├── Package.swift
├── Sources/
│   └── nnUNetPreprocessing/
│       ├── Models/
│       │   ├── PreprocessingParameters.swift
│       │   ├── VolumeBuffer.swift
│       │   ├── BoundingBox.swift
│       │   └── CTNormalizationProperties.swift
│       ├── Pipeline/
│       │   └── PreprocessingPipeline.swift
│       ├── CPU/
│       │   ├── Transpose.swift
│       │   ├── CropToNonzero.swift
│       │   ├── CTNormalization.swift
│       │   └── Resampling.swift
│       ├── Metal/
│       │   ├── Shaders/
│       │   │   ├── CTNormalization.metal
│       │   │   └── Resampling.metal
│       │   ├── MetalCTNormalizer.swift
│       │   └── MetalResampler.swift
│       └── Bridge/
│           └── DicomBridge.swift
├── Tests/
│   ├── Fixtures/
│   │   ├── preprocessing_params.json
│   │   ├── 01_raw.npy
│   │   ├── 02_transposed.npy
│   │   ├── 03_cropped.npy
│   │   ├── 04_normalized.npy
│   │   ├── 05_resampled.npy
│   │   └── checksums.json
│   └── nnUNetPreprocessingTests/
│       ├── TransposeTests.swift
│       ├── CropToNonzeroTests.swift
│       ├── CTNormalizationTests.swift
│       ├── ResamplingTests.swift
│       ├── FixtureValidationTests.swift
│       └── MetalValidationTests.swift
└── Scripts/
    ├── extract_preprocessing_params.py
    └── generate_fixtures.py
```

### Tech Stack

- **Swift 6.2+**
- **iOS 26+ / macOS 15+**
- **Metal 4**
- **Python 3.10+** (for fixture generation)

---

## Data Contract

### Input: DICOM Series (from DICOM-Decoder)

```swift
// DicomSeriesVolume from DICOM-Decoder
struct DicomSeriesVolume {
    let voxels: Data              // Contiguous Int16 buffer, Z-Y-X order
    let width: Int
    let height: Int
    let depth: Int
    let spacing: SIMD3<Double>    // (x, y, z) in mm
    let orientation: simd_double3x3
    let origin: SIMD3<Double>     // IPP of first slice (mm)
    let rescaleSlope: Double
    let rescaleIntercept: Double
    let isSignedPixel: Bool
}
```

### Internal: VolumeBuffer

```swift
/// Internal volume representation for preprocessing pipeline
public struct VolumeBuffer: Sendable {
    /// Raw voxel data (Float32 after HU conversion)
    public var data: Data

    /// Shape in nnUNet convention: (depth, height, width)
    public var shape: (depth: Int, height: Int, width: Int)

    /// Physical spacing in mm: (z, y, x)
    public var spacing: SIMD3<Double>

    /// World coordinates origin (mm)
    public var origin: SIMD3<Double>

    /// Direction cosines from DICOM IOP
    public var orientation: simd_double3x3

    /// Bounding box after crop-to-nonzero (nil if not cropped)
    public var bbox: BoundingBox?

    /// Number of voxels
    public var voxelCount: Int {
        shape.depth * shape.height * shape.width
    }
}
```

### Pipeline Stages & Transformations

```
DicomSeriesVolume (Int16, W×H×D, spacing x,y,z)
    │
    ▼ [1. HU Conversion] — Apply rescaleSlope/Intercept → Float32
    │
    ▼ [2. Transpose] — Reorder axes per plans.transpose_forward
    │
    ▼ [3. Crop to Nonzero] — Remove background, store bbox for inverse
    │
    ▼ [4. Normalize] — CT: clip to percentiles, z-score normalize
    │
    ▼ [5. Resample] — Cubic interpolation to target spacing
    │
VolumeBuffer (Float32, D×H×W, target spacing)
    │
    ▼ [Ready for Core ML inference]
```

### Output: Ready for Core ML

The final `VolumeBuffer` can be converted to:
- `MTLBuffer` for Metal compute shaders
- `MLMultiArray` for Core ML inference

---

## Prerequisites

**Required Skills:**
- @apple-senior-developer — For iOS 26 / Swift 6.2 patterns
- @superpowers:test-driven-development — For test-driven development

**Required Tools:**
- Xcode 26+ (for iOS 26 support)
- Python 3.10+ with nnUNet installed
- Access to trained nnUNet model with preprocessing parameters

**Required Libraries (local paths):**
- `/Users/leandroalmeida/DICOM-Decoder` — DICOM parsing
- `/Users/leandroalmeida/MTK` — Metal volume infrastructure

**Setup Before Starting:**
```bash
# Navigate to nnUNet repo
cd /Users/leandroalmeida/nnUNet

# Verify Python environment has nnUNet
python3 -c "import nnunetv2; print('nnUNet installed')"

# Verify Xcode
xcodebuild -version

# Verify DICOM-Decoder exists
ls /Users/leandroalmeida/DICOM-Decoder/Package.swift

# Verify MTK exists
ls /Users/leandroalmeida/MTK/Package.swift
```

---

## Task 1: Project Setup

**Goal:** Create SPM package with dependencies and directory structure.

**Files:**
- Create: `Package.swift`
- Create: Directory structure

### Step 1: Create directory structure

```bash
cd /Users/leandroalmeida/nnUNet
mkdir -p Sources/nnUNetPreprocessing/Models
mkdir -p Sources/nnUNetPreprocessing/Pipeline
mkdir -p Sources/nnUNetPreprocessing/CPU
mkdir -p Sources/nnUNetPreprocessing/Metal/Shaders
mkdir -p Sources/nnUNetPreprocessing/Bridge
mkdir -p Tests/nnUNetPreprocessingTests/Fixtures
mkdir -p Scripts
```

### Step 2: Create Package.swift

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "nnUNetPreprocessing",
    platforms: [
        .iOS(.v26),
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "nnUNetPreprocessing",
            targets: ["nnUNetPreprocessing"]
        )
    ],
    dependencies: [
        .package(path: "../DICOM-Decoder"),
        .package(path: "../MTK"),
    ],
    targets: [
        .target(
            name: "nnUNetPreprocessing",
            dependencies: [
                .product(name: "DicomCore", package: "DICOM-Decoder"),
                .product(name: "MTKCore", package: "MTK"),
            ],
            resources: [
                .process("Metal/Shaders")
            ]
        ),
        .testTarget(
            name: "nnUNetPreprocessingTests",
            dependencies: ["nnUNetPreprocessing"],
            resources: [
                .process("Fixtures")
            ]
        )
    ]
)
```

### Step 3: Verify package resolves

```bash
swift package resolve
```

Expected: Dependencies resolve successfully

### Step 4: Commit

```bash
git add Package.swift Sources/ Tests/ Scripts/
git commit -m "feat: create nnUNetPreprocessing package structure"
```

---

## Task 2: Extract Full Preprocessing Parameters

**Goal:** Extract all preprocessing parameters including resampling function details that the v1 plan missed.

**Files:**
- Create: `Scripts/extract_preprocessing_params.py`
- Create: `Tests/nnUNetPreprocessingTests/Fixtures/preprocessing_params.json`

### Step 1: Create parameter extraction script

```python
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
```

### Step 2: Make script executable

```bash
chmod +x Scripts/extract_preprocessing_params.py
```

### Step 3: Test with your trained model

```bash
python3 Scripts/extract_preprocessing_params.py \
    --plans-json /path/to/nnUNet_preprocessed/DatasetXXX/nnUNetPlans.json \
    --dataset-fingerprint /path/to/nnUNet_preprocessed/DatasetXXX/dataset_fingerprint.json \
    --configuration 3d_fullres \
    --output Tests/nnUNetPreprocessingTests/Fixtures/preprocessing_params.json
```

### Step 4: Commit

```bash
git add Scripts/extract_preprocessing_params.py
git commit -m "feat: add parameter extraction script with full resampling spec"
```

---

## Task 3: Python Fixture Generator

**Goal:** Generate golden outputs at each pipeline stage for Swift validation.

**Files:**
- Create: `Scripts/generate_fixtures.py`
- Create: Fixture files in `Tests/nnUNetPreprocessingTests/Fixtures/`

### Step 1: Create fixture generator script

```python
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
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate preprocessing fixtures for Swift validation"
    )
    parser.add_argument(
        "--input-nifti",
        type=Path,
        required=True,
        help="Input NIfTI volume for fixture generation"
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
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fixtures"
    )

    args = parser.parse_args()

    generate_fixtures(
        args.input_nifti,
        args.plans_json,
        args.dataset_fingerprint,
        args.configuration,
        args.output_dir
    )


if __name__ == "__main__":
    main()
```

### Step 2: Make script executable and generate fixtures

```bash
chmod +x Scripts/generate_fixtures.py

# Generate fixtures (use a small test volume)
python3 Scripts/generate_fixtures.py \
    --input-nifti /path/to/small_test_ct.nii.gz \
    --plans-json /path/to/nnUNetPlans.json \
    --dataset-fingerprint /path/to/dataset_fingerprint.json \
    --configuration 3d_fullres \
    --output-dir Tests/nnUNetPreprocessingTests/Fixtures
```

### Step 3: Commit

```bash
git add Scripts/generate_fixtures.py
git commit -m "feat: add Python fixture generator for per-stage validation"
```

---

## Task 4: DICOM-Decoder Bridge

**Goal:** Convert `DicomSeriesVolume` to internal `VolumeBuffer` with HU conversion.

**Files:**
- Create: `Sources/nnUNetPreprocessing/Models/VolumeBuffer.swift`
- Create: `Sources/nnUNetPreprocessing/Models/BoundingBox.swift`
- Create: `Sources/nnUNetPreprocessing/Bridge/DicomBridge.swift`
- Create: `Tests/nnUNetPreprocessingTests/DicomBridgeTests.swift`

### Step 1: Create VolumeBuffer model

```swift
// Sources/nnUNetPreprocessing/Models/VolumeBuffer.swift

import Foundation
import simd

/// Internal volume representation for preprocessing pipeline
public struct VolumeBuffer: Sendable, Equatable {
    /// Raw voxel data (Float32)
    public var data: Data

    /// Shape in nnUNet convention: (depth, height, width)
    public var shape: (depth: Int, height: Int, width: Int)

    /// Physical spacing in mm: (z, y, x)
    public var spacing: SIMD3<Double>

    /// World coordinates origin (mm)
    public var origin: SIMD3<Double>

    /// Direction cosines from DICOM IOP
    public var orientation: simd_double3x3

    /// Bounding box after crop-to-nonzero (nil if not cropped)
    public var bbox: BoundingBox?

    /// Number of voxels
    public var voxelCount: Int {
        shape.depth * shape.height * shape.width
    }

    /// Size in bytes
    public var byteCount: Int {
        voxelCount * MemoryLayout<Float>.size
    }

    /// Initialize with raw data
    public init(
        data: Data,
        shape: (depth: Int, height: Int, width: Int),
        spacing: SIMD3<Double>,
        origin: SIMD3<Double> = .zero,
        orientation: simd_double3x3 = simd_double3x3(1),
        bbox: BoundingBox? = nil
    ) {
        self.data = data
        self.shape = shape
        self.spacing = spacing
        self.origin = origin
        self.orientation = orientation
        self.bbox = bbox
    }

    /// Access voxel data as Float array
    public func withUnsafeFloatPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeBytes { buffer in
            try body(buffer.bindMemory(to: Float.self).baseAddress!)
        }
    }

    /// Access voxel data as mutable Float array
    public mutating func withUnsafeMutableFloatPointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeMutableBytes { buffer in
            try body(buffer.bindMemory(to: Float.self).baseAddress!)
        }
    }
}

// MARK: - Equatable conformance for shape tuple
extension VolumeBuffer {
    public static func == (lhs: VolumeBuffer, rhs: VolumeBuffer) -> Bool {
        lhs.data == rhs.data &&
        lhs.shape.depth == rhs.shape.depth &&
        lhs.shape.height == rhs.shape.height &&
        lhs.shape.width == rhs.shape.width &&
        lhs.spacing == rhs.spacing &&
        lhs.origin == rhs.origin &&
        lhs.bbox == rhs.bbox
    }
}
```

### Step 2: Create BoundingBox model

```swift
// Sources/nnUNetPreprocessing/Models/BoundingBox.swift

import Foundation

/// Bounding box for crop-to-nonzero operation
public struct BoundingBox: Sendable, Equatable, Codable {
    /// Start indices (inclusive) for each axis: (z, y, x)
    public var start: (z: Int, y: Int, x: Int)

    /// End indices (exclusive) for each axis: (z, y, x)
    public var end: (z: Int, y: Int, x: Int)

    /// Size of bounding box
    public var size: (depth: Int, height: Int, width: Int) {
        (end.z - start.z, end.y - start.y, end.x - start.x)
    }

    public init(start: (z: Int, y: Int, x: Int), end: (z: Int, y: Int, x: Int)) {
        self.start = start
        self.end = end
    }

    // Codable conformance for tuple
    enum CodingKeys: String, CodingKey {
        case startZ, startY, startX
        case endZ, endY, endX
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        start = (
            try container.decode(Int.self, forKey: .startZ),
            try container.decode(Int.self, forKey: .startY),
            try container.decode(Int.self, forKey: .startX)
        )
        end = (
            try container.decode(Int.self, forKey: .endZ),
            try container.decode(Int.self, forKey: .endY),
            try container.decode(Int.self, forKey: .endX)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(start.z, forKey: .startZ)
        try container.encode(start.y, forKey: .startY)
        try container.encode(start.x, forKey: .startX)
        try container.encode(end.z, forKey: .endZ)
        try container.encode(end.y, forKey: .endY)
        try container.encode(end.x, forKey: .endX)
    }

    public static func == (lhs: BoundingBox, rhs: BoundingBox) -> Bool {
        lhs.start == rhs.start && lhs.end == rhs.end
    }
}
```

### Step 3: Create DicomBridge

```swift
// Sources/nnUNetPreprocessing/Bridge/DicomBridge.swift

import Foundation
import DicomCore
import simd

/// Bridge between DICOM-Decoder and internal VolumeBuffer representation
public struct DicomBridge {

    /// Convert DICOM-Decoder output to internal VolumeBuffer with HU conversion
    /// - Parameter volume: DicomSeriesVolume from DICOM-Decoder
    /// - Returns: VolumeBuffer with Float32 data in HU units
    public static func convert(_ volume: DicomSeriesVolume) -> VolumeBuffer {
        let voxelCount = volume.width * volume.height * volume.depth

        // Apply HU conversion: HU = raw * slope + intercept
        let float32Data = volume.voxels.withUnsafeBytes { rawBuffer -> Data in
            var floatArray = [Float](repeating: 0, count: voxelCount)

            let slope = Float(volume.rescaleSlope)
            let intercept = Float(volume.rescaleIntercept)

            if volume.isSignedPixel {
                let int16Ptr = rawBuffer.bindMemory(to: Int16.self)
                for i in 0..<voxelCount {
                    floatArray[i] = Float(int16Ptr[i]) * slope + intercept
                }
            } else {
                let uint16Ptr = rawBuffer.bindMemory(to: UInt16.self)
                for i in 0..<voxelCount {
                    floatArray[i] = Float(uint16Ptr[i]) * slope + intercept
                }
            }

            return floatArray.withUnsafeBytes { Data($0) }
        }

        // DICOM-Decoder uses W×H×D order, we convert to D×H×W for nnUNet
        // Note: The actual data layout remains the same, we just interpret axes differently
        // Full transpose will be handled in the Transpose step based on plans
        return VolumeBuffer(
            data: float32Data,
            shape: (depth: volume.depth, height: volume.height, width: volume.width),
            spacing: SIMD3(volume.spacing.z, volume.spacing.y, volume.spacing.x),
            origin: SIMD3(volume.origin.z, volume.origin.y, volume.origin.x),
            orientation: volume.orientation
        )
    }
}
```

### Step 4: Create test

```swift
// Tests/nnUNetPreprocessingTests/DicomBridgeTests.swift

import XCTest
@testable import nnUNetPreprocessing
import DicomCore

final class DicomBridgeTests: XCTestCase {

    func testHUConversionWithSignedPixels() {
        // Create mock DicomSeriesVolume
        let width = 4, height = 4, depth = 2
        let voxelCount = width * height * depth

        // Raw Int16 values
        var rawPixels = [Int16](repeating: 0, count: voxelCount)
        rawPixels[0] = -1000  // Should become -1000 * 1.0 + 0 = -1000 HU
        rawPixels[1] = 0      // Should become 0 HU
        rawPixels[2] = 1000   // Should become 1000 HU

        let rawData = rawPixels.withUnsafeBytes { Data($0) }

        let volume = DicomSeriesVolume(
            voxels: rawData,
            width: width,
            height: height,
            depth: depth,
            spacing: SIMD3(1.0, 1.0, 2.0),
            orientation: simd_double3x3(1),
            origin: SIMD3(0, 0, 0),
            rescaleSlope: 1.0,
            rescaleIntercept: 0.0,
            bitsAllocated: 16,
            isSignedPixel: true,
            seriesDescription: "Test"
        )

        let buffer = DicomBridge.convert(volume)

        // Verify HU conversion
        buffer.withUnsafeFloatPointer { ptr in
            XCTAssertEqual(ptr[0], -1000.0, accuracy: 0.001)
            XCTAssertEqual(ptr[1], 0.0, accuracy: 0.001)
            XCTAssertEqual(ptr[2], 1000.0, accuracy: 0.001)
        }

        // Verify shape (D×H×W)
        XCTAssertEqual(buffer.shape.depth, depth)
        XCTAssertEqual(buffer.shape.height, height)
        XCTAssertEqual(buffer.shape.width, width)

        // Verify spacing (z, y, x)
        XCTAssertEqual(buffer.spacing.x, 2.0, accuracy: 0.001)  // z in DICOM
        XCTAssertEqual(buffer.spacing.y, 1.0, accuracy: 0.001)  // y in DICOM
        XCTAssertEqual(buffer.spacing.z, 1.0, accuracy: 0.001)  // x in DICOM
    }

    func testHUConversionWithRescale() {
        let width = 2, height = 2, depth = 1
        var rawPixels: [Int16] = [0, 100, 200, 300]
        let rawData = rawPixels.withUnsafeBytes { Data($0) }

        let volume = DicomSeriesVolume(
            voxels: rawData,
            width: width,
            height: height,
            depth: depth,
            spacing: SIMD3(0.5, 0.5, 1.0),
            orientation: simd_double3x3(1),
            origin: .zero,
            rescaleSlope: 2.0,      // HU = raw * 2 - 1024
            rescaleIntercept: -1024.0,
            bitsAllocated: 16,
            isSignedPixel: true,
            seriesDescription: "Test"
        )

        let buffer = DicomBridge.convert(volume)

        buffer.withUnsafeFloatPointer { ptr in
            // 0 * 2 - 1024 = -1024
            XCTAssertEqual(ptr[0], -1024.0, accuracy: 0.001)
            // 100 * 2 - 1024 = -824
            XCTAssertEqual(ptr[1], -824.0, accuracy: 0.001)
            // 200 * 2 - 1024 = -624
            XCTAssertEqual(ptr[2], -624.0, accuracy: 0.001)
            // 300 * 2 - 1024 = -424
            XCTAssertEqual(ptr[3], -424.0, accuracy: 0.001)
        }
    }
}
```

### Step 5: Run tests

```bash
swift test --filter DicomBridgeTests
```

### Step 6: Commit

```bash
git add Sources/nnUNetPreprocessing/Models/ \
        Sources/nnUNetPreprocessing/Bridge/ \
        Tests/nnUNetPreprocessingTests/DicomBridgeTests.swift
git commit -m "feat: add DICOM-Decoder bridge with HU conversion"
```

---

## Task 5: CPU Preprocessing Pipeline

**Goal:** Implement each preprocessing stage in pure Swift CPU code, matching nnUNet exactly.

**Files:**
- Create: `Sources/nnUNetPreprocessing/CPU/Transpose.swift`
- Create: `Sources/nnUNetPreprocessing/CPU/CropToNonzero.swift`
- Create: `Sources/nnUNetPreprocessing/CPU/CTNormalization.swift`
- Create: `Sources/nnUNetPreprocessing/CPU/Resampling.swift`
- Create: `Sources/nnUNetPreprocessing/Models/CTNormalizationProperties.swift`
- Create: Tests for each

### Step 1: Create CTNormalizationProperties model

```swift
// Sources/nnUNetPreprocessing/Models/CTNormalizationProperties.swift

import Foundation

/// CT-specific normalization parameters extracted from dataset fingerprint
public struct CTNormalizationProperties: Sendable, Codable, Equatable {
    /// Mean intensity in foreground regions
    public let mean: Double

    /// Standard deviation in foreground regions
    public let std: Double

    /// Lower clipping bound (0.5th percentile)
    public let lowerBound: Double

    /// Upper clipping bound (99.5th percentile)
    public let upperBound: Double

    public init(mean: Double, std: Double, lowerBound: Double, upperBound: Double) {
        self.mean = mean
        self.std = std
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }
}
```

### Step 2: Implement Transpose

```swift
// Sources/nnUNetPreprocessing/CPU/Transpose.swift

import Foundation

/// Axis transpose operation matching numpy.transpose
public struct Transpose {

    /// Reorder axes according to transpose_forward from nnUNet plans
    /// - Parameters:
    ///   - volume: Input volume buffer
    ///   - axes: New axis order, e.g., [2, 1, 0] to reverse axes
    /// - Returns: Transposed volume buffer
    public static func apply(_ volume: VolumeBuffer, axes: [Int]) -> VolumeBuffer {
        let oldShape = [volume.shape.depth, volume.shape.height, volume.shape.width]
        let newShape = (
            depth: oldShape[axes[0]],
            height: oldShape[axes[1]],
            width: oldShape[axes[2]]
        )

        // If axes are identity [0, 1, 2], return copy
        if axes == [0, 1, 2] {
            return VolumeBuffer(
                data: volume.data,
                shape: newShape,
                spacing: volume.spacing,
                origin: volume.origin,
                orientation: volume.orientation,
                bbox: volume.bbox
            )
        }

        let voxelCount = volume.voxelCount
        var outputData = Data(count: voxelCount * MemoryLayout<Float>.size)

        volume.data.withUnsafeBytes { srcBuffer in
            outputData.withUnsafeMutableBytes { dstBuffer in
                let src = srcBuffer.bindMemory(to: Float.self)
                let dst = dstBuffer.bindMemory(to: Float.self)

                let oldD = oldShape[0], oldH = oldShape[1], oldW = oldShape[2]
                let newD = newShape.depth, newH = newShape.height, newW = newShape.width

                // Compute inverse permutation for source indexing
                var invAxes = [0, 0, 0]
                for i in 0..<3 {
                    invAxes[axes[i]] = i
                }

                for d in 0..<newD {
                    for h in 0..<newH {
                        for w in 0..<newW {
                            let newCoord = [d, h, w]
                            let oldCoord = [newCoord[invAxes[0]], newCoord[invAxes[1]], newCoord[invAxes[2]]]

                            let srcIdx = oldCoord[0] * (oldH * oldW) + oldCoord[1] * oldW + oldCoord[2]
                            let dstIdx = d * (newH * newW) + h * newW + w

                            dst[dstIdx] = src[srcIdx]
                        }
                    }
                }
            }
        }

        // Transpose spacing to match new axis order
        let oldSpacing = [volume.spacing.x, volume.spacing.y, volume.spacing.z]
        let newSpacing = SIMD3(oldSpacing[axes[0]], oldSpacing[axes[1]], oldSpacing[axes[2]])

        return VolumeBuffer(
            data: outputData,
            shape: newShape,
            spacing: newSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }
}
```

### Step 3: Implement CropToNonzero

```swift
// Sources/nnUNetPreprocessing/CPU/CropToNonzero.swift

import Foundation

/// Crop volume to bounding box of non-zero voxels
public struct CropToNonzero {

    /// Crop volume to smallest bounding box containing all non-zero voxels
    /// - Parameter volume: Input volume buffer
    /// - Returns: Tuple of (cropped volume, bounding box for inverse transform)
    public static func apply(_ volume: VolumeBuffer) -> (VolumeBuffer, BoundingBox) {
        let (d, h, w) = volume.shape

        // Find bounding box of non-zero voxels
        var minZ = d, maxZ = 0
        var minY = h, maxY = 0
        var minX = w, maxX = 0

        volume.withUnsafeFloatPointer { ptr in
            for z in 0..<d {
                for y in 0..<h {
                    for x in 0..<w {
                        let idx = z * (h * w) + y * w + x
                        if ptr[idx] != 0 {
                            minZ = min(minZ, z)
                            maxZ = max(maxZ, z)
                            minY = min(minY, y)
                            maxY = max(maxY, y)
                            minX = min(minX, x)
                            maxX = max(maxX, x)
                        }
                    }
                }
            }
        }

        // Handle case where all voxels are zero
        if minZ > maxZ {
            let bbox = BoundingBox(start: (0, 0, 0), end: (d, h, w))
            return (volume, bbox)
        }

        let bbox = BoundingBox(
            start: (minZ, minY, minX),
            end: (maxZ + 1, maxY + 1, maxX + 1)
        )

        let newShape = bbox.size
        let newVoxelCount = newShape.depth * newShape.height * newShape.width
        var outputData = Data(count: newVoxelCount * MemoryLayout<Float>.size)

        volume.data.withUnsafeBytes { srcBuffer in
            outputData.withUnsafeMutableBytes { dstBuffer in
                let src = srcBuffer.bindMemory(to: Float.self)
                let dst = dstBuffer.bindMemory(to: Float.self)

                var dstIdx = 0
                for z in bbox.start.z..<bbox.end.z {
                    for y in bbox.start.y..<bbox.end.y {
                        for x in bbox.start.x..<bbox.end.x {
                            let srcIdx = z * (h * w) + y * w + x
                            dst[dstIdx] = src[srcIdx]
                            dstIdx += 1
                        }
                    }
                }
            }
        }

        return (
            VolumeBuffer(
                data: outputData,
                shape: newShape,
                spacing: volume.spacing,
                origin: volume.origin,
                orientation: volume.orientation,
                bbox: bbox
            ),
            bbox
        )
    }
}
```

### Step 4: Implement CTNormalization

```swift
// Sources/nnUNetPreprocessing/CPU/CTNormalization.swift

import Foundation

/// CT normalization: clip to percentile bounds, then z-score normalize
public struct CTNormalization {

    /// Apply CT normalization matching nnUNet's CTNormalization scheme
    /// - Parameters:
    ///   - volume: Input volume buffer (in HU)
    ///   - properties: Normalization parameters from dataset fingerprint
    /// - Returns: Normalized volume buffer
    public static func apply(
        _ volume: VolumeBuffer,
        properties: CTNormalizationProperties
    ) -> VolumeBuffer {
        let voxelCount = volume.voxelCount
        var outputData = Data(count: voxelCount * MemoryLayout<Float>.size)

        let lower = Float(properties.lowerBound)
        let upper = Float(properties.upperBound)
        let mean = Float(properties.mean)
        let std = Float(max(properties.std, 1e-8))  // Avoid division by zero

        volume.data.withUnsafeBytes { srcBuffer in
            outputData.withUnsafeMutableBytes { dstBuffer in
                let src = srcBuffer.bindMemory(to: Float.self)
                let dst = dstBuffer.bindMemory(to: Float.self)

                for i in 0..<voxelCount {
                    // 1. Clip to percentile bounds
                    var value = src[i]
                    value = min(max(value, lower), upper)

                    // 2. Z-score normalization
                    value = (value - mean) / std

                    dst[i] = value
                }
            }
        }

        return VolumeBuffer(
            data: outputData,
            shape: volume.shape,
            spacing: volume.spacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }
}
```

### Step 5: Implement Resampling (CPU reference)

```swift
// Sources/nnUNetPreprocessing/CPU/Resampling.swift

import Foundation
import simd

/// Resampling to target spacing matching nnUNet's resample_data_or_seg_to_shape
public struct Resampling {

    /// Resample volume to target spacing using cubic interpolation
    /// - Parameters:
    ///   - volume: Input volume buffer
    ///   - targetSpacing: Target spacing in mm (z, y, x)
    ///   - order: Interpolation order (3 = cubic for data, 1 = linear for seg)
    ///   - orderZ: Interpolation order for Z axis when using separate-Z
    ///   - forceSeparateZ: Override automatic separate-Z detection
    ///   - anisotropyThreshold: Threshold for automatic separate-Z (default 3.0)
    /// - Returns: Resampled volume buffer
    public static func apply(
        _ volume: VolumeBuffer,
        targetSpacing: SIMD3<Double>,
        order: Int = 3,
        orderZ: Int = 0,
        forceSeparateZ: Bool? = nil,
        anisotropyThreshold: Double = 3.0
    ) -> VolumeBuffer {
        let currentSpacing = volume.spacing

        // Compute target shape
        let scaleFactors = currentSpacing / targetSpacing
        let targetShape = (
            depth: Int(round(Double(volume.shape.depth) * scaleFactors.x)),
            height: Int(round(Double(volume.shape.height) * scaleFactors.y)),
            width: Int(round(Double(volume.shape.width) * scaleFactors.z))
        )

        // Determine if we should use separate-Z resampling
        let useSeparateZ: Bool
        if let force = forceSeparateZ {
            useSeparateZ = force
        } else {
            useSeparateZ = shouldUseSeparateZ(
                spacing: currentSpacing,
                threshold: anisotropyThreshold
            )
        }

        if useSeparateZ {
            return resampleSeparateZ(
                volume,
                targetShape: targetShape,
                targetSpacing: targetSpacing,
                orderXY: order,
                orderZ: orderZ
            )
        } else {
            return resampleCubic(
                volume,
                targetShape: targetShape,
                targetSpacing: targetSpacing
            )
        }
    }

    /// Determine if separate-Z resampling should be used based on anisotropy
    public static func shouldUseSeparateZ(
        spacing: SIMD3<Double>,
        threshold: Double
    ) -> Bool {
        let minSpacing = min(spacing.x, min(spacing.y, spacing.z))
        let maxSpacing = max(spacing.x, max(spacing.y, spacing.z))
        let ratio = maxSpacing / minSpacing
        return ratio > threshold
    }

    // MARK: - Private Implementation

    private static func resampleCubic(
        _ volume: VolumeBuffer,
        targetShape: (depth: Int, height: Int, width: Int),
        targetSpacing: SIMD3<Double>
    ) -> VolumeBuffer {
        let (srcD, srcH, srcW) = volume.shape
        let (dstD, dstH, dstW) = targetShape
        let dstVoxelCount = dstD * dstH * dstW

        var outputData = Data(count: dstVoxelCount * MemoryLayout<Float>.size)

        volume.data.withUnsafeBytes { srcBuffer in
            outputData.withUnsafeMutableBytes { dstBuffer in
                let src = srcBuffer.bindMemory(to: Float.self)
                let dst = dstBuffer.bindMemory(to: Float.self)

                // Scale factors for coordinate mapping
                let scaleZ = Double(srcD - 1) / Double(max(dstD - 1, 1))
                let scaleY = Double(srcH - 1) / Double(max(dstH - 1, 1))
                let scaleX = Double(srcW - 1) / Double(max(dstW - 1, 1))

                for dz in 0..<dstD {
                    for dy in 0..<dstH {
                        for dx in 0..<dstW {
                            // Map to source coordinates
                            let sz = Double(dz) * scaleZ
                            let sy = Double(dy) * scaleY
                            let sx = Double(dx) * scaleX

                            // Cubic interpolation
                            let value = cubicInterpolate3D(
                                src: src,
                                shape: (srcD, srcH, srcW),
                                z: sz, y: sy, x: sx
                            )

                            let dstIdx = dz * (dstH * dstW) + dy * dstW + dx
                            dst[dstIdx] = value
                        }
                    }
                }
            }
        }

        return VolumeBuffer(
            data: outputData,
            shape: targetShape,
            spacing: targetSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }

    private static func resampleSeparateZ(
        _ volume: VolumeBuffer,
        targetShape: (depth: Int, height: Int, width: Int),
        targetSpacing: SIMD3<Double>,
        orderXY: Int,
        orderZ: Int
    ) -> VolumeBuffer {
        // Step 1: Resample in-plane (X-Y) with cubic interpolation
        // Step 2: Resample through-plane (Z) with nearest-neighbor (orderZ=0)

        let (srcD, srcH, srcW) = volume.shape
        let (dstD, dstH, dstW) = targetShape

        // First pass: resample X-Y for each slice
        let intermediateShape = (depth: srcD, height: dstH, width: dstW)
        let intermediateCount = srcD * dstH * dstW
        var intermediateData = Data(count: intermediateCount * MemoryLayout<Float>.size)

        volume.data.withUnsafeBytes { srcBuffer in
            intermediateData.withUnsafeMutableBytes { intBuffer in
                let src = srcBuffer.bindMemory(to: Float.self)
                let intermediate = intBuffer.bindMemory(to: Float.self)

                let scaleY = Double(srcH - 1) / Double(max(dstH - 1, 1))
                let scaleX = Double(srcW - 1) / Double(max(dstW - 1, 1))

                for z in 0..<srcD {
                    for dy in 0..<dstH {
                        for dx in 0..<dstW {
                            let sy = Double(dy) * scaleY
                            let sx = Double(dx) * scaleX

                            let value = cubicInterpolate2D(
                                src: src,
                                sliceOffset: z * srcH * srcW,
                                height: srcH,
                                width: srcW,
                                y: sy,
                                x: sx
                            )

                            let intIdx = z * (dstH * dstW) + dy * dstW + dx
                            intermediate[intIdx] = value
                        }
                    }
                }
            }
        }

        // Second pass: resample Z with nearest-neighbor or linear
        let dstVoxelCount = dstD * dstH * dstW
        var outputData = Data(count: dstVoxelCount * MemoryLayout<Float>.size)

        intermediateData.withUnsafeBytes { intBuffer in
            outputData.withUnsafeMutableBytes { dstBuffer in
                let intermediate = intBuffer.bindMemory(to: Float.self)
                let dst = dstBuffer.bindMemory(to: Float.self)

                let scaleZ = Double(srcD - 1) / Double(max(dstD - 1, 1))

                for dz in 0..<dstD {
                    let sz = Double(dz) * scaleZ

                    for dy in 0..<dstH {
                        for dx in 0..<dstW {
                            let value: Float
                            if orderZ == 0 {
                                // Nearest-neighbor
                                let nearestZ = min(Int(round(sz)), srcD - 1)
                                let intIdx = nearestZ * (dstH * dstW) + dy * dstW + dx
                                value = intermediate[intIdx]
                            } else {
                                // Linear interpolation along Z
                                let z0 = Int(floor(sz))
                                let z1 = min(z0 + 1, srcD - 1)
                                let t = Float(sz - Double(z0))

                                let idx0 = z0 * (dstH * dstW) + dy * dstW + dx
                                let idx1 = z1 * (dstH * dstW) + dy * dstW + dx

                                value = intermediate[idx0] * (1 - t) + intermediate[idx1] * t
                            }

                            let dstIdx = dz * (dstH * dstW) + dy * dstW + dx
                            dst[dstIdx] = value
                        }
                    }
                }
            }
        }

        return VolumeBuffer(
            data: outputData,
            shape: targetShape,
            spacing: targetSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }

    // MARK: - Interpolation Helpers

    /// Cubic B-spline weight function matching skimage order=3
    private static func cubicWeight(_ t: Float) -> Float {
        let at = abs(t)
        if at < 1.0 {
            return (1.5 * at - 2.5) * at * at + 1.0
        } else if at < 2.0 {
            return ((-0.5 * at + 2.5) * at - 4.0) * at + 2.0
        }
        return 0.0
    }

    /// Edge-clamped coordinate (mode='edge' in skimage)
    private static func edgeClamp(_ coord: Int, _ size: Int) -> Int {
        max(0, min(coord, size - 1))
    }

    /// 3D cubic interpolation with edge padding
    private static func cubicInterpolate3D(
        src: UnsafePointer<Float>,
        shape: (Int, Int, Int),
        z: Double, y: Double, x: Double
    ) -> Float {
        let (d, h, w) = shape

        let iz = Int(floor(z))
        let iy = Int(floor(y))
        let ix = Int(floor(x))

        let fz = Float(z - Double(iz))
        let fy = Float(y - Double(iy))
        let fx = Float(x - Double(ix))

        var result: Float = 0

        for dz in -1...2 {
            let wz = cubicWeight(fz - Float(dz))
            let cz = edgeClamp(iz + dz, d)

            for dy in -1...2 {
                let wy = cubicWeight(fy - Float(dy))
                let cy = edgeClamp(iy + dy, h)

                for dx in -1...2 {
                    let wx = cubicWeight(fx - Float(dx))
                    let cx = edgeClamp(ix + dx, w)

                    let idx = cz * (h * w) + cy * w + cx
                    result += src[idx] * wz * wy * wx
                }
            }
        }

        return result
    }

    /// 2D cubic interpolation for a single slice
    private static func cubicInterpolate2D(
        src: UnsafePointer<Float>,
        sliceOffset: Int,
        height: Int,
        width: Int,
        y: Double,
        x: Double
    ) -> Float {
        let iy = Int(floor(y))
        let ix = Int(floor(x))

        let fy = Float(y - Double(iy))
        let fx = Float(x - Double(ix))

        var result: Float = 0

        for dy in -1...2 {
            let wy = cubicWeight(fy - Float(dy))
            let cy = edgeClamp(iy + dy, height)

            for dx in -1...2 {
                let wx = cubicWeight(fx - Float(dx))
                let cx = edgeClamp(ix + dx, width)

                let idx = sliceOffset + cy * width + cx
                result += src[idx] * wy * wx
            }
        }

        return result
    }
}
```

### Step 6: Commit CPU implementations

```bash
git add Sources/nnUNetPreprocessing/CPU/ \
        Sources/nnUNetPreprocessing/Models/CTNormalizationProperties.swift
git commit -m "feat: implement CPU preprocessing pipeline (transpose, crop, normalize, resample)"
```

---

## Task 6: Metal CT Normalization

**Goal:** Port CT normalization to Metal after CPU reference passes fixture validation.

**Files:**
- Create: `Sources/nnUNetPreprocessing/Metal/Shaders/CTNormalization.metal`
- Create: `Sources/nnUNetPreprocessing/Metal/MetalCTNormalizer.swift`
- Create: `Tests/nnUNetPreprocessingTests/MetalCTNormalizationTests.swift`

### Step 1: Create Metal shader

```metal
// Sources/nnUNetPreprocessing/Metal/Shaders/CTNormalization.metal

#include <metal_stdlib>
using namespace metal;

/// CT normalization parameters
struct CTNormParams {
    float mean;
    float std;
    float lowerBound;
    float upperBound;
};

/// CT normalization kernel
/// Clips to percentile bounds and applies z-score normalization
kernel void ct_normalize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CTNormParams& params [[buffer(2)]],
    constant uint& voxelCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= voxelCount) return;

    float value = input[gid];

    // Clip to percentile bounds
    value = clamp(value, params.lowerBound, params.upperBound);

    // Z-score normalization
    value = (value - params.mean) / max(params.std, 1e-8f);

    output[gid] = value;
}
```

### Step 2: Create Metal wrapper

```swift
// Sources/nnUNetPreprocessing/Metal/MetalCTNormalizer.swift

import Metal
import Foundation

/// GPU-accelerated CT normalization using Metal compute shaders
public actor MetalCTNormalizer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    /// Initialize with Metal device
    public init(device: MTLDevice) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.failedToCreateCommandQueue
        }
        self.commandQueue = queue

        // Load shader from bundle
        guard let library = try? device.makeDefaultLibrary(bundle: Bundle.module) else {
            throw MetalError.failedToLoadLibrary
        }

        guard let function = library.makeFunction(name: "ct_normalize") else {
            throw MetalError.failedToFindFunction("ct_normalize")
        }

        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Normalize volume using GPU
    public func normalize(
        _ volume: VolumeBuffer,
        properties: CTNormalizationProperties
    ) async throws -> VolumeBuffer {
        let voxelCount = volume.voxelCount
        let bufferSize = voxelCount * MemoryLayout<Float>.size

        // Create input buffer
        guard let inputBuffer = device.makeBuffer(
            bytes: (volume.data as NSData).bytes,
            length: bufferSize,
            options: .storageModeShared
        ) else {
            throw MetalError.failedToCreateBuffer
        }

        // Create output buffer
        guard let outputBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        ) else {
            throw MetalError.failedToCreateBuffer
        }

        // Create parameters buffer
        var params = CTNormParams(
            mean: Float(properties.mean),
            std: Float(properties.std),
            lowerBound: Float(properties.lowerBound),
            upperBound: Float(properties.upperBound)
        )

        var count = UInt32(voxelCount)

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.failedToCreateCommandBuffer
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<CTNormParams>.size, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

        // Dispatch
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (voxelCount + 255) / 256,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Extract output data
        let outputData = Data(
            bytes: outputBuffer.contents(),
            count: bufferSize
        )

        return VolumeBuffer(
            data: outputData,
            shape: volume.shape,
            spacing: volume.spacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }
}

// MARK: - Supporting Types

struct CTNormParams {
    var mean: Float
    var std: Float
    var lowerBound: Float
    var upperBound: Float
}

public enum MetalError: Error {
    case failedToCreateCommandQueue
    case failedToLoadLibrary
    case failedToFindFunction(String)
    case failedToCreateBuffer
    case failedToCreateCommandBuffer
}
```

### Step 3: Commit Metal normalization

```bash
git add Sources/nnUNetPreprocessing/Metal/
git commit -m "feat: add Metal CT normalization shader and wrapper"
```

---

## Task 7: Metal Resampling

**Goal:** Port resampling to Metal after CPU reference passes fixture validation.

**Files:**
- Create: `Sources/nnUNetPreprocessing/Metal/Shaders/Resampling.metal`
- Create: `Sources/nnUNetPreprocessing/Metal/MetalResampler.swift`

### Implementation Note

Metal resampling is the most complex shader. It should only be implemented after:
1. CPU resampling passes all fixture tests
2. CPU normalization passes all fixture tests
3. Metal normalization matches CPU reference

The Metal resampling shader needs to implement:
- Cubic B-spline interpolation (4×4×4 neighborhood)
- Edge padding (mode='edge')
- Separate-Z logic (cubic in-plane, nearest through-plane)

This task is marked as **deferred until CPU validation is complete**.

---

## Task 8: Integration Tests with Fixtures

**Goal:** Validate each pipeline stage against Python-generated fixtures.

**Files:**
- Create: `Tests/nnUNetPreprocessingTests/FixtureValidationTests.swift`
- Create: `Tests/nnUNetPreprocessingTests/Helpers/FixtureLoader.swift`
- Create: `Tests/nnUNetPreprocessingTests/Helpers/ArrayComparison.swift`

### Step 1: Create fixture loader

```swift
// Tests/nnUNetPreprocessingTests/Helpers/FixtureLoader.swift

import Foundation
@testable import nnUNetPreprocessing

/// Helper for loading numpy fixtures in tests
struct FixtureLoader {

    /// Load a .npy file as VolumeBuffer
    static func loadNpy(_ name: String) throws -> (data: [Float], shape: [Int]) {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy") else {
            throw FixtureError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        return try parseNpy(data)
    }

    /// Load preprocessing parameters
    static func loadParams() throws -> PreprocessingParameters {
        guard let url = Bundle.module.url(
            forResource: "preprocessing_params",
            withExtension: "json"
        ) else {
            throw FixtureError.fileNotFound("preprocessing_params.json")
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PreprocessingParameters.self, from: data)
    }

    /// Load fixture metadata
    static func loadMetadata() throws -> FixtureMetadata {
        guard let url = Bundle.module.url(
            forResource: "fixture_metadata",
            withExtension: "json"
        ) else {
            throw FixtureError.fileNotFound("fixture_metadata.json")
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(FixtureMetadata.self, from: data)
    }

    // MARK: - NPY Parser

    private static func parseNpy(_ data: Data) throws -> (data: [Float], shape: [Int]) {
        // Simple .npy parser for float32 arrays
        // Format: magic (6 bytes) + version (2 bytes) + header_len (2 bytes) + header + data

        guard data.count > 10 else {
            throw FixtureError.invalidFormat
        }

        // Check magic number
        let magic = data.prefix(6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw FixtureError.invalidFormat
        }

        // Get header length
        let headerLen = data.subdata(in: 8..<10).withUnsafeBytes {
            $0.load(as: UInt16.self)
        }

        // Parse header (simplified - assumes float32, C-order)
        let headerStart = 10
        let headerEnd = headerStart + Int(headerLen)
        guard let headerString = String(data: data.subdata(in: headerStart..<headerEnd), encoding: .ascii) else {
            throw FixtureError.invalidFormat
        }

        // Extract shape from header (e.g., "'shape': (64, 64, 32)")
        let shape = parseShape(from: headerString)

        // Extract float32 data
        let dataStart = headerEnd
        let floatData = data.subdata(in: dataStart..<data.count)
        let floatCount = floatData.count / MemoryLayout<Float>.size

        let floats = floatData.withUnsafeBytes { buffer -> [Float] in
            Array(buffer.bindMemory(to: Float.self))
        }

        return (floats, shape)
    }

    private static func parseShape(from header: String) -> [Int] {
        // Extract shape tuple from numpy header
        guard let shapeStart = header.range(of: "'shape': ("),
              let shapeEnd = header.range(of: ")", range: shapeStart.upperBound..<header.endIndex) else {
            return []
        }

        let shapeString = header[shapeStart.upperBound..<shapeEnd.lowerBound]
        return shapeString
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
}

enum FixtureError: Error {
    case fileNotFound(String)
    case invalidFormat
}

struct FixtureMetadata: Codable {
    let inputFile: String
    let configuration: String
    let stages: [String: StageMetadata]
    let checksums: [String: String]

    enum CodingKeys: String, CodingKey {
        case inputFile = "input_file"
        case configuration
        case stages
        case checksums
    }
}

struct StageMetadata: Codable {
    let shape: [Int]?
    let spacing: [Double]?
    let bbox: [[Int]]?
    let mean: Double?
    let std: Double?
}
```

### Step 2: Create array comparison helper

```swift
// Tests/nnUNetPreprocessingTests/Helpers/ArrayComparison.swift

import XCTest

/// Helper for comparing float arrays with tolerance
struct ArrayComparison {

    /// Assert two float arrays are equal within tolerance
    /// - Parameters:
    ///   - actual: Actual values
    ///   - expected: Expected values
    ///   - tolerance: Maximum allowed absolute difference
    ///   - message: Failure message prefix
    static func assertEqual(
        _ actual: [Float],
        _ expected: [Float],
        tolerance: Float,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.count,
            expected.count,
            "Array sizes differ: \(actual.count) vs \(expected.count)",
            file: file,
            line: line
        )

        var maxDiff: Float = 0
        var maxDiffIndex = 0
        var diffCount = 0

        for i in 0..<actual.count {
            let diff = abs(actual[i] - expected[i])
            if diff > tolerance {
                diffCount += 1
            }
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIndex = i
            }
        }

        if maxDiff > tolerance {
            XCTFail(
                "Arrays differ: max diff = \(maxDiff) at index \(maxDiffIndex) " +
                "(actual: \(actual[maxDiffIndex]), expected: \(expected[maxDiffIndex]])), " +
                "\(diffCount) values exceed tolerance \(tolerance)",
                file: file,
                line: line
            )
        }
    }

    /// Compute mean absolute error between arrays
    static func meanAbsoluteError(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !actual.isEmpty else { return Float.infinity }

        var sum: Float = 0
        for i in 0..<actual.count {
            sum += abs(actual[i] - expected[i])
        }
        return sum / Float(actual.count)
    }
}
```

### Step 3: Create fixture validation tests

```swift
// Tests/nnUNetPreprocessingTests/FixtureValidationTests.swift

import XCTest
@testable import nnUNetPreprocessing

final class FixtureValidationTests: XCTestCase {

    var params: PreprocessingParameters!
    var metadata: FixtureMetadata!

    override func setUp() async throws {
        try await super.setUp()

        do {
            params = try FixtureLoader.loadParams()
            metadata = try FixtureLoader.loadMetadata()
        } catch {
            throw XCTSkip("Fixtures not available: \(error)")
        }
    }

    // MARK: - Transpose Tests

    func testTransposeMatchesPythonFixture() throws {
        let (rawData, rawShape) = try FixtureLoader.loadNpy("01_raw")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("02_transposed")

        let input = VolumeBuffer(
            data: rawData.withUnsafeBytes { Data($0) },
            shape: (rawShape[0], rawShape[1], rawShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        let result = Transpose.apply(input, axes: params.transposeForward)

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0])
        XCTAssertEqual(result.shape.height, expectedShape[1])
        XCTAssertEqual(result.shape.width, expectedShape[2])

        // Verify data (exact match expected)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.0)
        }
    }

    // MARK: - Crop Tests

    func testCropToNonzeroMatchesPythonFixture() throws {
        let (transposedData, transposedShape) = try FixtureLoader.loadNpy("02_transposed")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("03_cropped")

        let input = VolumeBuffer(
            data: transposedData.withUnsafeBytes { Data($0) },
            shape: (transposedShape[0], transposedShape[1], transposedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        let (result, bbox) = CropToNonzero.apply(input)

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0])
        XCTAssertEqual(result.shape.height, expectedShape[1])
        XCTAssertEqual(result.shape.width, expectedShape[2])

        // Verify bbox matches metadata
        if let expectedBbox = metadata.stages["03_cropped"]?.bbox {
            XCTAssertEqual(bbox.start.z, expectedBbox[0][0])
            XCTAssertEqual(bbox.end.z, expectedBbox[0][1])
            XCTAssertEqual(bbox.start.y, expectedBbox[1][0])
            XCTAssertEqual(bbox.end.y, expectedBbox[1][1])
            XCTAssertEqual(bbox.start.x, expectedBbox[2][0])
            XCTAssertEqual(bbox.end.x, expectedBbox[2][1])
        }

        // Verify data (exact match expected)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.0)
        }
    }

    // MARK: - Normalization Tests

    func testCTNormalizationMatchesPythonFixture() throws {
        let (croppedData, croppedShape) = try FixtureLoader.loadNpy("03_cropped")
        let (expectedData, _) = try FixtureLoader.loadNpy("04_normalized")

        guard let ctProps = params.ctNormalizationProperties else {
            throw XCTSkip("CT normalization properties not available")
        }

        let input = VolumeBuffer(
            data: croppedData.withUnsafeBytes { Data($0) },
            shape: (croppedShape[0], croppedShape[1], croppedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        let result = CTNormalization.apply(input, properties: ctProps)

        // Verify data (tolerance for floating point)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            let mae = ArrayComparison.meanAbsoluteError(actual, expectedData)
            XCTAssertLessThan(mae, 0.01, "MAE should be < 0.01, got \(mae)")
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.01)
        }
    }

    // MARK: - Resampling Tests

    func testResamplingMatchesPythonFixture() throws {
        let (normalizedData, normalizedShape) = try FixtureLoader.loadNpy("04_normalized")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("05_resampled")

        // Get spacing from metadata
        guard let originalSpacing = metadata.stages["04_normalized"]?.spacing,
              let targetSpacing = metadata.stages["05_resampled"]?.spacing else {
            throw XCTSkip("Spacing metadata not available")
        }

        let input = VolumeBuffer(
            data: normalizedData.withUnsafeBytes { Data($0) },
            shape: (normalizedShape[0], normalizedShape[1], normalizedShape[2]),
            spacing: SIMD3(originalSpacing[0], originalSpacing[1], originalSpacing[2])
        )

        let result = Resampling.apply(
            input,
            targetSpacing: SIMD3(targetSpacing[0], targetSpacing[1], targetSpacing[2]),
            order: params.resamplingOrder ?? 3,
            orderZ: params.resamplingOrderZ ?? 0
        )

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0])
        XCTAssertEqual(result.shape.height, expectedShape[1])
        XCTAssertEqual(result.shape.width, expectedShape[2])

        // Verify data (higher tolerance for resampling due to interpolation)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            let mae = ArrayComparison.meanAbsoluteError(actual, expectedData)
            XCTAssertLessThan(mae, 0.5, "MAE should be < 0.5, got \(mae)")
        }
    }

    // MARK: - Metal vs CPU Tests

    func testMetalNormalizationMatchesCPU() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }

        let (croppedData, croppedShape) = try FixtureLoader.loadNpy("03_cropped")

        guard let ctProps = params.ctNormalizationProperties else {
            throw XCTSkip("CT normalization properties not available")
        }

        let input = VolumeBuffer(
            data: croppedData.withUnsafeBytes { Data($0) },
            shape: (croppedShape[0], croppedShape[1], croppedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        // CPU reference
        let cpuResult = CTNormalization.apply(input, properties: ctProps)

        // Metal implementation
        let metalNormalizer = try MetalCTNormalizer(device: device)
        let metalResult = try await metalNormalizer.normalize(input, properties: ctProps)

        // Compare
        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                let cpu = Array(UnsafeBufferPointer(start: cpuPtr, count: cpuResult.voxelCount))
                let metal = Array(UnsafeBufferPointer(start: metalPtr, count: metalResult.voxelCount))

                let mae = ArrayComparison.meanAbsoluteError(cpu, metal)
                XCTAssertLessThan(mae, 0.001, "Metal should match CPU within 0.001, got \(mae)")
            }
        }
    }
}
```

### Step 4: Commit tests

```bash
git add Tests/nnUNetPreprocessingTests/
git commit -m "test: add fixture validation tests for preprocessing pipeline"
```

---

## Task 9: Final Validation & Documentation

### Step 1: Run all tests

```bash
swift test
```

Expected: All tests pass

### Step 2: Verify against Python reference

```bash
# Generate fresh fixtures
python3 Scripts/generate_fixtures.py \
    --input-nifti /path/to/test_ct.nii.gz \
    --plans-json /path/to/nnUNetPlans.json \
    --dataset-fingerprint /path/to/dataset_fingerprint.json \
    --output-dir Tests/nnUNetPreprocessingTests/Fixtures

# Run validation tests
swift test --filter FixtureValidationTests
```

### Step 3: Commit and tag

```bash
git add .
git commit -m "feat: Phase 1 complete - Metal preprocessing pipeline with fixture validation"
git tag -a v1.0.0-phase1 -m "Phase 1: Metal preprocessing pipeline"
```

---

## Success Criteria

Phase 1 is complete when:

| Criterion | Target | Validation |
|-----------|--------|------------|
| Transpose | Exact match (0 tolerance) | `testTransposeMatchesPythonFixture` |
| Crop to Nonzero | Exact bbox match | `testCropToNonzeroMatchesPythonFixture` |
| CT Normalization | MAE < 0.01 | `testCTNormalizationMatchesPythonFixture` |
| Resampling | MAE < 0.5 voxels | `testResamplingMatchesPythonFixture` |
| Metal vs CPU | MAE < 0.001 | `testMetalNormalizationMatchesCPU` |
| Test Coverage | > 80% | `swift test --enable-code-coverage` |
| Build | No warnings | `swift build` |

---

## Next Phase

After completing Phase 1, proceed to **Phase 2: Core ML Model Integration** which will:
1. Convert trained nnUNet PyTorch model to Core ML
2. Implement inference pipeline using preprocessed volumes
3. Create Swift wrapper for model execution
4. Validate segmentation output against Python reference

---

**Plan Status:** Ready for Execution

**Last Updated:** 2026-01-10

**Dependencies:**
- DICOM-Decoder at `/Users/leandroalmeida/DICOM-Decoder`
- MTK at `/Users/leandroalmeida/MTK`

**Addresses GPT Audit Issues:**
- [x] P0: Plan matches nnUNet pipeline order (transpose → crop → normalize → resample)
- [x] P0: Swift/Metal code bugs fixed (Float types, actor isolation, texture flags)
- [x] P0: Resampling uses cubic B-spline with edge padding (not Lanczos)
- [x] P0: iOS app architecture gap solved (DICOM-Decoder integration)
- [x] P1: Full resampling params extracted from plans
- [x] P1: Fixture-based validation for correctness
