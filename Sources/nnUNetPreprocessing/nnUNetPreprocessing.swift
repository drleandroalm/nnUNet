/// nnUNet Preprocessing Pipeline for iOS/macOS
///
/// This package provides GPU-accelerated preprocessing for CT DICOM volumes,
/// matching nnUNet's preprocessing behavior exactly with fixture-validated correctness.
///
/// Pipeline stages:
/// 1. HU Conversion - Apply rescaleSlope/Intercept to convert to Hounsfield Units
/// 2. Transpose - Reorder axes per nnUNet plans.transpose_forward
/// 3. Crop to Nonzero - Remove background, store bbox for inverse transform
/// 4. Normalize - CT: clip to percentiles, z-score normalize
/// 5. Resample - Cubic interpolation to target spacing
public enum nnUNetPreprocessing {
    /// Library version
    public static let version = "1.0.0-phase1"
}
