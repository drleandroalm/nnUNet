// Sources/nnUNetPreprocessing/Models/PreprocessingParameters.swift

import Foundation

/// Preprocessing parameters extracted from nnUNet plans and dataset fingerprint
/// Used to configure the preprocessing pipeline for inference
public struct PreprocessingParameters: Sendable, Codable, Equatable {

    // MARK: - Basic Configuration

    /// Configuration name (e.g., "3d_fullres")
    public let configurationName: String

    /// Target spacing in mm (z, y, x)
    public let targetSpacing: [Double]

    /// Patch size for inference
    public let patchSize: [Int]

    // MARK: - Transpose Configuration

    /// Axis order for forward transpose (e.g., [0, 1, 2] for identity)
    public let transposeForward: [Int]

    /// Axis order for backward transpose (inverse of forward)
    public let transposeBackward: [Int]

    // MARK: - Normalization Configuration

    /// Normalization schemes to apply (e.g., ["CTNormalization"])
    public let normalizationSchemes: [String]

    /// Whether to use mask for normalization per channel
    public let useMaskForNorm: [Bool]

    /// Foreground intensity properties per channel
    public let foregroundIntensityProperties: [String: ChannelIntensityProperties]

    // MARK: - Resampling Configuration

    /// Resampling function name for data
    public let resamplingFnData: String

    /// Resampling function kwargs for data
    public let resamplingFnDataKwargs: ResamplingKwargs

    /// Resampling function name for segmentation
    public let resamplingFnSeg: String

    /// Resampling function kwargs for segmentation
    public let resamplingFnSegKwargs: ResamplingKwargs

    /// Threshold for automatic separate-Z detection
    public let anisotropyThreshold: Double

    // MARK: - Dataset Properties

    /// Original spacing from dataset fingerprint
    public let originalSpacing: [Double]

    /// Original median shape from dataset fingerprint
    public let originalMedianShape: [Int]

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case configurationName = "configuration_name"
        case targetSpacing = "target_spacing"
        case patchSize = "patch_size"
        case transposeForward = "transpose_forward"
        case transposeBackward = "transpose_backward"
        case normalizationSchemes = "normalization_schemes"
        case useMaskForNorm = "use_mask_for_norm"
        case foregroundIntensityProperties = "foreground_intensity_properties"
        case resamplingFnData = "resampling_fn_data"
        case resamplingFnDataKwargs = "resampling_fn_data_kwargs"
        case resamplingFnSeg = "resampling_fn_seg"
        case resamplingFnSegKwargs = "resampling_fn_seg_kwargs"
        case anisotropyThreshold = "anisotropy_threshold"
        case originalSpacing = "original_spacing"
        case originalMedianShape = "original_median_shape"
    }

    // MARK: - Computed Properties

    /// CT normalization properties for channel 0 (if available)
    public var ctNormalizationProperties: CTNormalizationProperties? {
        guard let props = foregroundIntensityProperties["0"] else { return nil }
        return CTNormalizationProperties(
            mean: props.mean,
            std: props.std,
            lowerBound: props.percentile005,
            upperBound: props.percentile995
        )
    }

    /// Interpolation order for data resampling
    public var resamplingOrder: Int {
        resamplingFnDataKwargs.order
    }

    /// Interpolation order for Z axis in separate-Z mode
    public var resamplingOrderZ: Int {
        resamplingFnDataKwargs.orderZ
    }

    /// Whether separate-Z is forced
    public var forceSeparateZ: Bool? {
        resamplingFnDataKwargs.forceSeparateZ
    }
}

// MARK: - Supporting Types

/// Intensity properties for a single channel
public struct ChannelIntensityProperties: Sendable, Codable, Equatable {
    public let mean: Double
    public let std: Double
    public let percentile005: Double
    public let percentile995: Double

    enum CodingKeys: String, CodingKey {
        case mean
        case std
        case percentile005 = "percentile_00_5"
        case percentile995 = "percentile_99_5"
    }
}

/// Resampling function keyword arguments
public struct ResamplingKwargs: Sendable, Codable, Equatable {
    public let isSeg: Bool
    public let order: Int
    public let orderZ: Int
    public let forceSeparateZ: Bool?

    enum CodingKeys: String, CodingKey {
        case isSeg = "is_seg"
        case order
        case orderZ = "order_z"
        case forceSeparateZ = "force_separate_z"
    }
}
