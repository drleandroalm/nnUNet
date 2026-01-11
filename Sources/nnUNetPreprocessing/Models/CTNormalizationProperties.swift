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
