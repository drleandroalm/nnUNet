// Sources/nnUNetPreprocessing/CPU/CTNormalization.swift

import Foundation

/// CT normalization: clip to percentile bounds, then z-score normalize
public struct CTNormalization: Sendable {

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
