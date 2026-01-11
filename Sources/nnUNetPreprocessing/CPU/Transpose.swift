// Sources/nnUNetPreprocessing/CPU/Transpose.swift

import Foundation

/// Axis transpose operation matching numpy.transpose
public struct Transpose: Sendable {

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

                let oldH = oldShape[1], oldW = oldShape[2]
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
