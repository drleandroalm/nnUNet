// Sources/nnUNetPreprocessing/CPU/CropToNonzero.swift

import Foundation

/// Crop volume to bounding box of non-zero voxels
public struct CropToNonzero: Sendable {

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
