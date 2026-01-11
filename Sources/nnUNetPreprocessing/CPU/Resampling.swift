// Sources/nnUNetPreprocessing/CPU/Resampling.swift

import Foundation
import simd

/// Resampling to target spacing matching nnUNet's resample_data_or_seg_to_shape
public struct Resampling: Sendable {

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
                                src: src.baseAddress!,
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
                                src: src.baseAddress!,
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
