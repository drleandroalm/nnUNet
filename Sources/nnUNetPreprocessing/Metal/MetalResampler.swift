// Sources/nnUNetPreprocessing/Metal/MetalResampler.swift

import Metal
import Foundation
import simd

/// GPU-accelerated volume resampling using Metal compute shaders
/// Implements cubic B-spline interpolation with separate-Z mode for anisotropic volumes
public actor MetalResampler {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let cubic3DPipeline: MTLComputePipelineState
    private let separateZXYPipeline: MTLComputePipelineState
    private let separateZZPipeline: MTLComputePipelineState

    /// Initialize with Metal device
    public init(device: MTLDevice) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.failedToCreateCommandQueue
        }
        self.commandQueue = queue

        guard let library = PreprocessingShaderLibraryLoader.makeDefaultLibrary(on: device) else {
            throw MetalError.failedToLoadLibrary
        }

        // Load all resampling kernels
        guard let cubic3DFunc = library.makeFunction(name: "resample_cubic_3d") else {
            throw MetalError.failedToFindFunction("resample_cubic_3d")
        }
        guard let separateZXYFunc = library.makeFunction(name: "resample_separate_z_xy") else {
            throw MetalError.failedToFindFunction("resample_separate_z_xy")
        }
        guard let separateZZFunc = library.makeFunction(name: "resample_separate_z_z") else {
            throw MetalError.failedToFindFunction("resample_separate_z_z")
        }

        self.cubic3DPipeline = try device.makeComputePipelineState(function: cubic3DFunc)
        self.separateZXYPipeline = try device.makeComputePipelineState(function: separateZXYFunc)
        self.separateZZPipeline = try device.makeComputePipelineState(function: separateZZFunc)
    }

    /// Resample volume to target spacing using GPU acceleration
    /// - Parameters:
    ///   - volume: Input volume buffer
    ///   - targetSpacing: Target spacing in mm (z, y, x)
    ///   - order: Interpolation order (3 = cubic) - currently only 3 is supported
    ///   - orderZ: Interpolation order for Z axis in separate-Z mode (0 = nearest, 1 = linear)
    ///   - forceSeparateZ: Override automatic separate-Z detection
    ///   - anisotropyThreshold: Threshold for automatic separate-Z (default 3.0)
    /// - Returns: Resampled volume buffer
    public func resample(
        _ volume: VolumeBuffer,
        targetSpacing: SIMD3<Double>,
        order: Int = 3,
        orderZ: Int = 0,
        forceSeparateZ: Bool? = nil,
        anisotropyThreshold: Double = 3.0
    ) async throws -> VolumeBuffer {
        let currentSpacing = volume.spacing

        // Compute target shape
        let scaleFactors = currentSpacing / targetSpacing
        let targetShape = (
            depth: Int((Double(volume.shape.depth) * scaleFactors.x).rounded(.toNearestOrEven)),
            height: Int((Double(volume.shape.height) * scaleFactors.y).rounded(.toNearestOrEven)),
            width: Int((Double(volume.shape.width) * scaleFactors.z).rounded(.toNearestOrEven))
        )

        // Determine if we should use separate-Z resampling
        let useSeparateZ: Bool
        if let force = forceSeparateZ {
            useSeparateZ = force
        } else {
            useSeparateZ = Resampling.shouldUseSeparateZ(
                spacing: currentSpacing,
                threshold: anisotropyThreshold
            )
        }

        if useSeparateZ {
            return try await resampleSeparateZ(
                volume,
                targetShape: targetShape,
                targetSpacing: targetSpacing,
                orderZ: orderZ
            )
        } else {
            return try await resampleCubic3D(
                volume,
                targetShape: targetShape,
                targetSpacing: targetSpacing
            )
        }
    }

    // MARK: - Private Implementation

    private func resampleCubic3D(
        _ volume: VolumeBuffer,
        targetShape: (depth: Int, height: Int, width: Int),
        targetSpacing: SIMD3<Double>
    ) async throws -> VolumeBuffer {
        let (srcD, srcH, srcW) = volume.shape
        let (dstD, dstH, dstW) = targetShape
        let srcBufferSize = volume.voxelCount * MemoryLayout<Float>.size
        let dstBufferSize = dstD * dstH * dstW * MemoryLayout<Float>.size

        // Create input buffer
        guard let inputBuffer = volume.data.withUnsafeBytes({ ptr -> MTLBuffer? in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: srcBufferSize,
                options: .storageModeShared
            )
        }) else {
            throw MetalError.failedToCreateBuffer
        }

        // Create output buffer
        guard let outputBuffer = device.makeBuffer(
            length: dstBufferSize,
            options: .storageModeShared
        ) else {
            throw MetalError.failedToCreateBuffer
        }

        // Create parameters
        var params = ResampleParams(
            srcDepth: UInt32(srcD),
            srcHeight: UInt32(srcH),
            srcWidth: UInt32(srcW),
            dstDepth: UInt32(dstD),
            dstHeight: UInt32(dstH),
            dstWidth: UInt32(dstW),
            scaleZ: Float(Double(srcD - 1) / Double(max(dstD - 1, 1))),
            scaleY: Float(Double(srcH - 1) / Double(max(dstH - 1, 1))),
            scaleX: Float(Double(srcW - 1) / Double(max(dstW - 1, 1))),
            orderZ: 0
        )

        // Create and execute command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.failedToCreateCommandBuffer
        }

        encoder.setComputePipelineState(cubic3DPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<ResampleParams>.size, index: 2)

        // 3D dispatch
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroups = MTLSize(
            width: (dstW + 7) / 8,
            height: (dstH + 7) / 8,
            depth: (dstD + 7) / 8
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: MetalError.commandBufferFailed(error.localizedDescription))
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }

        // Extract output data
        let outputData = Data(
            bytes: outputBuffer.contents(),
            count: dstBufferSize
        )

        return VolumeBuffer(
            data: outputData,
            shape: targetShape,
            spacing: targetSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }

    private func resampleSeparateZ(
        _ volume: VolumeBuffer,
        targetShape: (depth: Int, height: Int, width: Int),
        targetSpacing: SIMD3<Double>,
        orderZ: Int
    ) async throws -> VolumeBuffer {
        let (srcD, srcH, srcW) = volume.shape
        let (dstD, dstH, dstW) = targetShape
        let srcBufferSize = volume.voxelCount * MemoryLayout<Float>.size
        let intermediateBufferSize = srcD * dstH * dstW * MemoryLayout<Float>.size
        let dstBufferSize = dstD * dstH * dstW * MemoryLayout<Float>.size

        // Create buffers
        guard let inputBuffer = volume.data.withUnsafeBytes({ ptr -> MTLBuffer? in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: srcBufferSize,
                options: .storageModeShared
            )
        }) else {
            throw MetalError.failedToCreateBuffer
        }

        guard let intermediateBuffer = device.makeBuffer(
            length: intermediateBufferSize,
            options: .storageModeShared
        ) else {
            throw MetalError.failedToCreateBuffer
        }

        guard let outputBuffer = device.makeBuffer(
            length: dstBufferSize,
            options: .storageModeShared
        ) else {
            throw MetalError.failedToCreateBuffer
        }

        // Parameters for XY pass
        var paramsXY = ResampleParams(
            srcDepth: UInt32(srcD),
            srcHeight: UInt32(srcH),
            srcWidth: UInt32(srcW),
            dstDepth: UInt32(srcD),  // Keep source depth for intermediate
            dstHeight: UInt32(dstH),
            dstWidth: UInt32(dstW),
            scaleZ: 1.0,  // Not used in XY pass
            scaleY: Float(Double(srcH - 1) / Double(max(dstH - 1, 1))),
            scaleX: Float(Double(srcW - 1) / Double(max(dstW - 1, 1))),
            orderZ: UInt32(orderZ)
        )

        // Parameters for Z pass
        var paramsZ = ResampleParams(
            srcDepth: UInt32(srcD),
            srcHeight: UInt32(srcH),
            srcWidth: UInt32(srcW),
            dstDepth: UInt32(dstD),
            dstHeight: UInt32(dstH),
            dstWidth: UInt32(dstW),
            scaleZ: Float(Double(srcD - 1) / Double(max(dstD - 1, 1))),
            scaleY: 1.0,  // Not used in Z pass
            scaleX: 1.0,  // Not used in Z pass
            orderZ: UInt32(orderZ)
        )

        // Create command buffer with both passes
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalError.failedToCreateCommandBuffer
        }

        // First pass: XY resampling
        guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.failedToCreateCommandBuffer
        }

        encoder1.setComputePipelineState(separateZXYPipeline)
        encoder1.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder1.setBuffer(intermediateBuffer, offset: 0, index: 1)
        encoder1.setBytes(&paramsXY, length: MemoryLayout<ResampleParams>.size, index: 2)

        let threadgroupSizeXY = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroupsXY = MTLSize(
            width: (dstW + 7) / 8,
            height: (dstH + 7) / 8,
            depth: (srcD + 7) / 8
        )

        encoder1.dispatchThreadgroups(threadgroupsXY, threadsPerThreadgroup: threadgroupSizeXY)
        encoder1.endEncoding()

        // Second pass: Z resampling
        guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.failedToCreateCommandBuffer
        }

        encoder2.setComputePipelineState(separateZZPipeline)
        encoder2.setBuffer(intermediateBuffer, offset: 0, index: 0)
        encoder2.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder2.setBytes(&paramsZ, length: MemoryLayout<ResampleParams>.size, index: 2)

        let threadgroupSizeZ = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroupsZ = MTLSize(
            width: (dstW + 7) / 8,
            height: (dstH + 7) / 8,
            depth: (dstD + 7) / 8
        )

        encoder2.dispatchThreadgroups(threadgroupsZ, threadsPerThreadgroup: threadgroupSizeZ)
        encoder2.endEncoding()

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: MetalError.commandBufferFailed(error.localizedDescription))
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }

        // Extract output data
        let outputData = Data(
            bytes: outputBuffer.contents(),
            count: dstBufferSize
        )

        return VolumeBuffer(
            data: outputData,
            shape: targetShape,
            spacing: targetSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )
    }
}

// MARK: - Supporting Types

/// Parameters struct matching Metal shader
struct ResampleParams {
    var srcDepth: UInt32
    var srcHeight: UInt32
    var srcWidth: UInt32
    var dstDepth: UInt32
    var dstHeight: UInt32
    var dstWidth: UInt32
    var scaleZ: Float
    var scaleY: Float
    var scaleX: Float
    var orderZ: UInt32
}
