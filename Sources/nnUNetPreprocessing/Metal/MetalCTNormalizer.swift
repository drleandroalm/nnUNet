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

        guard let library = PreprocessingShaderLibraryLoader.makeDefaultLibrary(on: device) else {
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
        guard let inputBuffer = volume.data.withUnsafeBytes({ ptr -> MTLBuffer? in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: bufferSize,
                options: .storageModeShared
            )
        }) else {
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

public enum MetalError: Error, Sendable {
    case failedToCreateCommandQueue
    case failedToLoadLibrary
    case failedToFindFunction(String)
    case failedToCreateBuffer
    case failedToCreateCommandBuffer
    case commandBufferFailed(String)
}
