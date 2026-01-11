// Tests/nnUNetPreprocessingTests/MetalResamplingTests.swift

import XCTest
import Metal
import simd
@testable import nnUNetPreprocessing

final class MetalResamplingTests: XCTestCase {

    var device: MTLDevice?
    var resampler: MetalResampler?

    override func setUp() async throws {
        try await super.setUp()

        device = MTLCreateSystemDefaultDevice()
        if let device = device {
            do {
                resampler = try MetalResampler(device: device)
            } catch {
                // Metal library loading may fail in CLI environment
                resampler = nil
            }
        }
    }

    // MARK: - Metal vs CPU Comparison Tests

    func testMetalResamplingMatchesCPU() async throws {
        guard let resampler = resampler else {
            throw XCTSkip("Metal not available")
        }

        // Create a small test volume (8×8×8)
        let size = 8
        let voxelCount = size * size * size
        var floats = [Float](repeating: 0, count: voxelCount)

        // Fill with gradient pattern for testing interpolation
        for z in 0..<size {
            for y in 0..<size {
                for x in 0..<size {
                    let idx = z * size * size + y * size + x
                    floats[idx] = Float(x + y * 10 + z * 100)
                }
            }
        }

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: size, height: size, width: size),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        // Target: 4×4×4 (downsampling)
        let targetSpacing = SIMD3<Double>(2.0, 2.0, 2.0)

        // CPU reference
        let cpuResult = Resampling.apply(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: false
        )

        // Metal implementation
        let metalResult = try await resampler.resample(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: false
        )

        // Verify shapes match
        XCTAssertEqual(cpuResult.shape.depth, metalResult.shape.depth)
        XCTAssertEqual(cpuResult.shape.height, metalResult.shape.height)
        XCTAssertEqual(cpuResult.shape.width, metalResult.shape.width)

        // Compare values
        var maxDiff: Float = 0
        var diffCount = 0

        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                for i in 0..<cpuResult.voxelCount {
                    let diff = abs(cpuPtr[i] - metalPtr[i])
                    if diff > 0.01 {
                        diffCount += 1
                    }
                    maxDiff = max(maxDiff, diff)
                }
            }
        }

        XCTAssertLessThan(maxDiff, 0.1, "Max difference exceeds tolerance: \(maxDiff)")
        XCTAssertEqual(diffCount, 0, "\(diffCount) values differ by more than 0.01")
    }

    func testMetalSeparateZMatchesCPU() async throws {
        guard let resampler = resampler else {
            throw XCTSkip("Metal not available")
        }

        // Create anisotropic volume (spacing ratio > 3.0)
        let size = 8
        let voxelCount = size * size * size
        var floats = [Float](repeating: 0, count: voxelCount)

        for z in 0..<size {
            for y in 0..<size {
                for x in 0..<size {
                    let idx = z * size * size + y * size + x
                    floats[idx] = Float(x + y * 10 + z * 100)
                }
            }
        }

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: size, height: size, width: size),
            spacing: SIMD3(4.0, 1.0, 1.0)  // Anisotropic: ratio = 4.0 > 3.0
        )

        let targetSpacing = SIMD3<Double>(2.0, 0.5, 0.5)

        // CPU reference with separate-Z
        let cpuResult = Resampling.apply(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: true
        )

        // Metal implementation with separate-Z
        let metalResult = try await resampler.resample(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: true
        )

        // Verify shapes match
        XCTAssertEqual(cpuResult.shape.depth, metalResult.shape.depth)
        XCTAssertEqual(cpuResult.shape.height, metalResult.shape.height)
        XCTAssertEqual(cpuResult.shape.width, metalResult.shape.width)

        // Compare values
        var maxDiff: Float = 0

        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                for i in 0..<cpuResult.voxelCount {
                    let diff = abs(cpuPtr[i] - metalPtr[i])
                    maxDiff = max(maxDiff, diff)
                }
            }
        }

        XCTAssertLessThan(maxDiff, 0.1, "Max difference exceeds tolerance: \(maxDiff)")
    }

    func testMetalResamplingLargeVolume() async throws {
        guard let resampler = resampler else {
            throw XCTSkip("Metal not available")
        }

        // Test with larger volume (32×32×32 = 32K voxels)
        let size = 32
        let voxelCount = size * size * size
        var floats = [Float](repeating: 0, count: voxelCount)

        // Fill with random-ish pattern
        for i in 0..<voxelCount {
            floats[i] = Float(i % 256) / 255.0
        }

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: size, height: size, width: size),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        // Upsample to 64×64×64
        let targetSpacing = SIMD3<Double>(0.5, 0.5, 0.5)

        // CPU reference
        let cpuResult = Resampling.apply(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: false
        )

        // Metal implementation
        let metalResult = try await resampler.resample(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: false
        )

        // Verify shapes
        XCTAssertEqual(cpuResult.shape.depth, metalResult.shape.depth)
        XCTAssertEqual(cpuResult.shape.height, metalResult.shape.height)
        XCTAssertEqual(cpuResult.shape.width, metalResult.shape.width)

        // Sample comparison (checking every 100th voxel for speed)
        var maxDiff: Float = 0

        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                for i in stride(from: 0, to: cpuResult.voxelCount, by: 100) {
                    let diff = abs(cpuPtr[i] - metalPtr[i])
                    maxDiff = max(maxDiff, diff)
                }
            }
        }

        XCTAssertLessThan(maxDiff, 0.1, "Max difference exceeds tolerance: \(maxDiff)")
    }

    func testMetalResamplingPreservesConstantVolume() async throws {
        guard let resampler = resampler else {
            throw XCTSkip("Metal not available")
        }

        // Create constant volume
        let size = 8
        let voxelCount = size * size * size
        let constantValue: Float = 42.5
        let floats = [Float](repeating: constantValue, count: voxelCount)

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: size, height: size, width: size),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        // Resample
        let targetSpacing = SIMD3<Double>(0.5, 0.5, 0.5)
        let result = try await resampler.resample(
            volume,
            targetSpacing: targetSpacing,
            order: 3,
            orderZ: 0,
            forceSeparateZ: false
        )

        // All values should still be approximately constant
        var maxDeviation: Float = 0

        result.withUnsafeFloatPointer { ptr in
            for i in 0..<result.voxelCount {
                let deviation = abs(ptr[i] - constantValue)
                maxDeviation = max(maxDeviation, deviation)
            }
        }

        XCTAssertLessThan(maxDeviation, 0.001,
            "Constant volume should preserve value, max deviation: \(maxDeviation)")
    }
}
