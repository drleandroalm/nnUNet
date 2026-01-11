// Tests/nnUNetPreprocessingTests/MetalCTNormalizationTests.swift

import XCTest
@testable import nnUNetPreprocessing
import Metal
import simd

final class MetalCTNormalizationTests: XCTestCase {

    var device: MTLDevice?
    var normalizer: MetalCTNormalizer?

    override func setUp() async throws {
        try await super.setUp()

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available on this device")
        }
        self.device = device

        do {
            self.normalizer = try MetalCTNormalizer(device: device)
        } catch {
            throw XCTSkip("Failed to create MetalCTNormalizer: \(error)")
        }
    }

    func testMetalNormalizationMatchesCPU() async throws {
        guard let normalizer = self.normalizer else {
            throw XCTSkip("Metal normalizer not available")
        }

        // Create test volume with CT-like values
        let floats: [Float] = [-1024, -500, 0, 100, 500, 1000, 1500, 2000]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 2, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let props = CTNormalizationProperties(
            mean: 100.0,
            std: 200.0,
            lowerBound: -1024.0,
            upperBound: 1500.0
        )

        // Get CPU result
        let cpuResult = CTNormalization.apply(volume, properties: props)

        // Get Metal result
        let metalResult = try await normalizer.normalize(volume, properties: props)

        // Compare results
        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                for i in 0..<volume.voxelCount {
                    XCTAssertEqual(
                        cpuPtr[i],
                        metalPtr[i],
                        accuracy: 0.001,
                        "Mismatch at index \(i): CPU=\(cpuPtr[i]), Metal=\(metalPtr[i])"
                    )
                }
            }
        }
    }

    func testMetalNormalizationLargeVolume() async throws {
        guard let normalizer = self.normalizer else {
            throw XCTSkip("Metal normalizer not available")
        }

        // Create larger volume (64x64x64 = 262,144 voxels)
        let size = 64 * 64 * 64
        var floats = [Float](repeating: 0, count: size)
        for i in 0..<size {
            floats[i] = Float.random(in: -1024...2000)
        }

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 64, height: 64, width: 64),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let props = CTNormalizationProperties(
            mean: 100.0,
            std: 200.0,
            lowerBound: -1024.0,
            upperBound: 1500.0
        )

        // Get CPU result
        let cpuResult = CTNormalization.apply(volume, properties: props)

        // Get Metal result
        let metalResult = try await normalizer.normalize(volume, properties: props)

        // Sample comparison (checking every 1000th value for speed)
        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                var maxDiff: Float = 0
                for i in stride(from: 0, to: size, by: 1000) {
                    let diff = abs(cpuPtr[i] - metalPtr[i])
                    maxDiff = max(maxDiff, diff)
                }
                XCTAssertLessThan(maxDiff, 0.001, "Max difference exceeds tolerance")
            }
        }

        // Verify shape preserved
        XCTAssertEqual(metalResult.shape.depth, 64)
        XCTAssertEqual(metalResult.shape.height, 64)
        XCTAssertEqual(metalResult.shape.width, 64)
    }

    func testMetalNormalizationClipping() async throws {
        guard let normalizer = self.normalizer else {
            throw XCTSkip("Metal normalizer not available")
        }

        // Test that clipping works correctly
        let floats: [Float] = [-2000, -1024, 1500, 3000]  // Values outside bounds
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 1, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let props = CTNormalizationProperties(
            mean: 0.0,
            std: 100.0,
            lowerBound: -1024.0,
            upperBound: 1500.0
        )

        let result = try await normalizer.normalize(volume, properties: props)

        result.withUnsafeFloatPointer { ptr in
            // -2000 should be clipped to -1024, normalized: (-1024 - 0) / 100 = -10.24
            XCTAssertEqual(ptr[0], -10.24, accuracy: 0.01)

            // -1024 already at lower bound
            XCTAssertEqual(ptr[1], -10.24, accuracy: 0.01)

            // 1500 at upper bound
            XCTAssertEqual(ptr[2], 15.0, accuracy: 0.01)

            // 3000 should be clipped to 1500
            XCTAssertEqual(ptr[3], 15.0, accuracy: 0.01)
        }
    }
}
