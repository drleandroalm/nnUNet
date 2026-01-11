// Tests/nnUNetPreprocessingTests/CPUPreprocessingTests.swift

import XCTest
@testable import nnUNetPreprocessing
import simd

final class CPUPreprocessingTests: XCTestCase {

    // MARK: - Transpose Tests

    func testTransposeIdentity() {
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 2, height: 2, width: 2),
            spacing: SIMD3(1.0, 2.0, 3.0)
        )

        let result = Transpose.apply(volume, axes: [0, 1, 2])

        // Identity transpose should not change data
        XCTAssertEqual(result.shape.depth, 2)
        XCTAssertEqual(result.shape.height, 2)
        XCTAssertEqual(result.shape.width, 2)
        XCTAssertEqual(result.data, volume.data)
    }

    func testTransposeReverse() {
        // 2x2x3 volume
        let floats: [Float] = [
            // z=0, y=0
            1, 2, 3,
            // z=0, y=1
            4, 5, 6,
            // z=1, y=0
            7, 8, 9,
            // z=1, y=1
            10, 11, 12
        ]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 2, height: 2, width: 3),
            spacing: SIMD3(1.0, 2.0, 3.0)
        )

        // Reverse axes: [2, 1, 0] means new_d=old_w, new_h=old_h, new_w=old_d
        let result = Transpose.apply(volume, axes: [2, 1, 0])

        XCTAssertEqual(result.shape.depth, 3)  // was width
        XCTAssertEqual(result.shape.height, 2) // unchanged
        XCTAssertEqual(result.shape.width, 2)  // was depth

        // Spacing should also be transposed
        XCTAssertEqual(result.spacing.x, 3.0, accuracy: 0.001) // was z
        XCTAssertEqual(result.spacing.y, 2.0, accuracy: 0.001) // unchanged
        XCTAssertEqual(result.spacing.z, 1.0, accuracy: 0.001) // was x
    }

    // MARK: - CropToNonzero Tests

    func testCropToNonzeroWithPadding() {
        // 4x4x4 volume with non-zero values in center 2x2x2
        var floats = [Float](repeating: 0, count: 64)
        // Set center values (indices 21, 22, 25, 26, 37, 38, 41, 42)
        // z=1, y=1, x=1..2; z=1, y=2, x=1..2; z=2, y=1, x=1..2; z=2, y=2, x=1..2
        for z in 1...2 {
            for y in 1...2 {
                for x in 1...2 {
                    let idx = z * 16 + y * 4 + x
                    floats[idx] = Float(idx)
                }
            }
        }

        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 4, height: 4, width: 4),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let (result, bbox) = CropToNonzero.apply(volume)

        // Should be cropped to 2x2x2
        XCTAssertEqual(result.shape.depth, 2)
        XCTAssertEqual(result.shape.height, 2)
        XCTAssertEqual(result.shape.width, 2)

        // Bounding box should be [1:3, 1:3, 1:3]
        XCTAssertEqual(bbox.start.z, 1)
        XCTAssertEqual(bbox.start.y, 1)
        XCTAssertEqual(bbox.start.x, 1)
        XCTAssertEqual(bbox.end.z, 3)
        XCTAssertEqual(bbox.end.y, 3)
        XCTAssertEqual(bbox.end.x, 3)
    }

    func testCropToNonzeroAllNonzero() {
        // All values are non-zero, should return same volume
        let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 2, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let (result, bbox) = CropToNonzero.apply(volume)

        XCTAssertEqual(result.shape.depth, 2)
        XCTAssertEqual(result.shape.height, 2)
        XCTAssertEqual(result.shape.width, 2)
        XCTAssertEqual(bbox.start.z, 0)
        XCTAssertEqual(bbox.start.y, 0)
        XCTAssertEqual(bbox.start.x, 0)
        XCTAssertEqual(bbox.end.z, 2)
        XCTAssertEqual(bbox.end.y, 2)
        XCTAssertEqual(bbox.end.x, 2)
    }

    // MARK: - CTNormalization Tests

    func testCTNormalization() {
        // Values: -1024 (air), 0 (water), 500 (soft tissue), 2000 (bone)
        let floats: [Float] = [-1024, 0, 500, 2000]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 1, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let props = CTNormalizationProperties(
            mean: 100.0,
            std: 200.0,
            lowerBound: -1024.0,
            upperBound: 1500.0
        )

        let result = CTNormalization.apply(volume, properties: props)

        result.withUnsafeFloatPointer { ptr in
            // -1024 clipped to -1024, normalized: (-1024 - 100) / 200 = -5.62
            XCTAssertEqual(ptr[0], (-1024.0 - 100.0) / 200.0, accuracy: 0.01)

            // 0 normalized: (0 - 100) / 200 = -0.5
            XCTAssertEqual(ptr[1], (0.0 - 100.0) / 200.0, accuracy: 0.01)

            // 500 normalized: (500 - 100) / 200 = 2.0
            XCTAssertEqual(ptr[2], (500.0 - 100.0) / 200.0, accuracy: 0.01)

            // 2000 clipped to 1500, normalized: (1500 - 100) / 200 = 7.0
            XCTAssertEqual(ptr[3], (1500.0 - 100.0) / 200.0, accuracy: 0.01)
        }
    }

    // MARK: - Resampling Tests

    func testResamplingAnisotropyDetection() {
        // Isotropic spacing: ratio = 1.0, should NOT use separate-Z
        let isotropic = SIMD3<Double>(1.0, 1.0, 1.0)
        XCTAssertFalse(Resampling.shouldUseSeparateZ(spacing: isotropic, threshold: 3.0))

        // Slightly anisotropic: ratio = 2.0, should NOT use separate-Z
        let slightlyAniso = SIMD3<Double>(2.0, 1.0, 1.0)
        XCTAssertFalse(Resampling.shouldUseSeparateZ(spacing: slightlyAniso, threshold: 3.0))

        // Very anisotropic: ratio = 5.0, SHOULD use separate-Z
        let veryAniso = SIMD3<Double>(5.0, 1.0, 1.0)
        XCTAssertTrue(Resampling.shouldUseSeparateZ(spacing: veryAniso, threshold: 3.0))
    }

    func testResamplingDownsample() {
        // 4x4x4 volume with constant value
        let floats = [Float](repeating: 1.0, count: 64)
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 4, height: 4, width: 4),
            spacing: SIMD3(0.5, 0.5, 0.5)  // 0.5mm spacing
        )

        // Resample to 1mm spacing (should halve dimensions)
        let result = Resampling.apply(
            volume,
            targetSpacing: SIMD3(1.0, 1.0, 1.0),
            forceSeparateZ: false
        )

        XCTAssertEqual(result.shape.depth, 2)
        XCTAssertEqual(result.shape.height, 2)
        XCTAssertEqual(result.shape.width, 2)

        // All values should still be ~1.0
        result.withUnsafeFloatPointer { ptr in
            for i in 0..<result.voxelCount {
                XCTAssertEqual(ptr[i], 1.0, accuracy: 0.1)
            }
        }
    }

    func testResamplingUpsample() {
        // 2x2x2 volume
        let floats: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 2, height: 2, width: 2),
            spacing: SIMD3(2.0, 2.0, 2.0)  // 2mm spacing
        )

        // Resample to 1mm spacing (should double dimensions)
        let result = Resampling.apply(
            volume,
            targetSpacing: SIMD3(1.0, 1.0, 1.0),
            forceSeparateZ: false
        )

        XCTAssertEqual(result.shape.depth, 4)
        XCTAssertEqual(result.shape.height, 4)
        XCTAssertEqual(result.shape.width, 4)
    }

    func testResamplingSeparateZ() {
        // Create anisotropic volume
        let floats = [Float](repeating: 1.0, count: 64)
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: 4, height: 4, width: 4),
            spacing: SIMD3(5.0, 1.0, 1.0)  // Very anisotropic Z
        )

        // Force separate-Z resampling
        let result = Resampling.apply(
            volume,
            targetSpacing: SIMD3(2.5, 0.5, 0.5),
            orderZ: 0,
            forceSeparateZ: true
        )

        // Check output shape
        // Z: 4 * (5.0 / 2.5) = 8
        // Y: 4 * (1.0 / 0.5) = 8
        // X: 4 * (1.0 / 0.5) = 8
        XCTAssertEqual(result.shape.depth, 8)
        XCTAssertEqual(result.shape.height, 8)
        XCTAssertEqual(result.shape.width, 8)
    }

    func testResamplingTargetShapeUsesBankersRounding() {
        let size = 5
        let voxelCount = size * size * size
        let floats = [Float](repeating: 1.0, count: voxelCount)
        let data = floats.withUnsafeBytes { Data($0) }
        let volume = VolumeBuffer(
            data: data,
            shape: (depth: size, height: size, width: size),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let result = Resampling.apply(
            volume,
            targetSpacing: SIMD3(2.0, 2.0, 2.0),
            forceSeparateZ: false
        )

        XCTAssertEqual(result.shape.depth, 2)
        XCTAssertEqual(result.shape.height, 2)
        XCTAssertEqual(result.shape.width, 2)
    }
}
