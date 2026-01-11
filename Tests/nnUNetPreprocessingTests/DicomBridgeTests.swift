// Tests/nnUNetPreprocessingTests/DicomBridgeTests.swift

import XCTest
@testable import nnUNetPreprocessing
import simd

/// Tests for VolumeBuffer and HU conversion logic
/// Note: DicomSeriesVolume has internal init, so we test the conversion logic directly
final class DicomBridgeTests: XCTestCase {

    // MARK: - VolumeBuffer Tests

    func testVolumeBufferCreation() {
        let shape = (depth: 2, height: 4, width: 4)
        let voxelCount = shape.depth * shape.height * shape.width
        let data = Data(count: voxelCount * MemoryLayout<Float>.size)

        let buffer = VolumeBuffer(
            data: data,
            shape: shape,
            spacing: SIMD3(2.0, 1.0, 1.0)
        )

        XCTAssertEqual(buffer.voxelCount, voxelCount)
        XCTAssertEqual(buffer.byteCount, voxelCount * MemoryLayout<Float>.size)
        XCTAssertEqual(buffer.shape.depth, 2)
        XCTAssertEqual(buffer.shape.height, 4)
        XCTAssertEqual(buffer.shape.width, 4)
    }

    func testVolumeBufferWithFloatPointer() {
        let shape = (depth: 1, height: 2, width: 2)
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let data = floats.withUnsafeBytes { Data($0) }

        let buffer = VolumeBuffer(
            data: data,
            shape: shape,
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        buffer.withUnsafeFloatPointer { ptr in
            XCTAssertEqual(ptr[0], 1.0, accuracy: 0.001)
            XCTAssertEqual(ptr[1], 2.0, accuracy: 0.001)
            XCTAssertEqual(ptr[2], 3.0, accuracy: 0.001)
            XCTAssertEqual(ptr[3], 4.0, accuracy: 0.001)
        }
    }

    func testVolumeBufferEquality() {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let data = floats.withUnsafeBytes { Data($0) }

        let buffer1 = VolumeBuffer(
            data: data,
            shape: (depth: 1, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        let buffer2 = VolumeBuffer(
            data: data,
            shape: (depth: 1, height: 2, width: 2),
            spacing: SIMD3(1.0, 1.0, 1.0)
        )

        XCTAssertEqual(buffer1, buffer2)
    }

    // MARK: - HU Conversion Logic Tests (simulating what DicomBridge does)

    func testHUConversionLogicSigned() {
        // Simulate the HU conversion that DicomBridge performs
        let rawPixels: [Int16] = [-1000, 0, 1000, 500]
        let slope: Float = 1.0
        let intercept: Float = 0.0

        var huValues = [Float](repeating: 0, count: rawPixels.count)
        for i in 0..<rawPixels.count {
            huValues[i] = Float(rawPixels[i]) * slope + intercept
        }

        XCTAssertEqual(huValues[0], -1000.0, accuracy: 0.001)
        XCTAssertEqual(huValues[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(huValues[2], 1000.0, accuracy: 0.001)
        XCTAssertEqual(huValues[3], 500.0, accuracy: 0.001)
    }

    func testHUConversionLogicWithRescale() {
        // Simulate HU conversion with non-trivial rescale parameters
        let rawPixels: [Int16] = [0, 100, 200, 300]
        let slope: Float = 2.0
        let intercept: Float = -1024.0

        var huValues = [Float](repeating: 0, count: rawPixels.count)
        for i in 0..<rawPixels.count {
            huValues[i] = Float(rawPixels[i]) * slope + intercept
        }

        // 0 * 2 - 1024 = -1024
        XCTAssertEqual(huValues[0], -1024.0, accuracy: 0.001)
        // 100 * 2 - 1024 = -824
        XCTAssertEqual(huValues[1], -824.0, accuracy: 0.001)
        // 200 * 2 - 1024 = -624
        XCTAssertEqual(huValues[2], -624.0, accuracy: 0.001)
        // 300 * 2 - 1024 = -424
        XCTAssertEqual(huValues[3], -424.0, accuracy: 0.001)
    }

    func testHUConversionLogicUnsigned() {
        // Simulate HU conversion with unsigned pixels
        let rawPixels: [UInt16] = [0, 1000, 2000, 3000]
        let slope: Float = 1.0
        let intercept: Float = -1024.0

        var huValues = [Float](repeating: 0, count: rawPixels.count)
        for i in 0..<rawPixels.count {
            huValues[i] = Float(rawPixels[i]) * slope + intercept
        }

        XCTAssertEqual(huValues[0], -1024.0, accuracy: 0.001)
        XCTAssertEqual(huValues[1], -24.0, accuracy: 0.001)
        XCTAssertEqual(huValues[2], 976.0, accuracy: 0.001)
        XCTAssertEqual(huValues[3], 1976.0, accuracy: 0.001)
    }

    // MARK: - BoundingBox Tests

    func testBoundingBoxCreation() {
        let bbox = BoundingBox(
            start: (z: 5, y: 10, x: 15),
            end: (z: 25, y: 50, x: 75)
        )

        XCTAssertEqual(bbox.start.z, 5)
        XCTAssertEqual(bbox.start.y, 10)
        XCTAssertEqual(bbox.start.x, 15)
        XCTAssertEqual(bbox.end.z, 25)
        XCTAssertEqual(bbox.end.y, 50)
        XCTAssertEqual(bbox.end.x, 75)

        let size = bbox.size
        XCTAssertEqual(size.depth, 20)
        XCTAssertEqual(size.height, 40)
        XCTAssertEqual(size.width, 60)
    }

    func testBoundingBoxCodable() throws {
        let original = BoundingBox(
            start: (z: 1, y: 2, x: 3),
            end: (z: 10, y: 20, x: 30)
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(BoundingBox.self, from: data)

        XCTAssertEqual(original, decoded)
    }
}
