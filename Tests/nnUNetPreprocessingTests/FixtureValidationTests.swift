// Tests/nnUNetPreprocessingTests/FixtureValidationTests.swift

import XCTest
import Metal
import simd
@testable import nnUNetPreprocessing

/// Tests that validate Swift preprocessing implementation against Python-generated fixtures
final class FixtureValidationTests: XCTestCase {

    var params: PreprocessingParameters!
    var metadata: FixtureMetadata!

    override func setUp() async throws {
        try await super.setUp()

        do {
            params = try FixtureLoader.loadParams()
            metadata = try FixtureLoader.loadMetadata()
        } catch {
            throw XCTSkip("Fixtures not available: \(error)")
        }
    }

    // MARK: - Fixture Loading Tests

    func testFixturesLoadSuccessfully() throws {
        // Verify all fixtures can be loaded
        let stages = ["01_raw", "02_transposed", "03_cropped", "04_normalized", "05_resampled"]

        for stage in stages {
            let (data, shape) = try FixtureLoader.loadNpy(stage)
            XCTAssertFalse(data.isEmpty, "\(stage) should have data")
            XCTAssertFalse(shape.isEmpty, "\(stage) should have shape")

            let expectedCount = shape.reduce(1, *)
            XCTAssertEqual(data.count, expectedCount,
                "\(stage) data count \(data.count) should match shape product \(expectedCount)")
        }
    }

    func testParamsMatchMetadata() throws {
        XCTAssertEqual(params.configurationName, metadata.configuration)
    }

    // MARK: - Transpose Tests

    func testTransposeMatchesPythonFixture() throws {
        let (rawData, rawShape) = try FixtureLoader.loadNpy("01_raw")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("02_transposed")

        // Create input volume
        let input = VolumeBuffer(
            data: rawData.withUnsafeBytes { Data($0) },
            shape: (depth: rawShape[0], height: rawShape[1], width: rawShape[2]),
            spacing: SIMD3(
                metadata.stages["01_raw"]?.spacing?[0] ?? 1.0,
                metadata.stages["01_raw"]?.spacing?[1] ?? 1.0,
                metadata.stages["01_raw"]?.spacing?[2] ?? 1.0
            )
        )

        // Apply transpose
        let result = Transpose.apply(input, axes: params.transposeForward)

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0], "Depth mismatch")
        XCTAssertEqual(result.shape.height, expectedShape[1], "Height mismatch")
        XCTAssertEqual(result.shape.width, expectedShape[2], "Width mismatch")

        // Verify data (exact match expected for transpose)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.0)
        }
    }

    // MARK: - Crop Tests

    func testCropToNonzeroMatchesPythonFixture() throws {
        let (transposedData, transposedShape) = try FixtureLoader.loadNpy("02_transposed")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("03_cropped")

        let input = VolumeBuffer(
            data: transposedData.withUnsafeBytes { Data($0) },
            shape: (depth: transposedShape[0], height: transposedShape[1], width: transposedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        let (result, bbox) = CropToNonzero.apply(input)

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0], "Depth mismatch")
        XCTAssertEqual(result.shape.height, expectedShape[1], "Height mismatch")
        XCTAssertEqual(result.shape.width, expectedShape[2], "Width mismatch")

        // Verify bbox matches metadata
        if let expectedBbox = metadata.stages["03_cropped"]?.bbox {
            XCTAssertEqual(bbox.start.z, expectedBbox[0][0], "BBox start Z mismatch")
            XCTAssertEqual(bbox.end.z, expectedBbox[0][1], "BBox end Z mismatch")
            XCTAssertEqual(bbox.start.y, expectedBbox[1][0], "BBox start Y mismatch")
            XCTAssertEqual(bbox.end.y, expectedBbox[1][1], "BBox end Y mismatch")
            XCTAssertEqual(bbox.start.x, expectedBbox[2][0], "BBox start X mismatch")
            XCTAssertEqual(bbox.end.x, expectedBbox[2][1], "BBox end X mismatch")
        }

        // Verify data (exact match expected)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.0)
        }
    }

    // MARK: - Normalization Tests

    func testCTNormalizationMatchesPythonFixture() throws {
        let (croppedData, croppedShape) = try FixtureLoader.loadNpy("03_cropped")
        let (expectedData, _) = try FixtureLoader.loadNpy("04_normalized")

        guard let ctProps = params.ctNormalizationProperties else {
            throw XCTSkip("CT normalization properties not available")
        }

        let input = VolumeBuffer(
            data: croppedData.withUnsafeBytes { Data($0) },
            shape: (depth: croppedShape[0], height: croppedShape[1], width: croppedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        let result = CTNormalization.apply(input, properties: ctProps)

        // Verify data (small tolerance for floating point)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            let mae = ArrayComparison.meanAbsoluteError(actual, expectedData)
            XCTAssertLessThan(mae, 0.01, "MAE should be < 0.01, got \(mae)")
            ArrayComparison.assertEqual(actual, expectedData, tolerance: 0.01)
        }
    }

    // MARK: - Resampling Tests

    func testResamplingMatchesPythonFixture() throws {
        let (normalizedData, normalizedShape) = try FixtureLoader.loadNpy("04_normalized")
        let (expectedData, expectedShape) = try FixtureLoader.loadNpy("05_resampled")

        // Get spacing from metadata
        guard let originalSpacing = metadata.stages["04_normalized"]?.spacing,
              let resampledMeta = metadata.stages["05_resampled"],
              let targetSpacing = resampledMeta.targetSpacing else {
            throw XCTSkip("Spacing metadata not available")
        }

        let input = VolumeBuffer(
            data: normalizedData.withUnsafeBytes { Data($0) },
            shape: (depth: normalizedShape[0], height: normalizedShape[1], width: normalizedShape[2]),
            spacing: SIMD3(originalSpacing[0], originalSpacing[1], originalSpacing[2])
        )

        let result = Resampling.apply(
            input,
            targetSpacing: SIMD3(targetSpacing[0], targetSpacing[1], targetSpacing[2]),
            order: params.resamplingOrder,
            orderZ: params.resamplingOrderZ,
            forceSeparateZ: params.forceSeparateZ,
            anisotropyThreshold: params.anisotropyThreshold
        )

        // Verify shape
        XCTAssertEqual(result.shape.depth, expectedShape[0], "Depth mismatch: got \(result.shape.depth), expected \(expectedShape[0])")
        XCTAssertEqual(result.shape.height, expectedShape[1], "Height mismatch: got \(result.shape.height), expected \(expectedShape[1])")
        XCTAssertEqual(result.shape.width, expectedShape[2], "Width mismatch: got \(result.shape.width), expected \(expectedShape[2])")

        // Verify data (higher tolerance for resampling due to interpolation algorithm differences)
        result.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: result.voxelCount))
            let mae = ArrayComparison.meanAbsoluteError(actual, expectedData)
            let maxErr = ArrayComparison.maxAbsoluteError(actual, expectedData)

            // Report statistics
            print("Resampling comparison:")
            print("  MAE: \(mae)")
            print("  Max Error: \(maxErr)")

            // Resampling can have higher differences due to different cubic implementations
            XCTAssertLessThan(mae, 0.5, "MAE should be < 0.5, got \(mae)")
        }
    }

    // MARK: - End-to-End Pipeline Test

    func testFullPipelineMatchesPythonOutput() throws {
        // Load raw input
        let (rawData, rawShape) = try FixtureLoader.loadNpy("01_raw")
        let (expectedFinal, expectedFinalShape) = try FixtureLoader.loadNpy("05_resampled")

        guard let ctProps = params.ctNormalizationProperties,
              let originalSpacing = metadata.stages["01_raw"]?.spacing,
              let resampledMeta = metadata.stages["05_resampled"],
              let targetSpacing = resampledMeta.targetSpacing else {
            throw XCTSkip("Required metadata not available")
        }

        // Create initial volume
        var volume = VolumeBuffer(
            data: rawData.withUnsafeBytes { Data($0) },
            shape: (depth: rawShape[0], height: rawShape[1], width: rawShape[2]),
            spacing: SIMD3(originalSpacing[0], originalSpacing[1], originalSpacing[2])
        )

        // Step 1: Transpose
        volume = Transpose.apply(volume, axes: params.transposeForward)

        // Update spacing after transpose
        let transposedSpacing = SIMD3(
            originalSpacing[params.transposeForward[0]],
            originalSpacing[params.transposeForward[1]],
            originalSpacing[params.transposeForward[2]]
        )
        volume = VolumeBuffer(
            data: volume.data,
            shape: volume.shape,
            spacing: transposedSpacing,
            origin: volume.origin,
            orientation: volume.orientation,
            bbox: volume.bbox
        )

        // Step 2: Crop to nonzero
        let (croppedVolume, _) = CropToNonzero.apply(volume)
        volume = croppedVolume

        // Step 3: Normalize
        volume = CTNormalization.apply(volume, properties: ctProps)

        // Step 4: Resample
        volume = Resampling.apply(
            volume,
            targetSpacing: SIMD3(targetSpacing[0], targetSpacing[1], targetSpacing[2]),
            order: params.resamplingOrder,
            orderZ: params.resamplingOrderZ,
            forceSeparateZ: params.forceSeparateZ,
            anisotropyThreshold: params.anisotropyThreshold
        )

        // Verify final shape matches expected
        XCTAssertEqual(volume.shape.depth, expectedFinalShape[0], "Final depth mismatch")
        XCTAssertEqual(volume.shape.height, expectedFinalShape[1], "Final height mismatch")
        XCTAssertEqual(volume.shape.width, expectedFinalShape[2], "Final width mismatch")

        // Verify data matches within tolerance
        volume.withUnsafeFloatPointer { ptr in
            let actual = Array(UnsafeBufferPointer(start: ptr, count: volume.voxelCount))
            let mae = ArrayComparison.meanAbsoluteError(actual, expectedFinal)
            let maxErr = ArrayComparison.maxAbsoluteError(actual, expectedFinal)

            print("End-to-end pipeline comparison:")
            print("  MAE: \(mae)")
            print("  Max Error: \(maxErr)")

            XCTAssertLessThan(mae, 0.5, "End-to-end MAE should be < 0.5, got \(mae)")
        }
    }

    // MARK: - Metal vs CPU Comparison

    func testMetalNormalizationMatchesCPU() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }

        let (croppedData, croppedShape) = try FixtureLoader.loadNpy("03_cropped")

        guard let ctProps = params.ctNormalizationProperties else {
            throw XCTSkip("CT normalization properties not available")
        }

        let input = VolumeBuffer(
            data: croppedData.withUnsafeBytes { Data($0) },
            shape: (depth: croppedShape[0], height: croppedShape[1], width: croppedShape[2]),
            spacing: SIMD3(1, 1, 1)
        )

        // CPU reference
        let cpuResult = CTNormalization.apply(input, properties: ctProps)

        // Metal implementation
        let metalNormalizer: MetalCTNormalizer
        do {
            metalNormalizer = try MetalCTNormalizer(device: device)
        } catch {
            throw XCTSkip("Metal normalizer unavailable: \(error)")
        }

        let metalResult = try await metalNormalizer.normalize(input, properties: ctProps)

        // Compare
        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                let cpu = Array(UnsafeBufferPointer(start: cpuPtr, count: cpuResult.voxelCount))
                let metal = Array(UnsafeBufferPointer(start: metalPtr, count: metalResult.voxelCount))

                let mae = ArrayComparison.meanAbsoluteError(cpu, metal)
                XCTAssertLessThan(mae, 0.001, "Metal should match CPU within 0.001, got \(mae)")
            }
        }
    }

    func testMetalResamplingMatchesCPU() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }

        let (normalizedData, normalizedShape) = try FixtureLoader.loadNpy("04_normalized")

        guard let originalSpacing = metadata.stages["04_normalized"]?.spacing,
              let resampledMeta = metadata.stages["05_resampled"],
              let targetSpacing = resampledMeta.targetSpacing else {
            throw XCTSkip("Spacing metadata not available")
        }

        let input = VolumeBuffer(
            data: normalizedData.withUnsafeBytes { Data($0) },
            shape: (depth: normalizedShape[0], height: normalizedShape[1], width: normalizedShape[2]),
            spacing: SIMD3(originalSpacing[0], originalSpacing[1], originalSpacing[2])
        )

        let targetSpacingVec = SIMD3(targetSpacing[0], targetSpacing[1], targetSpacing[2])

        // CPU reference
        let cpuResult = Resampling.apply(
            input,
            targetSpacing: targetSpacingVec,
            order: params.resamplingOrder,
            orderZ: params.resamplingOrderZ,
            forceSeparateZ: params.forceSeparateZ,
            anisotropyThreshold: params.anisotropyThreshold
        )

        // Metal implementation
        let metalResampler: MetalResampler
        do {
            metalResampler = try MetalResampler(device: device)
        } catch {
            throw XCTSkip("Metal resampler unavailable: \(error)")
        }

        let metalResult = try await metalResampler.resample(
            input,
            targetSpacing: targetSpacingVec,
            order: params.resamplingOrder,
            orderZ: params.resamplingOrderZ,
            forceSeparateZ: params.forceSeparateZ,
            anisotropyThreshold: params.anisotropyThreshold
        )

        // Compare
        XCTAssertEqual(cpuResult.shape.depth, metalResult.shape.depth, "Shape depth mismatch")
        XCTAssertEqual(cpuResult.shape.height, metalResult.shape.height, "Shape height mismatch")
        XCTAssertEqual(cpuResult.shape.width, metalResult.shape.width, "Shape width mismatch")

        cpuResult.withUnsafeFloatPointer { cpuPtr in
            metalResult.withUnsafeFloatPointer { metalPtr in
                let cpu = Array(UnsafeBufferPointer(start: cpuPtr, count: cpuResult.voxelCount))
                let metal = Array(UnsafeBufferPointer(start: metalPtr, count: metalResult.voxelCount))

                let mae = ArrayComparison.meanAbsoluteError(cpu, metal)
                let maxErr = ArrayComparison.maxAbsoluteError(cpu, metal)

                print("Metal vs CPU resampling:")
                print("  MAE: \(mae)")
                print("  Max Error: \(maxErr)")

                // Metal and CPU cubic interpolation may differ slightly
                XCTAssertLessThan(mae, 0.1, "Metal should match CPU within 0.1, got \(mae)")
            }
        }
    }
}
