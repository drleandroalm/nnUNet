// Tests/nnUNetPreprocessingTests/Helpers/FixtureLoader.swift

import Foundation
@testable import nnUNetPreprocessing

/// Helper for loading numpy fixtures in tests
public struct FixtureLoader {

    /// Load a .npy file as float array with shape
    public static func loadNpy(_ name: String) throws -> (data: [Float], shape: [Int]) {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy") else {
            throw FixtureError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        return try parseNpy(data)
    }

    /// Load preprocessing parameters
    public static func loadParams() throws -> PreprocessingParameters {
        guard let url = Bundle.module.url(
            forResource: "preprocessing_params",
            withExtension: "json"
        ) else {
            throw FixtureError.fileNotFound("preprocessing_params.json")
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PreprocessingParameters.self, from: data)
    }

    /// Load fixture metadata
    public static func loadMetadata() throws -> FixtureMetadata {
        guard let url = Bundle.module.url(
            forResource: "fixture_metadata",
            withExtension: "json"
        ) else {
            throw FixtureError.fileNotFound("fixture_metadata.json")
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(FixtureMetadata.self, from: data)
    }

    // MARK: - NPY Parser

    private static func parseNpy(_ data: Data) throws -> (data: [Float], shape: [Int]) {
        // Simple .npy parser for float32 arrays
        // Format: magic (6 bytes) + version (2 bytes) + header_len (2 bytes) + header + data

        guard data.count > 10 else {
            throw FixtureError.invalidFormat("File too small")
        }

        // Check magic number: \x93NUMPY
        let magic = data.prefix(6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw FixtureError.invalidFormat("Invalid magic number")
        }

        // Get version
        let majorVersion = data[6]
        let minorVersion = data[7]

        // Get header length (depends on version)
        let headerLen: Int
        let headerStart: Int

        if majorVersion == 1 {
            // Version 1.0: 2-byte header length
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else if majorVersion == 2 {
            // Version 2.0: 4-byte header length
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        } else {
            throw FixtureError.invalidFormat("Unsupported NPY version \(majorVersion).\(minorVersion)")
        }

        // Parse header
        let headerEnd = headerStart + Int(headerLen)
        guard headerEnd <= data.count else {
            throw FixtureError.invalidFormat("Header extends beyond file")
        }

        guard let headerString = String(data: data.subdata(in: headerStart..<headerEnd), encoding: .ascii) else {
            throw FixtureError.invalidFormat("Cannot decode header as ASCII")
        }

        // Extract shape from header
        let shape = parseShape(from: headerString)
        guard !shape.isEmpty else {
            throw FixtureError.invalidFormat("Could not parse shape from header: \(headerString)")
        }

        // Verify dtype is float32
        guard headerString.contains("'<f4'") || headerString.contains("'float32'") else {
            throw FixtureError.invalidFormat("Expected float32 dtype, got header: \(headerString)")
        }

        // Extract float32 data
        let dataStart = headerEnd
        let floatData = data.subdata(in: dataStart..<data.count)
        let expectedFloatCount = shape.reduce(1, *)
        let expectedByteCount = expectedFloatCount * MemoryLayout<Float>.size

        guard floatData.count >= expectedByteCount else {
            throw FixtureError.invalidFormat("Data size mismatch: expected \(expectedByteCount) bytes, got \(floatData.count)")
        }

        let floats = floatData.withUnsafeBytes { buffer -> [Float] in
            Array(buffer.bindMemory(to: Float.self).prefix(expectedFloatCount))
        }

        return (floats, shape)
    }

    private static func parseShape(from header: String) -> [Int] {
        // Extract shape tuple from numpy header
        // Format: {'descr': '<f4', 'fortran_order': False, 'shape': (32, 64, 64), }

        // Find 'shape': (
        guard let shapeStart = header.range(of: "'shape': (") else {
            return []
        }

        // Find closing )
        guard let shapeEnd = header.range(of: ")", range: shapeStart.upperBound..<header.endIndex) else {
            return []
        }

        let shapeString = header[shapeStart.upperBound..<shapeEnd.lowerBound]

        // Parse comma-separated integers
        // Handle both "32, 64, 64" and "32," (single-element tuples)
        return shapeString
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
}

// MARK: - Error Types

public enum FixtureError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidFormat(String)

    public var description: String {
        switch self {
        case .fileNotFound(let name):
            return "Fixture file not found: \(name)"
        case .invalidFormat(let reason):
            return "Invalid fixture format: \(reason)"
        }
    }
}

// MARK: - Metadata Types

public struct FixtureMetadata: Codable {
    public let inputFile: String
    public let configuration: String
    public let stages: [String: StageMetadata]
    public let checksums: [String: String]

    enum CodingKeys: String, CodingKey {
        case inputFile = "input_file"
        case configuration
        case stages
        case checksums
    }
}

public struct StageMetadata: Codable {
    public let shape: [Int]?
    public let spacing: [Double]?
    public let bbox: [[Int]]?
    public let mean: Double?
    public let std: Double?
    public let min: Double?
    public let max: Double?
    public let dtype: String?
    public let transposeAxes: [Int]?
    public let targetSpacing: [Double]?
    public let originalSpacing: [Double]?
    public let resampleKwargs: ResamplingKwargs?

    enum CodingKeys: String, CodingKey {
        case shape
        case spacing
        case bbox
        case mean
        case std
        case min
        case max
        case dtype
        case transposeAxes = "transpose_axes"
        case targetSpacing = "target_spacing"
        case originalSpacing = "original_spacing"
        case resampleKwargs = "resample_kwargs"
    }
}
