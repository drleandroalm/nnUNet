// Sources/nnUNetPreprocessing/Models/VolumeBuffer.swift

import Foundation
import simd

/// Internal volume representation for preprocessing pipeline
public struct VolumeBuffer: Sendable, Equatable {
    /// Raw voxel data (Float32)
    public var data: Data

    /// Shape in nnUNet convention: (depth, height, width)
    public var shape: (depth: Int, height: Int, width: Int)

    /// Physical spacing in mm: (z, y, x)
    public var spacing: SIMD3<Double>

    /// World coordinates origin (mm)
    public var origin: SIMD3<Double>

    /// Direction cosines from DICOM IOP
    public var orientation: simd_double3x3

    /// Bounding box after crop-to-nonzero (nil if not cropped)
    public var bbox: BoundingBox?

    /// Number of voxels
    public var voxelCount: Int {
        shape.depth * shape.height * shape.width
    }

    /// Size in bytes
    public var byteCount: Int {
        voxelCount * MemoryLayout<Float>.size
    }

    /// Initialize with raw data
    public init(
        data: Data,
        shape: (depth: Int, height: Int, width: Int),
        spacing: SIMD3<Double>,
        origin: SIMD3<Double> = .zero,
        orientation: simd_double3x3 = simd_double3x3(1),
        bbox: BoundingBox? = nil
    ) {
        self.data = data
        self.shape = shape
        self.spacing = spacing
        self.origin = origin
        self.orientation = orientation
        self.bbox = bbox
    }

    /// Access voxel data as Float array
    public func withUnsafeFloatPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeBytes { buffer in
            try body(buffer.bindMemory(to: Float.self).baseAddress!)
        }
    }

    /// Access voxel data as mutable Float array
    public mutating func withUnsafeMutableFloatPointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeMutableBytes { buffer in
            try body(buffer.bindMemory(to: Float.self).baseAddress!)
        }
    }
}

// MARK: - Equatable conformance for shape tuple
extension VolumeBuffer {
    public static func == (lhs: VolumeBuffer, rhs: VolumeBuffer) -> Bool {
        lhs.data == rhs.data &&
        lhs.shape.depth == rhs.shape.depth &&
        lhs.shape.height == rhs.shape.height &&
        lhs.shape.width == rhs.shape.width &&
        lhs.spacing == rhs.spacing &&
        lhs.origin == rhs.origin &&
        lhs.bbox == rhs.bbox
    }
}
