// Sources/nnUNetPreprocessing/Models/BoundingBox.swift

import Foundation

/// Bounding box for crop-to-nonzero operation
public struct BoundingBox: Sendable, Equatable, Codable {
    /// Start indices (inclusive) for each axis: (z, y, x)
    public var start: (z: Int, y: Int, x: Int)

    /// End indices (exclusive) for each axis: (z, y, x)
    public var end: (z: Int, y: Int, x: Int)

    /// Size of bounding box
    public var size: (depth: Int, height: Int, width: Int) {
        (end.z - start.z, end.y - start.y, end.x - start.x)
    }

    public init(start: (z: Int, y: Int, x: Int), end: (z: Int, y: Int, x: Int)) {
        self.start = start
        self.end = end
    }

    // Codable conformance for tuple
    enum CodingKeys: String, CodingKey {
        case startZ, startY, startX
        case endZ, endY, endX
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        start = (
            try container.decode(Int.self, forKey: .startZ),
            try container.decode(Int.self, forKey: .startY),
            try container.decode(Int.self, forKey: .startX)
        )
        end = (
            try container.decode(Int.self, forKey: .endZ),
            try container.decode(Int.self, forKey: .endY),
            try container.decode(Int.self, forKey: .endX)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(start.z, forKey: .startZ)
        try container.encode(start.y, forKey: .startY)
        try container.encode(start.x, forKey: .startX)
        try container.encode(end.z, forKey: .endZ)
        try container.encode(end.y, forKey: .endY)
        try container.encode(end.x, forKey: .endX)
    }

    public static func == (lhs: BoundingBox, rhs: BoundingBox) -> Bool {
        lhs.start == rhs.start && lhs.end == rhs.end
    }
}
