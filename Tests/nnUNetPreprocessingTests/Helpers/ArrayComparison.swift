// Tests/nnUNetPreprocessingTests/Helpers/ArrayComparison.swift

import XCTest

/// Helper for comparing float arrays with tolerance
public struct ArrayComparison {

    /// Assert two float arrays are equal within tolerance
    /// - Parameters:
    ///   - actual: Actual values
    ///   - expected: Expected values
    ///   - tolerance: Maximum allowed absolute difference
    ///   - file: Source file for failure reporting
    ///   - line: Source line for failure reporting
    public static func assertEqual(
        _ actual: [Float],
        _ expected: [Float],
        tolerance: Float,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.count,
            expected.count,
            "Array sizes differ: \(actual.count) vs \(expected.count)",
            file: file,
            line: line
        )

        guard actual.count == expected.count else { return }

        var maxDiff: Float = 0
        var maxDiffIndex = 0
        var diffCount = 0

        for i in 0..<actual.count {
            let diff = abs(actual[i] - expected[i])
            if diff > tolerance {
                diffCount += 1
            }
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIndex = i
            }
        }

        if maxDiff > tolerance {
            XCTFail(
                "Arrays differ: max diff = \(maxDiff) at index \(maxDiffIndex) " +
                "(actual: \(actual[maxDiffIndex]), expected: \(expected[maxDiffIndex])), " +
                "\(diffCount) values exceed tolerance \(tolerance)",
                file: file,
                line: line
            )
        }
    }

    /// Compute mean absolute error between arrays
    public static func meanAbsoluteError(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !actual.isEmpty else { return Float.infinity }

        var sum: Float = 0
        for i in 0..<actual.count {
            sum += abs(actual[i] - expected[i])
        }
        return sum / Float(actual.count)
    }

    /// Compute max absolute error between arrays
    public static func maxAbsoluteError(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count else { return Float.infinity }

        var maxDiff: Float = 0
        for i in 0..<actual.count {
            maxDiff = max(maxDiff, abs(actual[i] - expected[i]))
        }
        return maxDiff
    }

    /// Compute root mean squared error between arrays
    public static func rmse(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !actual.isEmpty else { return Float.infinity }

        var sumSquared: Float = 0
        for i in 0..<actual.count {
            let diff = actual[i] - expected[i]
            sumSquared += diff * diff
        }
        return sqrt(sumSquared / Float(actual.count))
    }

    /// Compare arrays with detailed statistics
    public static func compareWithStats(
        _ actual: [Float],
        _ expected: [Float]
    ) -> ComparisonStats {
        guard actual.count == expected.count else {
            return ComparisonStats(
                count: 0,
                maxAbsoluteError: Float.infinity,
                meanAbsoluteError: Float.infinity,
                rmse: Float.infinity,
                numExceedingTolerance: actual.count != expected.count ? -1 : 0
            )
        }

        let mae = meanAbsoluteError(actual, expected)
        let maxErr = maxAbsoluteError(actual, expected)
        let rmsErr = rmse(actual, expected)

        return ComparisonStats(
            count: actual.count,
            maxAbsoluteError: maxErr,
            meanAbsoluteError: mae,
            rmse: rmsErr,
            numExceedingTolerance: 0
        )
    }
}

/// Statistics from array comparison
public struct ComparisonStats: CustomStringConvertible {
    public let count: Int
    public let maxAbsoluteError: Float
    public let meanAbsoluteError: Float
    public let rmse: Float
    public let numExceedingTolerance: Int

    public var description: String {
        return """
        Comparison Stats:
          Count: \(count)
          Max Absolute Error: \(maxAbsoluteError)
          Mean Absolute Error: \(meanAbsoluteError)
          RMSE: \(rmse)
        """
    }
}
