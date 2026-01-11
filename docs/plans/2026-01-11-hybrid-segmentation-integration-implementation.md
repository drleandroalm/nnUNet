# Hybrid Segmentation Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build end-to-end workflow integration between nnUNet Metal preprocessing pipeline and Niivue iOS segmentation tools, with parameter tuning system and enhanced segmentation features for urinary tract CT.

**Architecture:** Enhanced WebView with Native Preprocessing - Native Swift/Metal preprocessing improves input quality, all segmentation happens in Niivue WebView with enhanced feature exposure, results persisted to native stores.

**Tech Stack:** Swift 6.2, SwiftUI, Metal, WebKit, nnUNetPreprocessing SPM package, XCTest

---

## Prerequisites

**Required Skills:**
- @apple-senior-developer - For iOS/SwiftUI/Metal patterns
- @superpowers:test-driven-development - For test-driven development

**Required Before Starting:**
```bash
# Verify nnUNet preprocessing package exists and builds
cd /Users/leandroalmeida/nnUNet
swift build

# Verify Niivue iOS app exists
ls /Users/leandroalmeida/niivue/ios/NiiVue/NiiVue.xcodeproj

# Verify test CT volume exists
ls /Users/leandroalmeida/niivue-ios-foundation/Test_CT_DICOM_volumes/Dicom_Volume_1/
```

**Reference Documents:**
- Design: `docs/plans/2026-01-11-hybrid-segmentation-integration-design.md`
- Phase 1 Implementation: `docs/reports/2026-01-11-phase1-tasks1-6-implementation-report.md`

---

## Phase A: Foundation

### Task A1: Add nnUNetPreprocessing as SPM Dependency

**Files:**
- Modify: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue.xcodeproj/project.pbxproj`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/Package.swift` (if using SPM)

**Step 1: Navigate to Niivue iOS project**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
```

**Step 2: Create feature branch**

Run:
```bash
git checkout -b feature/hybrid-segmentation-integration
```

Expected: Branch created

**Step 3: Add local package dependency via Xcode**

In Xcode:
1. Open `NiiVue.xcodeproj`
2. Select project in navigator → Package Dependencies tab
3. Click "+" → "Add Local..."
4. Navigate to `/Users/leandroalmeida/nnUNet`
5. Select the `nnUNetPreprocessing` product

Alternative via `Package.swift` if project uses SPM:
```swift
// Package.swift
dependencies: [
    .package(path: "../../nnUNet")
],
targets: [
    .target(
        name: "NiiVue",
        dependencies: [
            .product(name: "nnUNetPreprocessing", package: "nnUNet")
        ]
    )
]
```

**Step 4: Verify package resolves**

Run in Xcode: Product → Resolve Package Versions (⌃⇧⌘R)

Expected: Package resolves without errors

**Step 5: Verify import works**

Create temporary test file:
```swift
// NiiVue/TestImport.swift (temporary)
import nnUNetPreprocessing

func testImport() {
    // If this compiles, import works
    let _ = nnUNetPreprocessing.version
}
```

**Step 6: Build to verify compilation**

Run in Xcode: Product → Build (⌘B)

Expected: Build succeeds

**Step 7: Remove temporary test file**

Delete `NiiVue/TestImport.swift`

**Step 8: Commit**

Run:
```bash
git add -A
git commit -m "build: add nnUNetPreprocessing as SPM dependency"
```

---

### Task A2: Create CTPreprocessingParameters Model

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/CTPreprocessingParameters.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/CTPreprocessingParametersTests.swift`

**Step 1: Create Preprocessing directory**

Run:
```bash
mkdir -p /Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing
mkdir -p /Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing
```

**Step 2: Write failing test for parameters model**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/CTPreprocessingParametersTests.swift`
```swift
import XCTest
@testable import NiiVue

final class CTPreprocessingParametersTests: XCTestCase {

    func testDefaultInitialization() {
        // Arrange & Act
        let params = CTPreprocessingParameters()

        // Assert
        XCTAssertEqual(params.lowerPercentile, 0.5)
        XCTAssertEqual(params.upperPercentile, 99.5)
        XCTAssertTrue(params.useZScoreNormalization)
        XCTAssertEqual(params.isotropicTarget, 1.0)
        XCTAssertTrue(params.cropToNonzero)
        XCTAssertEqual(params.paddingVoxels, 10)
    }

    func testUrinaryTractDefaults() {
        // Arrange & Act
        let params = CTPreprocessingParameters.urinaryTractDefaults

        // Assert
        XCTAssertEqual(params.lowerPercentile, 0.5)
        XCTAssertEqual(params.upperPercentile, 99.5)
        XCTAssertEqual(params.isotropicTarget, 1.0)
        XCTAssertTrue(params.cropToNonzero)
        XCTAssertEqual(params.paddingVoxels, 10)
    }

    func testQuickPreviewPreset() {
        // Arrange & Act
        let params = CTPreprocessingParameters.quickPreview

        // Assert
        XCTAssertEqual(params.isotropicTarget, 2.0)
        XCTAssertEqual(params.interpolationOrder, 1) // Linear for speed
    }

    func testCodable() throws {
        // Arrange
        let params = CTPreprocessingParameters.urinaryTractDefaults
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Act
        let data = try encoder.encode(params)
        let decoded = try decoder.decode(CTPreprocessingParameters.self, from: data)

        // Assert
        XCTAssertEqual(params.lowerPercentile, decoded.lowerPercentile)
        XCTAssertEqual(params.upperPercentile, decoded.upperPercentile)
        XCTAssertEqual(params.isotropicTarget, decoded.isotropicTarget)
    }

    func testParametersHash() {
        // Arrange
        let params1 = CTPreprocessingParameters.urinaryTractDefaults
        let params2 = CTPreprocessingParameters.urinaryTractDefaults
        var params3 = CTPreprocessingParameters.urinaryTractDefaults
        params3.lowerPercentile = 1.0

        // Assert
        XCTAssertEqual(params1.parametersHash, params2.parametersHash)
        XCTAssertNotEqual(params1.parametersHash, params3.parametersHash)
    }
}
```

**Step 3: Run test to verify it fails**

Run in Xcode: Product → Test (⌘U) or run specific test

Expected: FAIL with "Cannot find type 'CTPreprocessingParameters' in scope"

**Step 4: Implement CTPreprocessingParameters**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/CTPreprocessingParameters.swift`
```swift
import Foundation
import CryptoKit

/// CT preprocessing parameters for nnUNet-style normalization
public struct CTPreprocessingParameters: Codable, Sendable, Equatable {

    // MARK: - Normalization

    /// Lower percentile for intensity clipping (0-100)
    public var lowerPercentile: Double

    /// Upper percentile for intensity clipping (0-100)
    public var upperPercentile: Double

    /// Whether to apply z-score normalization after clipping
    public var useZScoreNormalization: Bool

    // MARK: - Resampling

    /// Target isotropic spacing in mm (nil = keep original spacing)
    public var isotropicTarget: Double?

    /// Interpolation order (1 = linear, 3 = cubic B-spline)
    public var interpolationOrder: Int

    // MARK: - Cropping

    /// Whether to crop to non-zero region
    public var cropToNonzero: Bool

    /// Padding voxels around crop region
    public var paddingVoxels: Int

    // MARK: - Optional HU Range

    /// Custom HU clip range (overrides percentile if set)
    public var customClipMin: Double?
    public var customClipMax: Double?

    /// Use custom clip as pre-filter before percentile calculation
    public var useCustomClipAsPrefilter: Bool

    // MARK: - Initialization

    public init(
        lowerPercentile: Double = 0.5,
        upperPercentile: Double = 99.5,
        useZScoreNormalization: Bool = true,
        isotropicTarget: Double? = 1.0,
        interpolationOrder: Int = 3,
        cropToNonzero: Bool = true,
        paddingVoxels: Int = 10,
        customClipMin: Double? = nil,
        customClipMax: Double? = nil,
        useCustomClipAsPrefilter: Bool = false
    ) {
        self.lowerPercentile = lowerPercentile
        self.upperPercentile = upperPercentile
        self.useZScoreNormalization = useZScoreNormalization
        self.isotropicTarget = isotropicTarget
        self.interpolationOrder = interpolationOrder
        self.cropToNonzero = cropToNonzero
        self.paddingVoxels = paddingVoxels
        self.customClipMin = customClipMin
        self.customClipMax = customClipMax
        self.useCustomClipAsPrefilter = useCustomClipAsPrefilter
    }

    // MARK: - Presets

    /// Default parameters for urinary tract CT
    public static let urinaryTractDefaults = CTPreprocessingParameters(
        lowerPercentile: 0.5,
        upperPercentile: 99.5,
        useZScoreNormalization: true,
        isotropicTarget: 1.0,
        interpolationOrder: 3,
        cropToNonzero: true,
        paddingVoxels: 10
    )

    /// Quick preview with lower resolution
    public static let quickPreview = CTPreprocessingParameters(
        lowerPercentile: 0.5,
        upperPercentile: 99.5,
        useZScoreNormalization: true,
        isotropicTarget: 2.0,
        interpolationOrder: 1,
        cropToNonzero: true,
        paddingVoxels: 5
    )

    /// Baseline - no intensity changes
    public static let baseline = CTPreprocessingParameters(
        lowerPercentile: 0.0,
        upperPercentile: 100.0,
        useZScoreNormalization: false,
        isotropicTarget: 1.0,
        interpolationOrder: 3,
        cropToNonzero: false,
        paddingVoxels: 0
    )

    // MARK: - Hash

    /// Unique hash for cache key generation
    public var parametersHash: String {
        let components = [
            String(lowerPercentile),
            String(upperPercentile),
            String(useZScoreNormalization),
            String(isotropicTarget ?? -1),
            String(interpolationOrder),
            String(cropToNonzero),
            String(paddingVoxels),
            String(customClipMin ?? -999999),
            String(customClipMax ?? -999999),
            String(useCustomClipAsPrefilter)
        ]
        let joined = components.joined(separator: "|")
        let data = Data(joined.utf8)
        let hash = SHA256.hash(data: data)
        return hash.prefix(16).map { String(format: "%02x", $0) }.joined()
    }
}
```

**Step 5: Run test to verify it passes**

Run in Xcode: Product → Test (⌘U)

Expected: All tests PASS

**Step 6: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Services/Preprocessing/CTPreprocessingParameters.swift \
        NiiVueTests/Preprocessing/CTPreprocessingParametersTests.swift
git commit -m "feat: add CTPreprocessingParameters model with presets"
```

---

### Task A3: Create All 12 Parameter Variants

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/ParameterVariants.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/ParameterVariantsTests.swift`

**Step 1: Write failing test for variants**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/ParameterVariantsTests.swift`
```swift
import XCTest
@testable import NiiVue

final class ParameterVariantsTests: XCTestCase {

    func testUrinaryTractVariantsCount() {
        // Assert
        XCTAssertEqual(ParameterVariants.urinaryTractCT.count, 12)
    }

    func testAllVariantsHaveUniqueIDs() {
        // Arrange
        let variants = ParameterVariants.urinaryTractCT
        let ids = variants.map { $0.id }
        let uniqueIDs = Set(ids)

        // Assert
        XCTAssertEqual(ids.count, uniqueIDs.count, "All variant IDs should be unique")
    }

    func testAllVariantsHaveNames() {
        // Arrange
        let variants = ParameterVariants.urinaryTractCT

        // Assert
        for variant in variants {
            XCTAssertFalse(variant.name.isEmpty, "Variant should have a name")
            XCTAssertFalse(variant.description.isEmpty, "Variant should have a description")
        }
    }

    func testBaselineVariantHasNoNormalization() {
        // Arrange
        guard let baseline = ParameterVariants.urinaryTractCT.first(where: { $0.name.contains("Baseline") }) else {
            XCTFail("Should have baseline variant")
            return
        }

        // Assert
        XCTAssertFalse(baseline.parameters.useZScoreNormalization)
        XCTAssertFalse(baseline.parameters.cropToNonzero)
    }

    func testHighResolutionVariant() {
        // Arrange
        guard let highRes = ParameterVariants.urinaryTractCT.first(where: { $0.name.contains("High Resolution") && !$0.name.contains("Ultra") }) else {
            XCTFail("Should have high resolution variant")
            return
        }

        // Assert
        XCTAssertEqual(highRes.parameters.isotropicTarget, 0.5)
    }

    func testQuickPreviewVariant() {
        // Arrange
        guard let quick = ParameterVariants.urinaryTractCT.first(where: { $0.name.contains("Quick") }) else {
            XCTFail("Should have quick preview variant")
            return
        }

        // Assert
        XCTAssertEqual(quick.parameters.isotropicTarget, 2.0)
        XCTAssertEqual(quick.parameters.interpolationOrder, 1)
    }
}
```

**Step 2: Run test to verify it fails**

Run in Xcode: Product → Test (⌘U)

Expected: FAIL with "Cannot find 'ParameterVariants' in scope"

**Step 3: Implement ParameterVariants**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/ParameterVariants.swift`
```swift
import Foundation

/// A named parameter configuration for testing/comparison
public struct ParameterVariant: Identifiable, Codable, Sendable {
    public let id: UUID
    public var name: String
    public var parameters: CTPreprocessingParameters
    public var description: String

    public init(
        id: UUID = UUID(),
        name: String,
        parameters: CTPreprocessingParameters,
        description: String
    ) {
        self.id = id
        self.name = name
        self.parameters = parameters
        self.description = description
    }
}

/// Predefined parameter variants for systematic testing
public enum ParameterVariants {

    /// 12 variants for comprehensive urinary tract CT parameter tuning
    public static let urinaryTractCT: [ParameterVariant] = [

        // ═══════════════════════════════════════════════════════════════
        // BASELINE & STANDARD
        // ═══════════════════════════════════════════════════════════════

        ParameterVariant(
            name: "1. Baseline (Raw)",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.0,
                upperPercentile: 100.0,
                useZScoreNormalization: false,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: false,
                paddingVoxels: 0
            ),
            description: "No intensity changes, only resampling. Control group for comparison."
        ),

        ParameterVariant(
            name: "2. Standard Balanced",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 10
            ),
            description: "Balanced settings for general urinary tract work. Good starting point."
        ),

        // ═══════════════════════════════════════════════════════════════
        // INTENSITY NORMALIZATION VARIANTS
        // ═══════════════════════════════════════════════════════════════

        ParameterVariant(
            name: "3. Conservative Intensity",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.1,
                upperPercentile: 99.9,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 10
            ),
            description: "Wide intensity range preserves subtle contrast differences. Good for heterogeneous tissues."
        ),

        ParameterVariant(
            name: "4. Aggressive Intensity",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 2.0,
                upperPercentile: 98.0,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 10
            ),
            description: "Tight clipping removes outliers, enhances organ-to-background contrast."
        ),

        ParameterVariant(
            name: "5. Maximum Contrast",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 5.0,
                upperPercentile: 95.0,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 5
            ),
            description: "Very aggressive clipping for maximum organ boundary definition. May lose subtle details."
        ),

        // ═══════════════════════════════════════════════════════════════
        // RESOLUTION VARIANTS
        // ═══════════════════════════════════════════════════════════════

        ParameterVariant(
            name: "6. High Resolution (0.5mm)",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 0.5,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 20
            ),
            description: "0.5mm isotropic for detailed boundary work. 8x more voxels, slower processing."
        ),

        ParameterVariant(
            name: "7. Ultra High Resolution (0.25mm)",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 0.25,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 40
            ),
            description: "0.25mm isotropic for finest detail. 64x more voxels, requires significant memory."
        ),

        ParameterVariant(
            name: "8. Quick Preview (2.0mm)",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 2.0,
                interpolationOrder: 1,
                cropToNonzero: true,
                paddingVoxels: 5
            ),
            description: "2.0mm isotropic with linear interpolation. Fast iteration, rough boundaries."
        ),

        // ═══════════════════════════════════════════════════════════════
        // CLINICAL SCENARIO VARIANTS
        // ═══════════════════════════════════════════════════════════════

        ParameterVariant(
            name: "9. Contrast-Enhanced CT",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 1.0,
                upperPercentile: 99.0,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 10,
                customClipMin: -100,
                customClipMax: 400
            ),
            description: "Optimized for contrast-enhanced CT. Clips to typical contrast HU range."
        ),

        ParameterVariant(
            name: "10. Non-Contrast CT",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 1.0,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 15,
                customClipMin: -150,
                customClipMax: 200
            ),
            description: "Optimized for non-contrast CT. Wider range to preserve subtle soft tissue differences."
        ),

        // ═══════════════════════════════════════════════════════════════
        // ANATOMICAL TARGET VARIANTS
        // ═══════════════════════════════════════════════════════════════

        ParameterVariant(
            name: "11. Kidney-Focused",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 1.0,
                upperPercentile: 99.0,
                useZScoreNormalization: true,
                isotropicTarget: 0.75,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 15,
                customClipMin: -50,
                customClipMax: 300
            ),
            description: "Optimized for kidney segmentation. Enhanced cortex-medulla differentiation."
        ),

        ParameterVariant(
            name: "12. Ureter-Focused",
            parameters: CTPreprocessingParameters(
                lowerPercentile: 0.5,
                upperPercentile: 99.5,
                useZScoreNormalization: true,
                isotropicTarget: 0.5,
                interpolationOrder: 3,
                cropToNonzero: true,
                paddingVoxels: 10,
                customClipMin: 0,
                customClipMax: 500
            ),
            description: "Optimized for ureter segmentation. High resolution for thin tubular anatomy."
        )
    ]

    /// Get variant by name (partial match)
    public static func find(named name: String) -> ParameterVariant? {
        urinaryTractCT.first { $0.name.lowercased().contains(name.lowercased()) }
    }
}
```

**Step 4: Run test to verify it passes**

Run in Xcode: Product → Test (⌘U)

Expected: All tests PASS

**Step 5: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Services/Preprocessing/ParameterVariants.swift \
        NiiVueTests/Preprocessing/ParameterVariantsTests.swift
git commit -m "feat: add 12 predefined parameter variants for urinary tract CT"
```

---

### Task A4: Create PreprocessedResult Model

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessedResult.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessedResultTests.swift`

**Step 1: Write failing test**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessedResultTests.swift`
```swift
import XCTest
@testable import NiiVue

final class PreprocessedResultTests: XCTestCase {

    func testResultCreation() {
        // Arrange & Act
        let result = PreprocessedResult(
            outputURL: URL(fileURLWithPath: "/tmp/test.nii.gz"),
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(256, 256, 245),
            processingTime: 2.5,
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertEqual(result.originalShape.x, 512)
        XCTAssertEqual(result.processedShape.x, 256)
        XCTAssertEqual(result.processingTime, 2.5, accuracy: 0.01)
    }

    func testResultWithBoundingBox() {
        // Arrange
        let bbox = PreprocessedResult.BoundingBox(
            minX: 50, maxX: 450,
            minY: 50, maxY: 450,
            minZ: 10, maxZ: 235
        )

        // Act
        let result = PreprocessedResult(
            outputURL: URL(fileURLWithPath: "/tmp/test.nii.gz"),
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(400, 400, 225),
            boundingBox: bbox,
            processingTime: 3.0,
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertNotNil(result.boundingBox)
        XCTAssertEqual(result.boundingBox?.minX, 50)
        XCTAssertEqual(result.boundingBox?.size.x, 400)
    }

    func testResultCodable() throws {
        // Arrange
        let result = PreprocessedResult(
            outputURL: URL(fileURLWithPath: "/tmp/test.nii.gz"),
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(256, 256, 245),
            processingTime: 2.5,
            parameters: .urinaryTractDefaults
        )

        // Act
        let encoder = JSONEncoder()
        let data = try encoder.encode(result)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PreprocessedResult.self, from: data)

        // Assert
        XCTAssertEqual(result.originalShape, decoded.originalShape)
        XCTAssertEqual(result.processedShape, decoded.processedShape)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL with "Cannot find type 'PreprocessedResult' in scope"

**Step 3: Implement PreprocessedResult**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessedResult.swift`
```swift
import Foundation

/// Result of preprocessing a volume
public struct PreprocessedResult: Codable, Sendable {

    /// URL to the preprocessed NIfTI file
    public let outputURL: URL

    /// Original volume shape (width, height, depth)
    public let originalShape: SIMD3<Int>

    /// Processed volume shape after resampling
    public let processedShape: SIMD3<Int>

    /// Bounding box used for cropping (for inverse transform)
    public let boundingBox: BoundingBox?

    /// Processing time in seconds
    public let processingTime: TimeInterval

    /// Parameters used for preprocessing
    public let parameters: CTPreprocessingParameters

    /// Timestamp when preprocessing completed
    public let timestamp: Date

    public init(
        outputURL: URL,
        originalShape: SIMD3<Int>,
        processedShape: SIMD3<Int>,
        boundingBox: BoundingBox? = nil,
        processingTime: TimeInterval,
        parameters: CTPreprocessingParameters,
        timestamp: Date = Date()
    ) {
        self.outputURL = outputURL
        self.originalShape = originalShape
        self.processedShape = processedShape
        self.boundingBox = boundingBox
        self.processingTime = processingTime
        self.parameters = parameters
        self.timestamp = timestamp
    }

    /// Bounding box for cropped region
    public struct BoundingBox: Codable, Sendable, Equatable {
        public let minX: Int
        public let maxX: Int
        public let minY: Int
        public let maxY: Int
        public let minZ: Int
        public let maxZ: Int

        public init(minX: Int, maxX: Int, minY: Int, maxY: Int, minZ: Int, maxZ: Int) {
            self.minX = minX
            self.maxX = maxX
            self.minY = minY
            self.maxY = maxY
            self.minZ = minZ
            self.maxZ = maxZ
        }

        /// Size of the bounding box
        public var size: SIMD3<Int> {
            SIMD3<Int>(maxX - minX, maxY - minY, maxZ - minZ)
        }

        /// Origin of the bounding box
        public var origin: SIMD3<Int> {
            SIMD3<Int>(minX, minY, minZ)
        }
    }
}

// MARK: - SIMD3<Int> Codable Extension

extension SIMD3: Codable where Scalar == Int {
    enum CodingKeys: String, CodingKey {
        case x, y, z
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Int.self, forKey: .x)
        let y = try container.decode(Int.self, forKey: .y)
        let z = try container.decode(Int.self, forKey: .z)
        self.init(x, y, z)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
}
```

**Step 4: Run test to verify it passes**

Expected: All tests PASS

**Step 5: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Services/Preprocessing/PreprocessedResult.swift \
        NiiVueTests/Preprocessing/PreprocessedResultTests.swift
git commit -m "feat: add PreprocessedResult model with bounding box support"
```

---

### Task A5: Create PreprocessingService Interface

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessingService.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessingServiceTests.swift`

**Step 1: Write failing test for service**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessingServiceTests.swift`
```swift
import XCTest
@testable import NiiVue

@MainActor
final class PreprocessingServiceTests: XCTestCase {

    var service: PreprocessingService!

    override func setUp() async throws {
        try await super.setUp()
        service = PreprocessingService()
    }

    func testInitialization() {
        // Assert
        XCTAssertTrue(service.isEnabled)
        XCTAssertEqual(service.parameters.lowerPercentile, 0.5)
    }

    func testToggleEnabled() {
        // Act
        service.isEnabled = false

        // Assert
        XCTAssertFalse(service.isEnabled)
    }

    func testSetParameters() {
        // Arrange
        let newParams = CTPreprocessingParameters.quickPreview

        // Act
        service.parameters = newParams

        // Assert
        XCTAssertEqual(service.parameters.isotropicTarget, 2.0)
    }

    func testProgressObservable() {
        // Assert initial state
        XCTAssertEqual(service.progress, 0.0)
        XCTAssertFalse(service.isProcessing)
        XCTAssertNil(service.currentOperation)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL with "Cannot find type 'PreprocessingService' in scope"

**Step 3: Implement PreprocessingService**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessingService.swift`
```swift
import Foundation
import Observation
import os.log
import nnUNetPreprocessing

/// Service for preprocessing CT volumes using nnUNet Metal pipeline
@MainActor
@Observable
public final class PreprocessingService {

    private let logger = Logger(subsystem: "com.niivue", category: "PreprocessingService")

    // MARK: - Configuration

    /// Whether preprocessing is enabled
    public var isEnabled: Bool = true

    /// Current preprocessing parameters
    public var parameters: CTPreprocessingParameters = .urinaryTractDefaults

    // MARK: - Progress State

    /// Current progress (0.0 - 1.0)
    public private(set) var progress: Double = 0.0

    /// Whether currently processing
    public private(set) var isProcessing: Bool = false

    /// Current operation description
    public private(set) var currentOperation: String?

    /// Last error (if any)
    public private(set) var lastError: Error?

    // MARK: - Dependencies

    private let cache: PreprocessedVolumeCache

    // MARK: - Initialization

    public init(cache: PreprocessedVolumeCache? = nil) {
        self.cache = cache ?? PreprocessedVolumeCache()
    }

    // MARK: - Public API

    /// Preprocess a volume
    /// - Parameters:
    ///   - studyID: Study identifier
    ///   - itemID: Item identifier within study
    ///   - sourceURL: URL to source NIfTI or DICOM
    /// - Returns: Preprocessing result with output URL
    public func preprocess(
        studyID: String,
        itemID: String,
        sourceURL: URL
    ) async throws -> PreprocessedResult {
        guard isEnabled else {
            throw PreprocessingError.disabled
        }

        logger.info("Starting preprocessing for \(studyID)/\(itemID)")

        // Check cache first
        if let cached = await cache.getCached(
            studyID: studyID,
            itemID: itemID,
            parameters: parameters
        ) {
            logger.info("Using cached result")
            return cached
        }

        // Reset state
        isProcessing = true
        progress = 0.0
        lastError = nil

        defer {
            isProcessing = false
            currentOperation = nil
        }

        do {
            let result = try await performPreprocessing(
                studyID: studyID,
                itemID: itemID,
                sourceURL: sourceURL
            )

            // Cache result
            try await cache.store(
                studyID: studyID,
                itemID: itemID,
                result: result
            )

            logger.info("Preprocessing complete: \(result.processingTime)s")
            return result

        } catch {
            lastError = error
            logger.error("Preprocessing failed: \(error.localizedDescription)")
            throw error
        }
    }

    /// Check if cached result exists
    public func hasCachedResult(studyID: String, itemID: String) async -> Bool {
        await cache.hasCached(
            studyID: studyID,
            itemID: itemID,
            parameters: parameters
        )
    }

    /// Clear cache for a study
    public func clearCache(studyID: String) async {
        await cache.invalidate(studyID: studyID)
    }

    /// Clear all cache
    public func clearAllCache() async {
        await cache.clearAll()
    }

    // MARK: - Private Implementation

    private func performPreprocessing(
        studyID: String,
        itemID: String,
        sourceURL: URL
    ) async throws -> PreprocessedResult {
        let startTime = Date()

        // Step 1: Load volume (20%)
        currentOperation = "Loading volume..."
        progress = 0.1

        // TODO: Integrate with nnUNetPreprocessing package
        // This is a placeholder - actual implementation will:
        // 1. Load NIfTI/DICOM using DicomBridge
        // 2. Convert to VolumeBuffer
        // 3. Apply Transpose, CropToNonzero, CTNormalization, Resampling
        // 4. Save output NIfTI

        // Step 2: Normalize (40%)
        currentOperation = "Normalizing intensities..."
        progress = 0.4
        try await Task.sleep(for: .milliseconds(100)) // Placeholder

        // Step 3: Resample (70%)
        currentOperation = "Resampling to target spacing..."
        progress = 0.7
        try await Task.sleep(for: .milliseconds(100)) // Placeholder

        // Step 4: Save output (90%)
        currentOperation = "Saving preprocessed volume..."
        progress = 0.9

        // Create output path
        let outputDir = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("NiiVue/Preprocessed")
            .appendingPathComponent(studyID)
            .appendingPathComponent(itemID)

        try FileManager.default.createDirectory(
            at: outputDir,
            withIntermediateDirectories: true
        )

        let outputURL = outputDir.appendingPathComponent("preprocessed.nii.gz")

        // Placeholder: copy source as "preprocessed" for now
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        try FileManager.default.copyItem(at: sourceURL, to: outputURL)

        progress = 1.0
        currentOperation = nil

        let processingTime = Date().timeIntervalSince(startTime)

        return PreprocessedResult(
            outputURL: outputURL,
            originalShape: SIMD3<Int>(512, 512, 245), // Placeholder
            processedShape: SIMD3<Int>(512, 512, 245), // Placeholder
            processingTime: processingTime,
            parameters: parameters
        )
    }
}

// MARK: - Errors

public enum PreprocessingError: LocalizedError {
    case disabled
    case fileNotFound(URL)
    case invalidFormat(String)
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .disabled:
            return "Preprocessing is disabled"
        case .fileNotFound(let url):
            return "File not found: \(url.path)"
        case .invalidFormat(let reason):
            return "Invalid format: \(reason)"
        case .processingFailed(let reason):
            return "Processing failed: \(reason)"
        }
    }
}
```

**Step 4: Run test to verify it passes**

Expected: All tests PASS

**Step 5: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Services/Preprocessing/PreprocessingService.swift \
        NiiVueTests/Preprocessing/PreprocessingServiceTests.swift
git commit -m "feat: add PreprocessingService with cache integration"
```

---

### Task A6: Create PreprocessedVolumeCache

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessedVolumeCache.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessedVolumeCacheTests.swift`

**Step 1: Write failing test for cache**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Preprocessing/PreprocessedVolumeCacheTests.swift`
```swift
import XCTest
@testable import NiiVue

final class PreprocessedVolumeCacheTests: XCTestCase {

    var cache: PreprocessedVolumeCache!
    var testDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        testDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("NiiVueCacheTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
        cache = PreprocessedVolumeCache(baseURL: testDir)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: testDir)
        try await super.tearDown()
    }

    func testCacheMiss() async {
        // Act
        let result = await cache.getCached(
            studyID: "nonexistent",
            itemID: "item",
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertNil(result)
    }

    func testHasCachedReturnsFalseForMiss() async {
        // Act
        let hasCached = await cache.hasCached(
            studyID: "nonexistent",
            itemID: "item",
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertFalse(hasCached)
    }

    func testStoreAndRetrieve() async throws {
        // Arrange
        let testFile = testDir.appendingPathComponent("test.nii.gz")
        try "test data".write(to: testFile, atomically: true, encoding: .utf8)

        let result = PreprocessedResult(
            outputURL: testFile,
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(256, 256, 245),
            processingTime: 2.5,
            parameters: .urinaryTractDefaults
        )

        // Act
        try await cache.store(studyID: "study1", itemID: "item1", result: result)
        let retrieved = await cache.getCached(
            studyID: "study1",
            itemID: "item1",
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.originalShape, result.originalShape)
    }

    func testCacheInvalidationByStudy() async throws {
        // Arrange
        let testFile = testDir.appendingPathComponent("test.nii.gz")
        try "test data".write(to: testFile, atomically: true, encoding: .utf8)

        let result = PreprocessedResult(
            outputURL: testFile,
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(256, 256, 245),
            processingTime: 2.5,
            parameters: .urinaryTractDefaults
        )

        try await cache.store(studyID: "study1", itemID: "item1", result: result)

        // Act
        await cache.invalidate(studyID: "study1")
        let retrieved = await cache.getCached(
            studyID: "study1",
            itemID: "item1",
            parameters: .urinaryTractDefaults
        )

        // Assert
        XCTAssertNil(retrieved)
    }

    func testDifferentParametersNotCached() async throws {
        // Arrange
        let testFile = testDir.appendingPathComponent("test.nii.gz")
        try "test data".write(to: testFile, atomically: true, encoding: .utf8)

        let result = PreprocessedResult(
            outputURL: testFile,
            originalShape: SIMD3<Int>(512, 512, 245),
            processedShape: SIMD3<Int>(256, 256, 245),
            processingTime: 2.5,
            parameters: .urinaryTractDefaults
        )

        try await cache.store(studyID: "study1", itemID: "item1", result: result)

        // Act - retrieve with different parameters
        let retrieved = await cache.getCached(
            studyID: "study1",
            itemID: "item1",
            parameters: .quickPreview  // Different parameters
        )

        // Assert
        XCTAssertNil(retrieved)
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL with "Cannot find type 'PreprocessedVolumeCache' in scope"

**Step 3: Implement PreprocessedVolumeCache**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Services/Preprocessing/PreprocessedVolumeCache.swift`
```swift
import Foundation
import os.log

/// Cache for preprocessed volumes with automatic eviction
public actor PreprocessedVolumeCache {

    private let logger = Logger(subsystem: "com.niivue", category: "PreprocessedVolumeCache")

    private let baseURL: URL
    private let maxCacheSizeBytes: Int64

    private var entries: [String: CacheEntry] = [:]

    // MARK: - Initialization

    public init(
        baseURL: URL? = nil,
        maxCacheSizeGB: Double = 10.0
    ) {
        self.baseURL = baseURL ?? FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("NiiVue/PreprocessedCache")
        self.maxCacheSizeBytes = Int64(maxCacheSizeGB * 1024 * 1024 * 1024)

        // Create base directory if needed
        try? FileManager.default.createDirectory(
            at: self.baseURL,
            withIntermediateDirectories: true
        )

        // Load existing entries from disk
        Task { await loadEntriesFromDisk() }
    }

    // MARK: - Public API

    /// Check if cached version exists for given parameters
    public func hasCached(
        studyID: String,
        itemID: String,
        parameters: CTPreprocessingParameters
    ) -> Bool {
        let key = cacheKey(studyID: studyID, itemID: itemID, parameters: parameters)
        guard let entry = entries[key] else { return false }
        return FileManager.default.fileExists(atPath: entry.volumeURL.path)
    }

    /// Get cached preprocessed result
    public func getCached(
        studyID: String,
        itemID: String,
        parameters: CTPreprocessingParameters
    ) -> PreprocessedResult? {
        let key = cacheKey(studyID: studyID, itemID: itemID, parameters: parameters)
        guard let entry = entries[key] else { return nil }

        // Verify file still exists
        guard FileManager.default.fileExists(atPath: entry.volumeURL.path) else {
            entries.removeValue(forKey: key)
            return nil
        }

        // Update last accessed time
        var updatedEntry = entry
        updatedEntry.lastAccessedAt = Date()
        entries[key] = updatedEntry

        return entry.result
    }

    /// Store preprocessed result
    public func store(
        studyID: String,
        itemID: String,
        result: PreprocessedResult
    ) throws {
        let key = cacheKey(studyID: studyID, itemID: itemID, parameters: result.parameters)

        // Create cache directory
        let cacheDir = baseURL
            .appendingPathComponent(studyID)
            .appendingPathComponent(itemID)
            .appendingPathComponent(result.parameters.parametersHash)

        try FileManager.default.createDirectory(
            at: cacheDir,
            withIntermediateDirectories: true
        )

        // Copy volume to cache
        let cachedVolumeURL = cacheDir.appendingPathComponent("preprocessed.nii.gz")
        if FileManager.default.fileExists(atPath: cachedVolumeURL.path) {
            try FileManager.default.removeItem(at: cachedVolumeURL)
        }
        try FileManager.default.copyItem(at: result.outputURL, to: cachedVolumeURL)

        // Save metadata
        let metadataURL = cacheDir.appendingPathComponent("metadata.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let metadataData = try encoder.encode(result)
        try metadataData.write(to: metadataURL)

        // Calculate size
        let attributes = try FileManager.default.attributesOfItem(atPath: cachedVolumeURL.path)
        let size = attributes[.size] as? Int64 ?? 0

        // Create entry
        let entry = CacheEntry(
            studyID: studyID,
            itemID: itemID,
            parametersHash: result.parameters.parametersHash,
            volumeURL: cachedVolumeURL,
            metadataURL: metadataURL,
            sizeBytes: size,
            createdAt: Date(),
            lastAccessedAt: Date(),
            result: PreprocessedResult(
                outputURL: cachedVolumeURL,
                originalShape: result.originalShape,
                processedShape: result.processedShape,
                boundingBox: result.boundingBox,
                processingTime: result.processingTime,
                parameters: result.parameters,
                timestamp: result.timestamp
            )
        )

        entries[key] = entry

        logger.info("Cached: \(key) (\(size) bytes)")

        // Evict if needed
        Task { await evictIfNeeded() }
    }

    /// Invalidate cache for a study
    public func invalidate(studyID: String) {
        let keysToRemove = entries.keys.filter { $0.hasPrefix("\(studyID)/") }
        for key in keysToRemove {
            if let entry = entries[key] {
                try? FileManager.default.removeItem(at: entry.volumeURL.deletingLastPathComponent())
            }
            entries.removeValue(forKey: key)
        }
        logger.info("Invalidated cache for study: \(studyID)")
    }

    /// Invalidate specific item
    public func invalidate(studyID: String, itemID: String) {
        let prefix = "\(studyID)/\(itemID)/"
        let keysToRemove = entries.keys.filter { $0.hasPrefix(prefix) }
        for key in keysToRemove {
            if let entry = entries[key] {
                try? FileManager.default.removeItem(at: entry.volumeURL.deletingLastPathComponent())
            }
            entries.removeValue(forKey: key)
        }
        logger.info("Invalidated cache for: \(studyID)/\(itemID)")
    }

    /// Get total cache size
    public func getCacheSizeBytes() -> Int64 {
        entries.values.reduce(0) { $0 + $1.sizeBytes }
    }

    /// Clear entire cache
    public func clearAll() {
        try? FileManager.default.removeItem(at: baseURL)
        try? FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        entries.removeAll()
        logger.info("Cleared all cache")
    }

    // MARK: - Private

    private func cacheKey(
        studyID: String,
        itemID: String,
        parameters: CTPreprocessingParameters
    ) -> String {
        "\(studyID)/\(itemID)/\(parameters.parametersHash)"
    }

    private func evictIfNeeded() {
        var currentSize = getCacheSizeBytes()

        guard currentSize > maxCacheSizeBytes else { return }

        // Sort by last accessed (oldest first)
        let sortedEntries = entries.sorted { $0.value.lastAccessedAt < $1.value.lastAccessedAt }

        for (key, entry) in sortedEntries {
            guard currentSize > maxCacheSizeBytes else { break }

            try? FileManager.default.removeItem(at: entry.volumeURL.deletingLastPathComponent())
            entries.removeValue(forKey: key)
            currentSize -= entry.sizeBytes

            logger.info("Evicted: \(key)")
        }
    }

    private func loadEntriesFromDisk() {
        // Scan cache directory for existing entries
        guard let studyDirs = try? FileManager.default.contentsOfDirectory(
            at: baseURL,
            includingPropertiesForKeys: nil
        ) else { return }

        for studyDir in studyDirs {
            let studyID = studyDir.lastPathComponent
            guard let itemDirs = try? FileManager.default.contentsOfDirectory(
                at: studyDir,
                includingPropertiesForKeys: nil
            ) else { continue }

            for itemDir in itemDirs {
                let itemID = itemDir.lastPathComponent
                guard let hashDirs = try? FileManager.default.contentsOfDirectory(
                    at: itemDir,
                    includingPropertiesForKeys: nil
                ) else { continue }

                for hashDir in hashDirs {
                    let metadataURL = hashDir.appendingPathComponent("metadata.json")
                    let volumeURL = hashDir.appendingPathComponent("preprocessed.nii.gz")

                    guard FileManager.default.fileExists(atPath: metadataURL.path),
                          FileManager.default.fileExists(atPath: volumeURL.path),
                          let metadataData = try? Data(contentsOf: metadataURL),
                          let result = try? JSONDecoder().decode(PreprocessedResult.self, from: metadataData)
                    else { continue }

                    let key = "\(studyID)/\(itemID)/\(hashDir.lastPathComponent)"
                    let attributes = try? FileManager.default.attributesOfItem(atPath: volumeURL.path)
                    let size = attributes?[.size] as? Int64 ?? 0

                    entries[key] = CacheEntry(
                        studyID: studyID,
                        itemID: itemID,
                        parametersHash: hashDir.lastPathComponent,
                        volumeURL: volumeURL,
                        metadataURL: metadataURL,
                        sizeBytes: size,
                        createdAt: result.timestamp,
                        lastAccessedAt: Date(),
                        result: result
                    )
                }
            }
        }

        logger.info("Loaded \(entries.count) cached entries")
    }
}

// MARK: - Cache Entry

private struct CacheEntry {
    let studyID: String
    let itemID: String
    let parametersHash: String
    let volumeURL: URL
    let metadataURL: URL
    let sizeBytes: Int64
    let createdAt: Date
    var lastAccessedAt: Date
    let result: PreprocessedResult
}
```

**Step 4: Run test to verify it passes**

Expected: All tests PASS

**Step 5: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Services/Preprocessing/PreprocessedVolumeCache.swift \
        NiiVueTests/Preprocessing/PreprocessedVolumeCacheTests.swift
git commit -m "feat: add PreprocessedVolumeCache with LRU eviction"
```

---

### Task A7: Extend NiivueURLSchemeHandler for Preprocessed Volumes

**Files:**
- Modify: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Web/NiivueURLRouter.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Web/NiivueURLRouterPreprocessedTests.swift`

**Step 1: Write failing test**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Web/NiivueURLRouterPreprocessedTests.swift`
```swift
import XCTest
@testable import NiiVue

final class NiivueURLRouterPreprocessedTests: XCTestCase {

    func testParsePreprocessedURL() throws {
        // Arrange
        let urlString = "niivue://app/preprocessed/study123/item456"
        let url = URL(string: urlString)!

        // Act
        let route = NiivueURLRouter.route(for: url)

        // Assert
        if case .preprocessed(let studyID, let itemID) = route {
            XCTAssertEqual(studyID, "study123")
            XCTAssertEqual(itemID, "item456")
        } else {
            XCTFail("Expected preprocessed route, got \(route)")
        }
    }

    func testPreprocessedRouteWithTrailingSlash() throws {
        // Arrange
        let url = URL(string: "niivue://app/preprocessed/study123/item456/")!

        // Act
        let route = NiivueURLRouter.route(for: url)

        // Assert
        if case .preprocessed(let studyID, let itemID) = route {
            XCTAssertEqual(studyID, "study123")
            XCTAssertEqual(itemID, "item456")
        } else {
            XCTFail("Expected preprocessed route")
        }
    }
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL (preprocessed case not handled)

**Step 3: Read existing NiivueURLRouter**

Run:
```bash
cat /Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Web/NiivueURLRouter.swift | head -100
```

Understand the existing route enum and add the preprocessed case.

**Step 4: Modify NiivueURLRouter to add preprocessed route**

Add to the Route enum:
```swift
case preprocessed(studyID: String, itemID: String)
```

Add to the route parsing logic:
```swift
case "preprocessed":
    // niivue://app/preprocessed/{studyID}/{itemID}
    guard components.count >= 4 else { return .notFound }
    let studyID = components[2]
    let itemID = components[3].replacingOccurrences(of: "/", with: "")
    return .preprocessed(studyID: studyID, itemID: itemID)
```

**Step 5: Run test to verify it passes**

Expected: All tests PASS

**Step 6: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVue/Web/NiivueURLRouter.swift \
        NiiVueTests/Web/NiivueURLRouterPreprocessedTests.swift
git commit -m "feat: add preprocessed route to NiivueURLRouter"
```

---

### Task A8: Phase A Integration Test

**Files:**
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Integration/PreprocessingIntegrationTests.swift`

**Step 1: Write integration test**

File: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Integration/PreprocessingIntegrationTests.swift`
```swift
import XCTest
@testable import NiiVue

@MainActor
final class PreprocessingIntegrationTests: XCTestCase {

    func testPreprocessingServiceWithCache() async throws {
        // Arrange
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("IntegrationTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let cache = PreprocessedVolumeCache(baseURL: tempDir.appendingPathComponent("cache"))
        let service = PreprocessingService(cache: cache)

        // Create a test file
        let testFile = tempDir.appendingPathComponent("test.nii.gz")
        try "test volume data".write(to: testFile, atomically: true, encoding: .utf8)

        // Act - First preprocessing (cache miss)
        let result1 = try await service.preprocess(
            studyID: "testStudy",
            itemID: "testItem",
            sourceURL: testFile
        )

        // Assert
        XCTAssertTrue(FileManager.default.fileExists(atPath: result1.outputURL.path))
        XCTAssertGreaterThan(result1.processingTime, 0)

        // Act - Second preprocessing (cache hit)
        let hasCached = await service.hasCachedResult(studyID: "testStudy", itemID: "testItem")
        XCTAssertTrue(hasCached)

        let result2 = try await service.preprocess(
            studyID: "testStudy",
            itemID: "testItem",
            sourceURL: testFile
        )

        // Should return cached result (same URL)
        XCTAssertEqual(result1.outputURL.lastPathComponent, result2.outputURL.lastPathComponent)
    }

    func testAllParameterVariantsAreValid() {
        // Assert all 12 variants can be created
        let variants = ParameterVariants.urinaryTractCT
        XCTAssertEqual(variants.count, 12)

        // Verify each variant has valid parameters
        for variant in variants {
            XCTAssertGreaterThanOrEqual(variant.parameters.lowerPercentile, 0)
            XCTAssertLessThanOrEqual(variant.parameters.upperPercentile, 100)
            XCTAssertGreaterThan(variant.parameters.lowerPercentile, -0.001)
        }
    }
}
```

**Step 2: Run integration test**

Run in Xcode: Product → Test (⌘U)

Expected: All tests PASS

**Step 3: Commit**

Run:
```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue
git add NiiVueTests/Integration/PreprocessingIntegrationTests.swift
git commit -m "test: add Phase A integration tests"
```

---

## Phase A Complete Checkpoint

**Verify Phase A completion:**

```bash
cd /Users/leandroalmeida/niivue/ios/NiiVue

# Run all tests
xcodebuild test -scheme NiiVue -destination 'platform=iOS Simulator,name=iPhone 16 Pro'

# Verify file structure
ls -la NiiVue/Services/Preprocessing/
# Expected:
# - CTPreprocessingParameters.swift
# - ParameterVariants.swift
# - PreprocessedResult.swift
# - PreprocessingService.swift
# - PreprocessedVolumeCache.swift

# Verify test structure
ls -la NiiVueTests/Preprocessing/
# Expected: 5+ test files

# Check git log
git log --oneline -10
```

**Expected state after Phase A:**
- nnUNetPreprocessing package integrated
- All preprocessing models defined
- 12 parameter variants available
- Cache with LRU eviction working
- URL routing for preprocessed volumes

---

## Phase B: Enhanced Segmentation

*(Tasks B1-B3 follow the same TDD pattern. Abbreviated for space.)*

### Task B1: Add 3D Flood Fill to WebViewManager

**Files:**
- Modify: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue/Web/WebViewManager.swift`
- Create: `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVueTests/Web/WebViewManagerSegmentationTests.swift`

**Implementation Summary:**
1. Add `setClickToSegment3DEnabled(_ enabled: Bool)` method
2. Add `setFloodFillConnectivity(_ connectivity: FloodFillConnectivity)` method
3. Add `FloodFillConnectivity` enum (face=6, edge=18, corner=26)
4. Wire to JavaScript via `nv.opts.clickToSegmentIs2D`

---

### Task B2: Add Slice Interpolation to WebViewManager

**Implementation Summary:**
1. Add `interpolateMaskSlices(startSlice:endSlice:method:)` method
2. Add `InterpolationMethod` enum (morphological, intensity, linear)
3. Wire to JavaScript `nv.interpolateMaskSlices()`

---

### Task B3: Add Connected Component Filter

**Implementation Summary:**
1. Add `filterConnectedComponents(minimumVoxels:keepLargestN:)` method
2. Return `ConnectedComponentResult` with statistics
3. Wire to JavaScript `nv.bwlabel()` and filtering logic

---

### Task B4: Add Morphological Operations

**Implementation Summary:**
1. Add `dilateMask(iterations:)`, `erodeMask(iterations:)` methods
2. Add `closeMask(iterations:)`, `openMask(iterations:)` convenience methods
3. Wire to JavaScript morphological operations

---

### Task B5: Add Multi-Label Support

**Implementation Summary:**
1. Add `SegmentationLabel` struct with urinary tract presets
2. Add `setDrawingLabel(_ label:)` method
3. Add `getDrawingLabels()` method to query current labels
4. Wire to JavaScript `nv.opts.penValue`

---

### Task B6: Create SegmentationStore

**Implementation Summary:**
1. Create actor-based SegmentationStore
2. Implement create/save/load/list/delete operations
3. Implement history tracking with snapshots
4. Integrate with StudyStore for `segmentationVolume` type

---

### Task B7: Create PresetStore

**Implementation Summary:**
1. Create PresetStore for user-customizable presets
2. Bundle system presets (12 variants)
3. Add user preset CRUD operations
4. Add default preset tracking

---

## Phase C: User Interface

*(Tasks C1-C3 implement SwiftUI views)*

### Task C1: Create SegmentationToolbar

**Files:**
- Create: `NiiVue/Views/Segmentation/SegmentationToolbar.swift`

---

### Task C2: Create Tool Context Panels

**Files:**
- Create: `NiiVue/Views/Segmentation/Panels/DrawToolPanel.swift`
- Create: `NiiVue/Views/Segmentation/Panels/FillToolPanel.swift`
- Create: `NiiVue/Views/Segmentation/Panels/ThresholdToolPanel.swift`
- Create: `NiiVue/Views/Segmentation/Panels/InterpolateToolPanel.swift`
- Create: `NiiVue/Views/Segmentation/Panels/RefineToolPanel.swift`

---

### Task C3: Create LabelSelectorBar

**Files:**
- Create: `NiiVue/Views/Segmentation/LabelSelectorBar.swift`

---

### Task C4: Enhance SegmentationTabView

**Files:**
- Modify: `NiiVue/Views/Segmentation/SegmentationTabView.swift`

---

### Task C5: Create PreprocessingToolbar

**Files:**
- Create: `NiiVue/Views/Segmentation/PreprocessingToolbar.swift`

---

## Phase D: Parameter Tuning System

### Task D1: Create TuningService

### Task D2: Create SnapshotCaptureService

### Task D3: Create TuningSessionStore

### Task D4: Create TestVolumeRegistry

### Task D5: Create ParameterTuningTabView

---

## Phase E: Integration & Polish

### Task E1: End-to-End Workflow Testing

### Task E2: Performance Optimization

### Task E3: Error Handling & Edge Cases

### Task E4: Documentation

---

## Success Criteria

This implementation is complete when:

- [ ] All Phase A tests pass (preprocessing foundation)
- [ ] All Phase B tests pass (enhanced segmentation)
- [ ] All Phase C UI components render correctly
- [ ] All Phase D tuning system works end-to-end
- [ ] Can load CT → preprocess → segment → export → persist
- [ ] Parameter tuning produces visual comparison gallery
- [ ] All 12 variants produce distinct preprocessing results

---

**Plan Status:** ✅ Ready for Execution

**Last Updated:** 2026-01-11

**Start Point:** Task A1 - Add nnUNetPreprocessing as SPM Dependency
