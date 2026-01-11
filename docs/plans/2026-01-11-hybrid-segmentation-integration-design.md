# Hybrid Segmentation Integration Design

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Maximize current utility of existing nnUNet preprocessing pipeline and Niivue segmentation tools by building end-to-end workflow integration and exposing hidden capabilities, enabling productive urinary tract CT segmentation workflows while the CoreML model is in development.

**Architecture:** Enhanced WebView with Native Preprocessing (Approach C) - Clean architecture where native Swift/Metal preprocessing improves input quality, and all segmentation happens in the proven Niivue WebView with enhanced feature exposure.

**Tech Stack:** Swift 6.2, SwiftUI, Metal, WebKit, nnUNetPreprocessing package

**Primary Target:** Urinary tract CT segmentation

---

## Prerequisites

**Required Skills:**
- @apple-senior-developer - For iOS/SwiftUI/Metal patterns
- @superpowers:test-driven-development - For test-driven development

**Required Dependencies:**
- Phase 1 complete (nnUNet Metal preprocessing pipeline)
- Niivue iOS app at `/Users/leandroalmeida/niivue/ios/NiiVue`
- Test CT volumes at `/Users/leandroalmeida/niivue-ios-foundation/Test_CT_DICOM_volumes`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENHANCED WEBVIEW ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      NATIVE SWIFT LAYER                               │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐   │   │
│  │  │ StudyStore  │───▶│ PreprocessingService │───▶│ ProcessedVolume │   │   │
│  │  │ (DICOM/NIfTI)│    │ (nnUNet Metal)      │    │ Cache           │   │   │
│  │  └─────────────┘    └─────────────────────┘    └────────┬────────┘   │   │
│  │                                                          │            │   │
│  │                      ┌───────────────────────────────────┘            │   │
│  │                      ▼                                                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │              NiivueURLSchemeHandler                              │ │   │
│  │  │   Serves: niivue://preprocessed/{studyID}/{volumeID}            │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      WEBVIEW LAYER (Niivue.js)                        │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐   │   │
│  │  │ Load Volume │───▶│ Segmentation Tools  │───▶│ Export Drawing  │   │   │
│  │  │ (raw or     │    │ (click-to-segment,  │    │ (NIfTI base64)  │   │   │
│  │  │ preprocessed)│    │ Otsu, draw, interp) │    │                 │   │   │
│  │  └─────────────┘    └─────────────────────┘    └────────┬────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      PERSISTENCE LAYER                                │   │
│  │                                                                       │   │
│  │  StudyStore.saveSegmentation(studyID, niftiData) → segmentationVolume │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **PreprocessingService** | Wraps nnUNet Metal pipeline, produces normalized NIfTI |
| **PreprocessedVolumeCache** | Caches preprocessed volumes to avoid recomputation |
| **NiivueURLSchemeHandler** | Extended to serve preprocessed volumes via `niivue://preprocessed/` |
| **WebViewManager** | Extended with new bridge methods for enhanced segmentation |
| **Niivue.js** | Unchanged core, but more features exposed via window functions |

---

## Native Preprocessing Module

### PreprocessingService

```swift
// Location: NiiVue/Services/PreprocessingService.swift

@MainActor
@Observable
public final class PreprocessingService {

    /// User-controllable preprocessing toggle
    public var isEnabled: Bool = true

    /// CT-specific parameters (urinary tract optimized defaults)
    public var ctParameters: CTPreprocessingParameters = .urinaryTractDefaults

    /// Preprocess a volume, returns URL to preprocessed NIfTI
    public func preprocess(
        studyID: String,
        itemID: String,
        sourceURL: URL
    ) async throws -> PreprocessedResult

    /// Check if preprocessed version exists in cache
    public func hasCachedResult(studyID: String, itemID: String) -> Bool

    /// Clear cache for a study
    public func clearCache(studyID: String) async
}
```

### CTPreprocessingParameters

```swift
public struct CTPreprocessingParameters: Codable, Sendable {
    // Normalization
    public var lowerPercentile: Double = 0.5
    public var upperPercentile: Double = 99.5
    public var useZScoreNormalization: Bool = true

    // Resampling
    public var targetSpacing: SIMD3<Double>? = nil
    public var isotropicTarget: Double? = 1.0
    public var interpolationOrder: Int = 3

    // Cropping
    public var cropToNonzero: Bool = true
    public var paddingVoxels: Int = 5

    // Optional fixed HU range clip
    public var customClipRange: (min: Double, max: Double)?
    public var useCustomClipAsPrefilter: Bool = false

    // Presets
    public static let urinaryTractDefaults = CTPreprocessingParameters(
        lowerPercentile: 0.5,
        upperPercentile: 99.5,
        isotropicTarget: 1.0,
        cropToNonzero: true,
        paddingVoxels: 10
    )
}
```

---

## 12 Predefined Parameter Variants

For systematic testing and optimization of urinary tract CT preprocessing:

| # | Name | Description | Key Settings |
|---|------|-------------|--------------|
| 1 | Baseline (Raw) | No intensity changes, only resampling | 0-100%, no z-score, no crop |
| 2 | Standard Balanced | Good starting point | 0.5-99.5%, 1.0mm iso |
| 3 | Conservative Intensity | Preserves subtle contrast | 0.1-99.9% |
| 4 | Aggressive Intensity | Removes outliers | 2.0-98.0% |
| 5 | Maximum Contrast | Very aggressive clipping | 5.0-95.0% |
| 6 | High Resolution (0.5mm) | Better boundary detail | 0.5mm iso |
| 7 | Ultra High Resolution (0.25mm) | Finest detail | 0.25mm iso |
| 8 | Quick Preview (2.0mm) | Fast iteration | 2.0mm iso, linear interp |
| 9 | Contrast-Enhanced CT | For contrast studies | HU: -100 to 400 |
| 10 | Non-Contrast CT | For non-contrast | HU: -150 to 200 |
| 11 | Kidney-Focused | Cortex-medulla differentiation | 0.75mm, HU: -50 to 300 |
| 12 | Ureter-Focused | Thin tubular structures | 0.5mm, HU: 0 to 500 |

---

## Parameter Tuning Test Suite

### Workflow

1. **SELECT TEST VOLUME** - User picks representative urinary tract CT
2. **DEFINE VARIANTS** - Select from 12 predefined + custom
3. **BATCH PROCESS** - Background preprocessing of all variants
4. **CAPTURE SNAPSHOTS** - Axial/coronal/sagittal at anatomical landmarks
5. **PRESENT GALLERY** - Grid view for comparison
6. **HUMAN SELECTION** - User marks winners
7. **SAVE AS PRESET** - Winning parameters become new preset

### Test Volume Location

```
/Users/leandroalmeida/niivue-ios-foundation/Test_CT_DICOM_volumes/
├── Dicom_Volume_1/    (existing)
├── Dicom_Volume_2/    (future)
└── ...
```

---

## Enhanced Segmentation Features

### New WebViewManager APIs

| Feature | Method | Description |
|---------|--------|-------------|
| 3D Flood Fill | `setClickToSegment3DEnabled(_:)` | Fill connected voxels in 3D |
| Connectivity | `setFloodFillConnectivity(_:)` | 6/18/26 neighbor modes |
| Slice Interpolation | `interpolateMaskSlices(start:end:method:)` | Auto-fill between drawn slices |
| Connected Components | `filterConnectedComponents(min:keepN:)` | Remove small regions |
| Morphological Ops | `dilateMask()`, `erodeMask()`, `closeMask()`, `openMask()` | Boundary refinement |
| Multi-Label | `setDrawingLabel(_:)`, `getDrawingLabels()` | Multiple structure support |
| Threshold Preview | `previewIntensityThreshold(lower:upper:)` | See before applying |

### Urinary Tract Label Presets

| Value | Name | Color |
|-------|------|-------|
| 1 | Right Kidney | Red |
| 2 | Left Kidney | Blue |
| 3 | Right Ureter | Orange |
| 4 | Left Ureter | Cyan |
| 5 | Bladder | Yellow |
| 6 | Urethra | Purple |
| 7 | Tumor/Lesion | Green |
| 8 | Other | Gray |

---

## User Interface Design

### Tab Structure

```
┌─────────┐  ┌─────────┐  ┌─────────────┐  ┌─────────┐
│ Viewer  │  │ Studies │  │ Segmentation│  │ Tuning  │
│   🔬    │  │   📁    │  │     ✂️      │  │   ⚙️    │
└─────────┘  └─────────┘  └─────────────┘  └─────────┘
```

### Segmentation Tab Layout

- **Top**: Preprocessing toggle dropdown
- **Center**: Niivue WebView with overlay
- **Toolbar**: [Draw] [Fill] [Threshold] [Interpolate] [Refine] | [Undo] [Redo]
- **Context Panel**: Tool-specific controls (changes per selected tool)
- **Bottom**: Label selector bar with color swatches

### Tool Context Panels

- **Draw**: Brush size, opacity, mode (draw/erase/smart edge)
- **Fill**: 2D/3D mode, connectivity, intensity range
- **Threshold**: Otsu levels, manual range, histogram
- **Interpolate**: Drawn slice detection, method selection
- **Refine**: Morphological ops, component filter, smoothing

---

## Data Persistence

### Directory Structure

```
~/Library/Application Support/NiiVue/
├── Studies/              (existing)
├── Preprocessed/         (NEW - cached preprocessed volumes)
├── Segmentations/        (NEW - saved segmentations with history)
├── TuningSessions/       (NEW - parameter tuning sessions)
└── Presets/              (NEW - user presets)
```

### Key Stores

| Store | Purpose |
|-------|---------|
| **PreprocessedVolumeCache** | Cache with 10GB limit, auto-eviction |
| **SegmentationStore** | Segmentations with full undo history |
| **TuningSessionStore** | Tuning sessions with snapshots |
| **PresetStore** | System + user presets |

---

## Implementation Phases

### Phase A: Foundation
- A1. Package Integration (add nnUNetPreprocessing as SPM dependency)
- A2. PreprocessingService (wrap nnUNet pipeline)
- A3. PreprocessedVolumeCache (caching layer)
- A4. URL Scheme Extension (niivue://preprocessed/ route)

### Phase B: Enhanced Segmentation
- B1. WebViewManager Extensions (all new APIs)
- B2. SegmentationStore (persistence with history)
- B3. PresetStore (user presets)

### Phase C: User Interface
- C1. SegmentationTabView Enhancement (tool panels)
- C2. Preprocessing Toggle UI (dropdown)
- C3. History/Undo UI (visual browser)

### Phase D: Parameter Tuning System
- D1. TuningService (batch processing)
- D2. SnapshotCaptureService (WebView screenshots)
- D3. TuningSessionStore (session persistence)
- D4. TestVolumeRegistry (auto-discovery)
- D5. ParameterTuningTabView (gallery UI)

### Phase E: Integration & Polish
- E1. End-to-End Workflow Testing
- E2. Performance Optimization
- E3. Error Handling & Edge Cases
- E4. Documentation

### Dependency Graph

```
A1 → A2 → A3 → A4
              ↓
        ┌─────┴─────┐
        ↓           ↓
    PHASE B     PHASE D (parallel)
        ↓           ↓
    PHASE C         │
        ↓           │
        └─────┬─────┘
              ↓
          PHASE E
```

---

## Success Criteria

This design is complete when:

- [ ] Preprocessing toggle works with all 12 variants
- [ ] Parameter tuning produces visual comparison gallery
- [ ] All enhanced segmentation features exposed in UI
- [ ] Full workflow: load → preprocess → segment → export → persist
- [ ] Undo/redo history works across sessions
- [ ] Multi-label segmentation with urinary tract presets
- [ ] Performance acceptable on iPad Pro (target device)

---

**Design Status:** ✅ Validated and Ready for Implementation

**Last Updated:** 2026-01-11

**Dependencies:** Phase 1 (nnUNet Metal preprocessing) complete

**Start Point:** Phase A, Task A1 - Package Integration
