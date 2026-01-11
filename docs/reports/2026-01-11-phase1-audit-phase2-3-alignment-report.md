# Phase 1 Audit + Phase 2/3 Alignment Report
**Project:** nnUNet iOS/macOS + Niivue iOS integration  
**Date:** 2026-01-11  
**Scope:** Validate Phase 1 implementation against Phase 1 plan/report + ground truth; verify runtime/test robustness (Metal + SwiftPM resources); verify iOS device XCUIAutomation; align Phase 2 & 3 plans to the corrected Phase 1 contracts.

---

## 0) Executive Summary

Phase 1 is functionally complete and fixture-validated, but the audit found several correctness/robustness gaps that would have caused either silent correctness drift (rounding, origin mapping) or reliability issues (Metal shader loading in SwiftPM test environments, command-buffer completion handler race). These were fixed in `nnUNetPreprocessing` and validated via existing test logs; Niivue device UI tests were also stabilized and now pass reliably on Leandro’s physical device.

The Phase 2 and Phase 3 plans need alignment in three major areas:

1. **Contract alignment:** axis/spacing/origin conventions + reversible preprocessing context for correct overlay in the viewer.  
2. **Platform alignment:** deployment targets between Niivue app and `nnUNetPreprocessing` (and its dependency `MTK`) must be made consistent before integration.  
3. **Implementation correctness:** several Phase 3 code stubs/examples would not compile or would be incorrect on iOS as written (e.g., wrong integer table types, `NSColor` usage, incomplete triTable).

---

## 1) What Was Audited

### Phase 1 inputs
- Plan: `docs/plans/2026-01-10-phase1-metal-preprocessing-pipeline-v2.md`
- Execution report (as-is): `docs/reports/2026-01-11-phase1-complete-implementation-report.md`
- Implementation: `Sources/nnUNetPreprocessing/**` + `Tests/nnUNetPreprocessingTests/**`
- Evidence logs already captured in-repo:
  - `docs/reports/2026-01-11-phase1-audit-swift-test-v5.log`
  - `docs/reports/2026-01-11-phase1-audit-swift-build-release-v3.log`

### iOS XCUIAutomation (device)
- Niivue iOS app repo: `/Users/leandroalmeida/niivue/ios` (Xcode project at `/Users/leandroalmeida/niivue/ios/NiiVue/NiiVue.xcodeproj`)
- Passing device test log: `docs/reports/2026-01-11-phase1-audit-niivue-uitests-device-v18.log`
- Example destination: `id=00008140-001664420413C01C` (Leandro’s iPhone)

---

## 2) Apple-Docs Grounding (Cupertino URIs)

These sources were used to validate correct API usage and best practices:

- SwiftPM resources + `Bundle.module`: `swift-evolution://SE-0271`
- Metal shader library loading options: `apple-docs://metal/documentation_metal_shader-library-and-archive-creation`
- `MTLCommandBuffer.addCompletedHandler(_:)` semantics: `apple-docs://metal/documentation_metal_mtlcommandbuffer_addcompletedhandler_c6bbaa92`
- Threadgroup sizing guidance: `apple-docs://metal/documentation_metal_calculating-threadgroup-and-grid-sizes`
- Metal library context (`default.metallib` and runtime compilation): `apple-docs://metal/documentation_metal_metal-libraries`
- Core ML model configuration option used in Phase 2 plan: `apple-docs://coreml/documentation_coreml_mlmodelconfiguration_allowlowprecisionaccumulationongpu`
- Core ML multiarray shape/stride concepts (Phase 2 integration): `apple-docs://coreml/documentation_coreml_mlmultiarray`
- SwiftUI localization behavior for `Text("literal")` vs `Text(StringVariable)` (device UI test fix): `apple-docs://swiftui/documentation_swiftui_localizedstringkey`
- Model I/O export path for STL/OBJ (Phase 3 export plan): `apple-docs://modelio/documentation_modelio_mdlasset_export_to_a095613d` and `apple-docs://modelio/documentation_modelio_mdlasset_canexportfileextension_dc4e8f17`

---

## 3) Phase 1 Audit Findings (nnUNetPreprocessing)

### 3.1 Fix: Metal shader library loading in SwiftPM tests (and general robustness)

**Problem:** In practice, `swift test` can run in contexts where a default library isn’t discoverable via the simplest “load default library from bundle” approach, which can cause Metal tests to skip/fail even when resources exist.

**Fix:** Added a robust loader that:
- Prefers loading a bundled `.metallib` from `Bundle.module` when present
- Falls back to `device.makeDefaultLibrary(bundle: .module)` (SwiftPM bundle pattern from SE-0271)
- Falls back to `device.makeDefaultLibrary()` (main bundle) as a last resort
- In `DEBUG`, can compile bundled `.metal` sources at runtime as a test/dev fallback

**Changed/added:**
- Added: `Sources/nnUNetPreprocessing/Metal/PreprocessingShaderLibraryLoader.swift`
- Updated: `Sources/nnUNetPreprocessing/Metal/MetalCTNormalizer.swift`
- Updated: `Sources/nnUNetPreprocessing/Metal/MetalResampler.swift`

**Docs grounding:** Metal library creation options are enumerated in Apple docs (`apple-docs://metal/documentation_metal_shader-library-and-archive-creation`), and SwiftPM bundle access is defined in SE-0271 (`swift-evolution://SE-0271`).

---

### 3.2 Fix: `addCompletedHandler` ordering to avoid a race

**Problem:** Registering completion handlers after committing a command buffer can create an avoidable race (especially for very small workloads) where completion happens before handler registration.

**Fix:** Register `commandBuffer.addCompletedHandler { … }` before `commandBuffer.commit()` in:
- `MetalCTNormalizer.normalize`
- `MetalResampler.resampleCubic3D` and `resampleSeparateZ`

**Docs grounding:** `addCompletedHandler(_:)` is explicitly about GPU completion callbacks (`apple-docs://metal/documentation_metal_mtlcommandbuffer_addcompletedhandler_c6bbaa92`).

---

### 3.3 Fix: Banker's rounding for target shape (Python/NumPy parity)

**Problem:** nnUNet’s shape computations follow NumPy’s rounding semantics; NumPy uses bankers rounding (ties-to-even). Swift’s default `rounded()` uses ties-to-away-from-zero unless you specify otherwise. This can cause shape drift in edge cases (exact `.5` cases).

**Fix:** Use `.rounded(.toNearestOrEven)` for computing `targetShape` in both CPU and Metal resamplers.

**Changed:**
- Updated: `Sources/nnUNetPreprocessing/CPU/Resampling.swift`
- Updated: `Sources/nnUNetPreprocessing/Metal/MetalResampler.swift`
- Added test: `Tests/nnUNetPreprocessingTests/CPUPreprocessingTests.swift` (`testResamplingTargetShapeUsesBankersRounding`)

---

### 3.4 Fix: DICOM origin mapping (avoid axis swap)

**Problem:** The DICOM-origin mapping in `DicomBridge` was swapped, producing inconsistent world-coordinate origins downstream.

**Fix:** Map origin directly as `origin: volume.origin` (x,y,z patient space).

**Changed:**
- Updated: `Sources/nnUNetPreprocessing/Bridge/DicomBridge.swift`

---

### 3.5 Package hygiene: SwiftPM “unhandled file” warnings

**Problem:** SwiftPM requires each file in a target directory tree to be assigned a rule (compile/copy/process/exclude). Markdown helper files (`CLAUDE.md`) inside `Sources/` and `Tests/` trigger warnings/errors unless excluded.

**Fix:** Explicitly exclude these from targets.

**Changed:**
- Updated: `Package.swift`

**Docs grounding:** SE-0271 notes SwiftPM emits errors for files without rules and discusses resource scoping (`swift-evolution://SE-0271`).

---

### 3.6 Documentation inconsistency found

`docs/reports/2026-01-11-phase1-complete-implementation-report.md` contains an outdated **Known Limitations** section claiming “deferred to Tasks 7–9” and “22/22 passing” even though the report header claims Phase 1 completed and the suite is larger.

**Status:** Corrected in this audit by removing the stale “deferred” section and updating the test summary to reflect the current suite.

---

## 4) iOS Device XCUIAutomation Audit (Niivue)

### 4.1 Symptom
Device UI test `testSegmentationOtsuIncreasesDrawSum` repeatedly failed: Otsu segmentation modified voxels (confirmed via debug payload), but the SwiftUI label for `niivue.drawing.drawSum` stayed at `0`.

### 4.2 Root cause (high confidence)
SwiftUI `Text("\(someInt)")` can end up treated as a localized string key path rather than a verbatim string in some contexts; UI tests relying on the label value can become flaky/non-updating. Apple’s `LocalizedStringKey` docs explain the literal vs variable behavior (`apple-docs://swiftui/documentation_swiftui_localizedstringkey`).

### 4.3 Fixes applied in Niivue iOS repo (summary)
- Use a verbatim/string value for the debug label: `Text(verbatim: String(webViewManager.drawingDrawSum))`
- Improve UI-test determinism:
  - Use `WKWebsiteDataStore.nonPersistent()` in `--ui-test` mode
  - Cache-bust the start URL in `--ui-test` mode
  - Ensure main-thread updates for published UI-test instrumentation state
- Improve JS→Swift debug payload: include numeric `drawSum` field in `otsuDebug`

### 4.4 Evidence
- Passing run: `docs/reports/2026-01-11-phase1-audit-niivue-uitests-device-v18.log`
- Earlier failing runs (for history): `docs/reports/2026-01-11-phase1-audit-niivue-uitests-device-v6.log` … `v17.log`

---

## 5) Phase 2 Plan Alignment Notes (Segmentation Module)

Plan: `docs/plans/2026-01-09-phase2-ios-segmentation-module.md`

### 5.1 Immediate corrections applied to the plan file
- Updated Niivue path to `/Users/leandroalmeida/niivue/ios/NiiVue`
- Updated UI test launch argument to `--ui-test` (matching Niivue’s UI-test mode)

### 5.2 Required alignment work before executing Phase 2

1. **Deployment target alignment (must decide up-front).**  
   - Niivue app target currently reports `IPHONEOS_DEPLOYMENT_TARGET = 16.4`.  
   - `nnUNetPreprocessing` currently declares iOS 26+/macOS 26+ in `Package.swift`, and it depends on `MTK` which declares iOS 17+/macOS 14+.  
   **Action:** Choose one path before integrating:
   - Raise Niivue’s deployment target (at least iOS 17 to satisfy `MTK`, and/or higher if you intentionally want iOS 26-only), or
   - Lower `nnUNetPreprocessing` platform requirements to match its dependency floor (iOS 17+/macOS 14+) and then decide whether Niivue should also move to iOS 17+.

2. **Define and enforce the Phase 1 ↔ Phase 2 preprocessing contract.**  
   Phase 2 must treat these as non-negotiable:
   - Array axis order is `D×H×W` (depth=z, height=y, width=x).
   - Spacing is stored as `(z, y, x)` in `VolumeBuffer`.
   - Origin is stored as `(x, y, z)` patient-space.
   **Action:** Add a “Contract” section to the Phase 2 plan and add unit tests that validate shape/spacing/origin mapping before inference.

3. **Add a reversible preprocessing context (required for correct overlay).**  
   Cropping + transpose + resample must be invertible for postprocessing segmentation back into original DICOM space (for display and export).  
   **Action:** Define a `PreprocessingContext` (or similar) that stores:
   - original shape + spacing + origin + orientation
   - `transposeForward` + `transposeBackward`
   - crop `bbox`
   - target spacing used for inference  
   Then require Phase 2 to apply inverse operations to the model output.

4. **Core ML IO shape/stride must be explicit and tested.**  
   Use model description constraints and construct `MLMultiArray` with correct shape/strides (see `apple-docs://coreml/documentation_coreml_mlmultiarray`).  
   **Action:** Add a “Model IO Contract” task in Phase 2 that loads the model, reads `modelDescription`, and asserts the exact expected input/output multiarray shapes.

5. **Avoid plan-level stubs that hide missing integration points.**  
   Several Phase 2 plan steps currently “throw not implemented” for DICOM loading, preprocessing, and inference.  
   **Action:** Convert these to explicit subtasks with acceptance criteria + tests (unit and integration) so progress is measurable.

---

## 6) Phase 3 Plan Alignment Notes (3D Mesh Generation + Visualization)

Plan: `docs/plans/2026-01-09-phase3-3d-mesh-generation.md`

### 6.1 Immediate corrections applied to the plan file
- Updated Niivue path to `/Users/leandroalmeida/niivue/ios/NiiVue`
- Fixed SceneKit availability command to `swift -e 'import SceneKit; …'`

### 6.2 Critical issues to fix in the Phase 3 plan before execution

1. **Marching cubes lookup table types are wrong in the plan.**  
   The plan defines `edgeTable: [UInt8]` but uses values like `0x109` (>255). This won’t compile and is semantically wrong.  
   **Action:** Plan should specify `UInt16`/`Int` tables and include the *complete* `triTable` (or adopt a vetted implementation).

2. **iOS vs macOS type errors in sample code.**  
   The plan uses `NSColor` in iOS-targeted code and returns mismatched types for material/shader helpers.  
   **Action:** Make the plan explicitly iOS-first and use `UIColor` (or SwiftUI `Color`) where appropriate.

3. **Export should use Model I/O (recommended).**  
   Model I/O supports exporting `.obj` and `.stl`, and chooses format based on file extension (`apple-docs://modelio/documentation_modelio_mdlasset_export_to_a095613d`, `apple-docs://modelio/documentation_modelio_mdlasset_canexportfileextension_dc4e8f17`).  
   **Action:** Prefer `MDLAsset.export(to:)` over manual string building for correctness and maintainability.

4. **World-space scaling is mandatory.**  
   Mesh vertices must incorporate voxel spacing and origin/orientation so that 3D render/export matches patient space.  
   **Action:** Add a “Coordinate Mapping” task requiring explicit conversion from `(z,y,x)` voxel indices into patient/world space using `spacing`, `origin`, and `orientation`.

5. **Multi-label segmentation support must be planned.**  
   nnUNet urinary tract segmentation is not purely binary; per-label meshing + coloring is required for a usable surgical-planning viewer.  
   **Action:** Update plan to generate one mesh per label with deterministic IDs and materials.

---

## 7) Recommended Next Actions

1. Fix the stale “Known Limitations” section in `docs/reports/2026-01-11-phase1-complete-implementation-report.md` to avoid conflicting guidance.
2. Decide and document the deployment-target strategy for Phase 2 integration (Niivue + `nnUNetPreprocessing` + `MTK`).
3. Add “inverse preprocessing” and “model IO contract” as explicit Phase 2 tasks with tests.
4. Rewrite Phase 3 marching cubes plan section to include correct tables/types and iOS-safe sample code; adopt Model I/O for exports.
