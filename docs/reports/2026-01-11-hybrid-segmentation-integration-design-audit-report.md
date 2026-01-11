# Hybrid Segmentation Integration Design — Audit Report (2026-01-11)

**Audited plan:** `docs/plans/2026-01-11-hybrid-segmentation-integration-design.md`  
**Primary targets (per user clarification):**
- **Upgraded Niivue iOS app:** `/Users/leandroalmeida/niivue-ios-foundation/NiiVue`
- **Niivue “non‑iOS” sources to adapt/integrate:** `/Users/leandroalmeida/niivue`

This is a meticulous audit of the design’s technical feasibility, its alignment with the *actual* Niivue iOS codebase, and the realism of the Niivue.js feature exposure claims. It also cross-references Apple platform constraints using official Apple Developer Documentation (via Cupertino MCP).

---

## Executive Summary (High-Signal Findings)

### P0 — Blocking issues (plan not executable “as written”)

1. **Deployment target + toolchain mismatch makes integration impossible without a decision.**
   - `nnUNetPreprocessing/Package.swift` in this repo is **Swift tools 6.2** and targets **iOS 26**:
     - `Package.swift` (this repo): `platforms: [.iOS(.v26), .macOS(.v26)]`
   - The upgraded Niivue iOS app at `/Users/leandroalmeida/niivue-ios-foundation/NiiVue` targets **iOS 16.4**:
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue.xcodeproj/project.pbxproj` contains `IPHONEOS_DEPLOYMENT_TARGET = 16.4;`
   - Result: **You cannot add `nnUNetPreprocessing` as an SPM dependency to that app without either (a) raising the app’s deployment target/toolchain, or (b) lowering the package’s deployment target/toolchain.**

2. **`nnUNetPreprocessing` is not currently “drop-in” as an external SPM dependency due to path-based dependencies.**
   - `Package.swift` (this repo) declares:
     - `.package(path: "../DICOM-Decoder")`
     - `.package(path: "../MTK")`
   - Result: Adding `nnUNetPreprocessing` to `/Users/leandroalmeida/niivue-ios-foundation/NiiVue` as an SPM dependency will fail unless the iOS project has the same sibling directory layout (or the package is refactored to use URL-based dependencies / vendored code).

3. **The plan’s `niivue://preprocessed/...` URL is incompatible with the existing custom-scheme router.**
   - The router in the upgraded iOS app enforces `host == "app"`:
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/NiivueURLRouter.swift` → `guard url.host == "app" else { return nil }`
   - Therefore `niivue://preprocessed/...` (host = `preprocessed`) **will be rejected** unless the router is redesigned.

4. **The “Native Preprocessing → URL to preprocessed NIfTI” output contract is missing required infrastructure.**
   - `nnUNetPreprocessing` currently provides low-level operations (transpose/crop/normalize/resample), but:
     - There is **no high-level pipeline orchestration** (the `Pipeline/` folder exists but is empty).
     - There is **no NIfTI writer** in `Sources/nnUNetPreprocessing` (no `NIfTI` / `.nii` references).
   - The plan requires a *preprocessed NIfTI file URL* to feed Niivue WebView. That requires:
     - a **DICOM → voxel buffer** source in Swift, and
     - a **Float32 → NIfTI(.nii.gz)** encoder, and
     - a **storage + routing strategy** that can serve the produced file via `niivue://app/...`.
   - None of these are specified end-to-end in the design.

5. **The upgraded iOS app’s Swift ↔ JS bridge is currently missing required segmentation entry points referenced by the UI.**
   - `ContentView.swift` uses `webViewManager.clickToSegmentAtScreenPoint(...)`:
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/ContentView.swift` (see the `onOneFingerTap` handler)
   - But `WebViewManager.swift` in the upgraded app does **not** define `clickToSegmentAtScreenPoint` (nor `drawOtsu`, nor native stroke functions):
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/WebViewManager.swift`
   - Separately, the upgraded app’s React bridge does **not** export `window.clickToSegmentAtScreenPoint` either:
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/React/src/App.tsx` exports `window.setClickToSegmentEnabled` but not the click action function.
   - This indicates the base iOS app you want to extend is **not internally consistent** in this area; the plan assumes a stable baseline that does not currently exist.

6. **`CTPreprocessingParameters` in the plan is not Codable as written.**
   - The plan defines:
     - `public var customClipRange: (min: Double, max: Double)?`
   - Swift tuples do not automatically conform to `Codable`; this is a compile-time design bug for a `Codable` struct.

### P1 — High-impact issues (even if P0s are fixed)

- **Crop-to-nonzero is usually ineffective for CT** when background is not exactly `0`. nnUNet’s crop is `data != 0`; typical CT air/background is ~`-1024`, so you often crop nothing.
  - `Sources/nnUNetPreprocessing/CPU/CropToNonzero.swift` scans for `ptr[idx] != 0`.
- **“Undo/redo history persists across sessions” is likely unrealistic** without a size budget + compression strategy. A full 3D uint8 mask can be tens of MB per snapshot (and much more if multiple labels/snapshots).
- **Several “enhanced segmentation” features don’t exist as described in Niivue core today**, and would require non-trivial new algorithms (or repackaging existing ones under different semantics).
- **Parameter tuning with 12 variants + snapshot gallery** will be CPU/GPU intensive and can easily exceed iOS memory limits unless engineered around streaming, caching, and throttled concurrency.

---

## Ground Truth — What the Upgraded Niivue iOS App Actually Does Today

### Custom `niivue://` scheme

- The app registers a custom handler:
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/WebViewManager.swift` calls:
    - `config.setURLSchemeHandler(handler, forURLScheme: "niivue")`
- The URL router is strict:
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/NiivueURLRouter.swift`
    - `guard url.scheme == "niivue" else { return nil }`
    - `guard url.host == "app" else { return nil }`
    - routes only `dist`, `samples`, `files/<id>`, and `dicom/<seriesId>/...`

### Persistent storage patterns (baseline)

- Imported files live under Application Support (library-style):
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Services/ImportedFileStore.swift`
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Services/FileImportService.swift`
- Sessions are persisted as “thin JSON” snapshots:
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Services/SessionStore.swift`
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Services/SessionSnapshotV1.swift`

### Swift ↔ JS evaluation and Promise handling

- The code intentionally uses `callAsyncJavaScript` to await Promise results from the React/Niivue layer:
  - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/WKWebView+JavaScriptEvaluating.swift`
  - Apple doc grounding: `apple-docs://webkit/documentation_webkit_wkwebview_callasyncjavascript_arguments_in_contentworld_e56f98b7`

---

## Ground Truth — What Niivue Core (non-iOS) Can Actually Do (Relevant to This Plan)

The Niivue core sources at `/Users/leandroalmeida/niivue/packages/niivue/src/niivue/index.ts` support several capabilities that map to parts of the design:

- **Flood fill / click-to-segment**
  - `doClickToSegment(...)` computes intensity thresholds and calls `drawFloodFill(...)`.
  - `opts.clickToSegmentIs2D` controls whether fill is constrained to a slice (2D) or allowed through the volume (3D).
  - `opts.floodFillNeighbors` supports connectivity control (`6|18|26`).
- **Mask slice interpolation**
  - `interpolateMaskSlices(...)` exists on the Niivue instance and calls into `src/drawing/masks.ts`.
- **Connected components labeling (for volume images)**
  - `bwlabel(...)` and `createConnectedLabelImage(...)` exist (primarily for *volume* labeling, not directly “filter components of a drawing mask”).
- **Some drawing/mask operations**
  - `drawOtsu(...)` exists.
  - `drawGrowCut(...)` exists.
  - `drawingBinaryDilationWithSeed(...)` exists (seeded dilation on a connected cluster).

But several plan-proposed features are **not present as named capabilities**:

- No general-purpose `erode/open/close` for drawing masks.
- No ready-made “filter connected components in the drawing mask by min size / keepN”.
- No `getDrawingLabels()` API — labels are implicit via `opts.penValue` and the contents of `drawBitmap`.

---

## Apple Developer Documentation — Verified Constraints (via Cupertino MCP)

These are the Apple APIs the plan relies on indirectly (custom scheme, async JS evaluation, snapshots), with official URIs for auditability:

- `WKURLSchemeHandler` (custom scheme loading): `apple-docs://webkit/documentation_webkit_wkurlschemehandler`
- `WKURLSchemeTask` (task interface; response/data/finish/fail): `apple-docs://webkit/documentation_webkit_wkurlschemetask`
- `WKWebViewConfiguration.setURLSchemeHandler(_:forURLScheme:)` (registration rules, scheme constraints): `apple-docs://webkit/documentation_webkit_wkwebviewconfiguration_seturlschemehandler_forurlscheme_a0b3f405`
  - Notable: **“programmer error to call more than once for the same scheme”**.
  - Notable: **scheme name restrictions** (ASCII letter start; allowed chars).
  - Notable: cannot override schemes WebKit already handles (use `WKWebView.handlesURLScheme`).
- `WKWebView.handlesURLScheme(_:)`: `apple-docs://webkit/documentation_webkit_wkwebview_handlesurlscheme_39da21dc`
- `WKWebView.callAsyncJavaScript(...)` (await Promise results): `apple-docs://webkit/documentation_webkit_wkwebview_callasyncjavascript_arguments_in_contentworld_e56f98b7`
  - Notable: **pass function body only** (not a callable wrapper).
  - Notable: if returned object has a callable `then`, WebKit awaits its resolution.
- `WKContentWorld.page` (namespace + collision warning): `apple-docs://webkit/documentation_webkit_wkcontentworld_page`
- `WKWebView.takeSnapshot(...)` + `WKSnapshotConfiguration`:  
  - `apple-docs://webkit/documentation_webkit_wkwebview_takesnapshot_with_completionhandler_ed173371`  
  - `apple-docs://webkit/documentation_webkit_wksnapshotconfiguration`
- `Migrating from ObservableObject → Observable macro` (Observation availability iOS 17+): `apple-docs://swiftui/documentation_swiftui_migrating-from-the-observable-object-protocol-to-the-observable-macro`

Implication for this plan:
- The plan’s choice of **custom URL scheme** and **Promise-aware JS bridge** aligns with Apple’s documented mechanisms.
- The plan’s **SwiftUI `@Observable` usage implies iOS 17+** (and therefore conflicts with the upgraded app’s iOS 16.4 deployment target unless that is raised).

---

## Detailed Audit of the Plan (Section by Section)

### 1) “Prerequisites” section: paths + baseline repo mismatch

The design plan states:
- “Niivue iOS app at `/Users/leandroalmeida/niivue/ios/NiiVue`”

But the clarified source of truth is:
- `/Users/leandroalmeida/niivue-ios-foundation/NiiVue`

**Why this matters:** the design references types/components that exist only in one of these variants (e.g., `StudyStore` and `studies/...` routing exist in one codebase but not the other). If you implement against the wrong baseline, you will produce dead code.

**Fix:** Update the plan’s prerequisites and all file-path annotations to the upgraded codebase. Also add an explicit “baseline inventory” subsection:
- URL scheme endpoints that exist today
- storage model that exists today
- what segmentation bridge functions exist today
- what deployment target + Swift toolchain exists today

### 2) “Architecture Overview”: the diagram assumes components that don’t exist (yet)

The plan’s diagram includes:
- `StudyStore (DICOM/NIfTI)`
- `ProcessedVolume Cache`
- `StudyStore.saveSegmentation(...)`

In the upgraded iOS app, there is:
- **no** `StudyStore` actor (search for `StudyStore` returns nothing under `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue`)
- instead there is:
  - `FileImportService`, `ImportedFileStore` (imported files)
  - `DicomSeriesStore` (in-memory mapping for a series ID to file URLs)
  - `SessionStore` (thin JSON)

**Fix:** Either:

**Option A (recommended for this plan):** Extend the upgraded iOS app to *add* a `StudyStore` layer (bringing it closer to the “studies” architecture you want long-term).  
or  
**Option B:** Rewrite the design so that “StudyStore” is replaced by the existing storage primitives (ImportedFileStore + SessionStore), and treat “study” as a virtual grouping until later.

### 3) Native Preprocessing Module: output contract is under-specified

The plan claims:
- Native preprocessing produces “URL to preprocessed NIfTI”.

But:
- `nnUNetPreprocessing` currently outputs `VolumeBuffer` (raw float data + shape/spacing metadata).
  - Example: `/Sources/nnUNetPreprocessing/CPU/CTNormalization.swift` returns a `VolumeBuffer`.
  - There is no NIfTI writer in `Sources/nnUNetPreprocessing`.

**Hard requirement for the plan to work:** Niivue’s WebView can only load what it can parse (NIfTI/DICOM/etc). If preprocessing is in Swift, you must emit a **real file** the WebView can fetch via the scheme handler.

**Concrete fix path (minimum viable):**
1. Add a `NiftiWriter` (Swift) that can serialize:
   - datatype: `Float32` (or `Int16` if you quantize)
   - dims: `x,y,z` + affine (or at least pixdim + identity affine if acceptable)
   - optionally gzip (`.nii.gz`)
2. Define a storage location:
   - Cache outputs into `Caches` if they’re re-computable (preprocessed “variants”).
   - Persist outputs into `Application Support` if they are user artifacts (exported segmentation, chosen best variant).
3. Add URL routing:
   - EITHER: store preprocessed volumes as imported files and serve them via existing `niivue://app/files/<id>`
   - OR: add a dedicated endpoint `niivue://app/preprocessed/<studyID>/<variantID>/<fileName>` and implement it in:
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/NiivueURLRouter.swift`
     - `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/NiivueURLSchemeHandler.swift`

**Additional required fix (SPM integration reality):** If `PreprocessingService` lives in the iOS app repo and depends on this repo’s `nnUNetPreprocessing`, the design must specify how the iOS app resolves `nnUNetPreprocessing`’s path-based dependencies (`../DICOM-Decoder`, `../MTK`). Without that, Phase A1 (“Package Integration”) is blocked.

### 4) `CTPreprocessingParameters`: compile-time issues + data model mismatch

**Issues in the plan’s struct:**

- `customClipRange` tuple is not `Codable`:
  ```swift
  public var customClipRange: (min: Double, max: Double)?
  ```
  Replace with:
  - `struct ClipRange: Codable, Sendable { let min: Double; let max: Double }`
  - then `public var customClipRange: ClipRange?`

- `SIMD3<Double>?` as `Codable` is a risk (depending on toolchain/SDK). If you need stability:
  - use `[Double]` with exactly 3 elements, or
  - a dedicated codable wrapper type.

- `@Observable` implies Observation (iOS 17+):
  - Apple’s migration doc explicitly states Observation support begins with iOS 17: `apple-docs://swiftui/documentation_swiftui_migrating-from-the-observable-object-protocol-to-the-observable-macro`
  - If the upgraded iOS app remains iOS 16.4, the design must use `ObservableObject` (or raise the deployment target).

**Mismatch with `nnUNetPreprocessing`’s existing models:**

- `nnUNetPreprocessing` already has:
  - `/Sources/nnUNetPreprocessing/Models/CTNormalizationProperties.swift`
  - `/Sources/nnUNetPreprocessing/Models/PreprocessingParameters.swift` (nnUNet plan-derived)
- The design introduces a *parallel* parameter model that is not a superset of those types and doesn’t state how the two relate.

**Fix:** Clarify in the design whether:
- `CTPreprocessingParameters` is an *experimental/tuning overlay* (user-driven, not nnUNet-derived), or
- it *replaces* the plan-derived `PreprocessingParameters`.

If it’s a tuning overlay, document exactly how it maps onto:
- clipping bounds
- normalization mean/std (volume-derived vs dataset-derived)
- resampling order and spacing

### 5) Enhanced Segmentation Features: feature map vs actual Niivue capabilities

Below is a reality map of “plan feature” → “what exists” → “what you actually need to implement”.

| Plan feature | Plan API name | Niivue core reality | Implementation guidance |
|---|---|---|---|
| 3D flood fill | `setClickToSegment3DEnabled(_:)` | Supported via `nv.opts.clickToSegmentIs2D` (set `false` for 3D fill). Seed still comes from a 2D slice click. | Add a JS bridge function to set `nv.opts.clickToSegmentIs2D`. |
| Connectivity 6/18/26 | `setFloodFillConnectivity(_:)` | Supported via `nv.opts.floodFillNeighbors` | Add a JS bridge function to set neighbors; validate values. |
| Slice interpolation | `interpolateMaskSlices(start:end:method:)` | Supported via `nv.interpolateMaskSlices(...)` and `src/drawing/masks.ts` | Bridge to Niivue’s existing API; define “method” mapping to `useIntensityGuided` etc. |
| Connected components filtering | `filterConnectedComponents(min:keepN:)` | Niivue has `bwlabel` for *volume images* and can label clusters. There isn’t a simple “filter drawing mask components” API. | Implement mask-component filtering yourself (likely per-label), or upstream a reusable API into Niivue core. |
| Morphological ops | `dilateMask`, `erodeMask`, `closeMask`, `openMask` | No general versions exist today. Some related tools exist (`drawGrowCut`, `drawingBinaryDilationWithSeed`). | Either implement new ops in JS (CPU heavy) or reduce scope to what exists. |
| Multi-label segmentation | `setDrawingLabel`, `getDrawingLabels` | Labels are implicit in `drawBitmap` values + `nv.opts.penValue`. | UI can track label palette; `setPenValue` already exists in bridge (Swift→JS). |
| Threshold preview | `previewIntensityThreshold(lower:upper:)` | Niivue click-to-segment has preview mode (`clickToSegmentIsGrowing` + `clickToSegmentGrowingBitmap`) but it’s not exposed in the iOS React bridge currently. | Expose preview toggle + apply/cancel semantics through JS bridge. |

**Key missing step in the design:** it doesn’t specify *where* these features get implemented:

- Some are **Niivue core changes** (in `/Users/leandroalmeida/niivue/packages/niivue/...`) that should ship in `@niivue/niivue`.
- Some are **iOS React bridge changes** (in `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/React/src/App.tsx`) that simply expose existing Niivue core capabilities to Swift.
- Some are **Swift WebViewManager changes** (in `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/WebViewManager.swift`) that provide typed APIs to the SwiftUI UI.

### 6) UI Design: tab architecture conflicts with the upgraded app

The plan proposes a 4-tab UI: Viewer / Studies / Segmentation / Tuning.

The upgraded app’s entrypoint is currently a single `ContentView()`:
- `/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/NiiVueApp.swift`

**Fix:** The plan needs to explicitly introduce “Phase 0: App shell”:
- Decide whether to adopt the multi-tab architecture now.
- If yes, define ownership boundaries:
  - a single `WebViewManager` instance shared across tabs
  - a single persistence layer shared across tabs
  - a single preprocessing pipeline service shared across tabs

### 7) Data Persistence: directory structure is plausible but must align with iOS realities

The design uses `~/Library/Application Support/NiiVue/...`. On iOS, you must resolve directories via `FileManager`.

The upgraded app already does this for sessions:
- `SessionStore.defaultSessionsDirectory()` returns:
  - `FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("NiiVue/Sessions")`

**Fix:** Define all plan directories in that same pattern and decide:
- which data is *cacheable* vs *durable*
- storage limits (10GB cache limit is huge on iOS; plan needs an eviction policy + “low storage” behavior)

### 8) Implementation phases: missing critical “bridge + packaging” tasks

The plan’s phases are reasonable conceptually, but it omits critical work items that will otherwise stall implementation:

- **React build + embedding pipeline**: how do JS changes land in the iOS app bundle?
  - The React app depends on `@niivue/niivue@0.66.0` (`/Users/leandroalmeida/niivue-ios-foundation/NiiVue/React/package.json`).
  - If you change `/Users/leandroalmeida/niivue` sources, you must also define:
    - local workspace linking or dependency pinning,
    - rebuilding `dist/`,
    - bundling into the iOS app resources.

- **Fix baseline inconsistencies** in the upgraded iOS app:
  - `ContentView` calls missing `WebViewManager` APIs.
  - React bridge doesn’t export the functions that the Swift layer expects.

---

## “Fix the Design Plan” — Recommended Corrections (Concrete)

Below is a minimal set of edits that would make the design plan internally consistent and executable.

### A) Add a “Compatibility Decision” gate up front (must choose one)

**Decision 1: deployment targets/toolchain**

- **Option A (forward-looking):** Raise `/Users/leandroalmeida/niivue-ios-foundation/NiiVue` to iOS 26 + Swift 6.2, adopt Observation (`@Observable`) where desired, then integrate `nnUNetPreprocessing` unchanged.
- **Option B (pragmatic):** Lower `nnUNetPreprocessing`’s deployment target/toolchain to iOS 16.4-compatible, then integrate without changing the Niivue iOS app target.

The plan should explicitly pick one (or stage them: B now, A later).

### B) Replace `niivue://preprocessed/...` with a router-compatible URL form

Recommended scheme:
- `niivue://app/preprocessed/<studyID>/<variantID>/<fileName>`

…or simplest:
- Store preprocessed outputs in the same “imported files” store and load them as:
  - `niivue://app/files/<id>`

### C) Explicitly introduce `NiftiWriter` (or equivalent) as a required component

Add a “Task 0”:
- “Implement NIfTI(.nii.gz) writer for Float32 volumes (Swift)”

### D) Re-scope “Enhanced Segmentation” into three implementation layers

For every new feature, specify:
1. **Niivue core change** (`/Users/leandroalmeida/niivue/...`) vs
2. **iOS React bridge change** (`/Users/leandroalmeida/niivue-ios-foundation/NiiVue/React/...`) vs
3. **Swift bridge change** (`/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/WebViewManager.swift`)

### E) Fix `CTPreprocessingParameters` Codable correctness

Replace the tuple, and specify codable strategy for any SIMD types.

---

## Appendix: Small, High-Value Code References

### 1) Router host constraint (why `niivue://preprocessed/...` fails)

`/Users/leandroalmeida/niivue-ios-foundation/NiiVue/NiiVue/Web/NiivueURLRouter.swift`:

```swift
guard url.host == "app" else { return nil }
```

### 2) Apple docs: scheme handler registration is single-shot per scheme

`setURLSchemeHandler(_:forURLScheme:)` discussion (Apple doc):
- `apple-docs://webkit/documentation_webkit_wkwebviewconfiguration_seturlschemehandler_forurlscheme_a0b3f405`

> “It is a programmer error to call this method more than once for the same scheme.”

### 3) Niivue click-to-segment uses `clickToSegmentIs2D` to constrain fill

`/Users/leandroalmeida/niivue/packages/niivue/src/niivue/index.ts` (inside `doClickToSegment`):

```ts
this.drawFloodFill(
  [pt[0], pt[1], pt[2]],
  this.opts.penValue,
  brightOrDark,
  this.opts.clickToSegmentIntensityMin,
  this.opts.clickToSegmentIntensityMax,
  this.opts.floodFillNeighbors,
  this.opts.clickToSegmentMaxDistanceMM,
  this.opts.clickToSegmentIs2D,
  targetBitmap
)
```

### 4) `nnUNetPreprocessing` crop semantics (why CT may not crop)

`Sources/nnUNetPreprocessing/CPU/CropToNonzero.swift`:

```swift
if ptr[idx] != 0 {
  // expands bbox
}
```

---

## Next Step (Recommended)

Before implementing any Phase A work from the design plan, do these two “alignment” actions first:

1. **Decide iOS deployment target strategy** (raise app vs lower package).
2. **Define the canonical “preprocessed volume” file contract** (NIfTI writer + routing + caching).

Once those are decided, the remainder of the plan can be rewritten into an executable, testable implementation plan (task-by-task) that matches the upgraded Niivue iOS app structure.
