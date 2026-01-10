# 09-01-26 — GPT Analysis Report on `phase2-ios-segmentation-module`

This report audits `docs/plans/2026-01-09-phase2-ios-segmentation-module.md` section-by-section and task-by-task, cross-checked against:

- nnUNet v2 source in this repo (`nnunetv2/`) — especially inference + export semantics.
- The Niivue iOS app sources at `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue`.
- Apple’s official documentation (grounded via Cupertino MCP; URIs included for auditability).

Assumption: **Phase 1 is already fixed** per `09-01-26_GPT_Analysis_Report_on_phase1-metal-preprocessing-pipeline.md`, meaning preprocessing outputs are *nnUNet-equivalent* and include all metadata required to invert preprocessing (transpose/crop/resample) during postprocessing.

---

## Executive Summary (What’s Broken / Risky)

The Phase 2 plan is directionally correct (Core ML inference + UI + WebView overlay), but **it’s not executable as written**. The largest issues are:

1. **nnUNet inference parity is missing**: The plan doesn’t implement nnUNet’s sliding-window inference, optional TTA mirroring, fold ensembling, or the critical *postprocessing inversion* (resample logits → argmax → reinsert crop → inverse transpose). Without these, output won’t match nnUNet, and overlays will misalign.
2. **The “DICOM → tensor” bridge is still undefined**: The existing Niivue iOS app loads DICOM into the WebView (Niivue JS) via a custom URL scheme; it does *not* provide a native voxel buffer. The plan stubs this out as `Data` and never resolves it into a 3D tensor, which blocks on-device inference.
3. **Swift code samples contain compile-time and runtime errors** (e.g., `Identifiable` conformance broken, wrong `StudyStore` mocking, incorrect SwiftUI `Alert` usage, missing `try await`, incorrect WebViewManager API names, and broken `niivue://` URL construction).
4. **Concurrency model is unsafe for performance**: Putting the entire `SegmentationService` on `@MainActor` guarantees that heavy preprocessing/inference/postprocessing can block UI unless explicitly offloaded.
5. **WebView integration is specified incorrectly** relative to the real app: the app already has `niivue://app/...` routing with path traversal protection and specific endpoints; the plan’s “file path → niivue://” conversion won’t route.

If executed naively, the result will be a Segmentation tab that “looks like progress exists” but can’t reliably run inference, can’t produce nnUNet-equivalent segmentations, and can’t load results into Niivue in a stable, persistent way.

---

## Ground Truth Reference (What nnUNet Actually Does in Phase 2)

nnUNet inference and export include several non-optional steps that must be matched (or consciously deviated from with documented accuracy impact):

- **Sliding-window inference over patches** with:
  - step size (default `tile_step_size = 0.5`)
  - optional Gaussian importance weighting for blending patch logits
  - optional mirroring Test-Time Augmentation (TTA)
  - optional fold ensembling (averaging logits from multiple trained folds)
  - input is **4D tensor** `(c, x, y, z)` post-preprocessing
  - output is logits in preprocessed space
  - Source: `nnunetv2/inference/predict_from_raw_data.py` (`nnUNetPredictor.predict_sliding_window_return_logits`, `_internal_maybe_mirror_and_predict`)

- **Export/postprocessing inversion**:
  1. resample logits back to `shape_after_cropping_and_before_resampling` using the configuration’s probability resampling function
  2. convert logits → segmentation (argmax or region logic)
  3. insert segmentation into `shape_before_cropping` using stored bbox
  4. inverse transpose back to original axis order
  - Source: `nnunetv2/inference/export_prediction.py` (`convert_predicted_logits_to_segmentation_with_correct_shape`)

The Phase 2 plan currently implements none of these mechanics beyond placeholders.

---

## Ground Truth Reference (What the Niivue iOS App Actually Provides Today)

Key facts from `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue`:

- `StudyStore` is an **actor** that persists studies and items to Application Support, and already defines `StudyItemKind.segmentationVolume`:
  - `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue/Services/StudyStore.swift`

- The app uses a **custom `niivue://` URL scheme** to serve local resources to `WKWebView` via `WKURLSchemeHandler`:
  - router validates `scheme == "niivue"`, `host == "app"`, then routes `niivue://app/studies/<studyID>/items/<itemID>/<fileName>` (and other endpoints):
    - `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue/Web/NiivueURLRouter.swift`
  - scheme handler serves `Studies` payload files from Application Support and DICOM manifests/files for Niivue JS:
    - `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue/Web/NiivueURLSchemeHandler.swift`

- `WebViewManager` already supports:
  - `loadVolumesFromUrls`, `addVolumesFromUrls`, `loadMeshesFromUrls`
  - per-volume `setColormap`, `setOpacity`
  - (separate) segmentation/draw tooling for Niivue JS
  - `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue/NiiVue/Web/WebViewManager.swift`

Implication: Phase 2 should **not invent a new file routing scheme**; it should store outputs as `StudyItemRecord`s (or imported files) and pass correct `niivue://app/...` URLs to `WebViewManager`.

---

## Apple Documentation Grounding (Cupertino URIs)

Core ML model lifecycle and configuration:

- `MLModelConfiguration` overview (compute units, GPU precision): `apple-docs://coreml/documentation_coreml_mlmodelconfiguration`
- `MLModelConfiguration.computeUnits`: `apple-docs://coreml/documentation_coreml_mlmodelconfiguration_computeunits`
- `MLComputeUnits` (including `.all`, `.cpuOnly`, etc): `apple-docs://coreml/documentation_coreml_mlcomputeunits`
- `MLModelConfiguration.allowLowPrecisionAccumulationOnGPU`: `apple-docs://coreml/documentation_coreml_mlmodelconfiguration_allowlowprecisionaccumulationongpu`
- `MLModel.compileModel(at:)` (async, yields `.mlmodelc`): `apple-docs://coreml/documentation_coreml_mlmodel_compilemodel_at_f7319eb1`
- “Downloading and Compiling a Model on the User’s Device”: `apple-docs://coreml/documentation_coreml_downloading-and-compiling-a-model-on-the-user-s-device`
- `MLMultiArray` shape/constraints and memory access: `apple-docs://coreml/documentation_coreml_mlmultiarray`

SwiftUI modals/alerts and deprecations:

- “Modal presentations” (alerts are presented via modifiers; `Alert` is listed under deprecated modal presentations): `apple-docs://swiftui/documentation_swiftui_modal-presentations`

Observation / `@Observable` guidance:

- “Migrating from the ObservableObject protocol to the Observable macro”: `apple-docs://swiftui/documentation_swiftui_migrating-from-the-observable-object-protocol-to-the-observable-macro`

WKWebView custom scheme loading:

- `WKURLSchemeHandler` protocol overview: `apple-docs://webkit/documentation_webkit_wkurlschemehandler`
- `WKWebViewConfiguration.setURLSchemeHandler(_:forURLScheme:)`: `apple-docs://webkit/documentation_webkit_wkwebviewconfiguration_seturlschemehandler_forurlscheme_a0b3f405`

---

## Section-by-Section Audit of the Phase 2 Plan

### Header / Goal / Architecture / Tech Stack

**Issues**

- “SwiftUI-based parallel native module” is underspecified. The existing app is SwiftUI + a `WebViewManager` bridging into Niivue JS; there’s no established native ML inference subsystem to “parallelize” against, and “parallel” here must not mean “run everything on MainActor”.
- Tech stack lists **SceneKit** and **Combine**, but the plan’s concrete tasks largely don’t use them. This is scope noise and causes confusion (Phase 3 owns mesh generation; Phase 2 should decide whether meshes are rendered by Niivue JS or SceneKit).
- “Core ML 5” as a version label is not actionable; use “Core ML” and then verify API availability for the app’s deployment target.

**Fix Approach**

- Define the module boundary explicitly: a single `SegmentationPipeline` abstraction with:
  - `prepareInput(studyID/itemID) -> Tensor + Metadata`
  - `infer(patch) -> logits` (Core ML)
  - `stitch(logitsPatches) -> logitsVolume`
  - `invertTransformsAndWriteSegmentation(...) -> StudyItemRecord`
  - `publishToViewer(...)` (WebViewManager overlay)
- Decide and document: **Phase 2 outputs a segmentation volume overlay** only; mesh generation is Phase 3.

---

### Prerequisites

**Issues**

- “Trained nnUNet model converted to .mlmodel” is treated as done, but conversion is a major engineering deliverable. The plan lacks:
  - conversion recipe, operator support constraints (3D conv), and fallback strategy
  - model input/output contract (shape order, dtype, normalization expectations)
  - packaging strategy (bundled vs on-demand download)
- The app path is correct, but the plan treats it as if it were inside this repo. It’s a separate project with existing tests and patterns that must be followed (e.g., there is already segmentation-related code for importing masks/meshes).

**Fix Approach**

- Add a prerequisite deliverable section:
  - “Model Contract Spec” (input name(s), output name(s), axis order, patch size, dtype, value range)
  - “Conversion + Validation Harness” (Python produces a golden inference output for a fixture volume; Swift compares)
  - “Packaging” (bundle `.mlmodelc` or download `.mlmodel` and compile via `MLModel.compileModel(at:)`)

---

## Task-by-Task Audit

### Task 1: Create Segmentation Module Structure

**Issues**

- Shell commands are incorrect: `cd NiiVue/Segmentation` assumes directory already exists. It should create `NiiVue/Segmentation` from the project root (`mkdir -p NiiVue/Segmentation/...`).
- Xcode group steps are fine, but the plan doesn’t mention target membership and ensuring files are included in the correct build targets and test targets.

**Fix Approach**

- Correct shell commands and add explicit Xcode checks:
  - new files appear under app target
  - tests appear under `NiiVueTests` / `NiiVueUITests`

---

### Task 2: “Define Core Data Models” (actually value models)

**Issues**

- Mislabeling: these are **plain Swift structs**, not Core Data models.
- `SegmentationResult` claims `Identifiable` but defines `taskID` instead of `id` → compile error.
- `meshURL` is `URL?` in test, but task initialization uses non-optional `meshURL: URL(string: "...")!` in the test sample (inconsistency with plan text elsewhere).
- `@objc public enum SegmentationStatus: Int, Sendable` is likely fine, but `@objc` isn’t necessary here unless Objective-C interop is required.

**Fix Approach**

- Rename section to “Define Segmentation Value Types”.
- Fix `Identifiable`:
  - either rename `taskID` → `id`, or add `var id: UUID { taskID }`.
- Add explicit model contract fields that Phase 2 needs:
  - input spacing, original spacing, transpose axes, bbox used for cropping, shapes before/after transforms (since Phase 1 is “fixed”, these should exist and be re-used)
  - label mapping / class count / model output semantics (argmax vs probabilities)

---

### Task 3: Implement Core ML Model Manager

**Issues**

- “Discover models” is hard-coded to two names; it doesn’t actually discover resources in the bundle or in Application Support.
- `Bundle.main.url(forResource:withExtension: "mlmodel")` is not a supported runtime loading strategy unless you compile it first. Apple docs emphasize that model instances are created from **compiled models** (`.mlmodelc`) and compilation can be done with `MLModel.compileModel(at:)`.
- The plan caches `UrinaryTractSegmentation` instances, but the type name implies a specific auto-generated wrapper class. If the model name changes, this breaks.
- Model compute policy is asserted as `.all` and “Use Neural Engine + GPU” — but that’s not guaranteed; Core ML may choose a backend based on operator support.
- Uses `import os.log` and `Logger(...)` — the existing app uses `print` and doesn’t standardize on OSLog. (This is not inherently wrong, but mismatched style and may not compile as written depending on imports.)

**Fix Approach**

- Define a model registry strategy:
  - bundled default model (`.mlmodelc`) for offline, plus optional downloaded updates in Application Support
  - verify and store compiled model URLs (see: `apple-docs://coreml/documentation_coreml_downloading-and-compiling-a-model-on-the-user-s-device`)
- Avoid tying to the generated wrapper type:
  - use `MLModel` directly + a typed input/output wrapper you own, OR
  - standardize the wrapper class name and enforce it in conversion tooling.
- Use `MLModelConfiguration` explicitly and document what you verified:
  - computeUnits and allowLowPrecisionAccumulationOnGPU are real (`apple-docs://coreml/documentation_coreml_mlmodelconfiguration`)
- Add a contract check using `MLModel.modelDescription` and `MLMultiArray` constraints (`apple-docs://coreml/documentation_coreml_mlmultiarray`):
  - validate input feature name and shape constraints at startup (fail fast with clear error).

---

### Task 4: Implement SegmentationService

**Issues**

- `SegmentationService` is marked `@MainActor`: this is a major performance footgun. Inference and large-volume memory operations must not run on the main thread.
- The test uses `MockStudyStore` but the initializer requires `StudyStore` (the real type is an actor, not a protocol). As written, this will not compile.
- `loadDICOMData` returns `Data` and throws “not implemented”. Even assuming Phase 1 fixed preprocessing exists, Phase 2 must consume **a tensor + metadata**, not opaque `Data`.
- `runInference` returns `MLFeatureProvider`, which is too generic for postprocessing; you need a well-defined output (logits buffer, probabilities, or argmax mask) with shape/dtype.
- Postprocessing is hand-waved (“connected components, morphological operations”) but nnUNet’s required inversion is missing (see Ground Truth section).
- Output storage writes `Data` to `.nii.gz` path — but `.nii.gz` is a structured file with header + gzip; “writing arbitrary Data” won’t produce a valid NIfTI segmentation file.

**Fix Approach**

- Concurrency design:
  - make `SegmentationService` a plain type or an actor that does heavy work off-main, and only publishes state updates to the UI via `@MainActor` view model.
- Replace `Data` placeholders with explicit types:
  - `PreprocessedVolume` (float buffer + shape + spacing + Phase1 metadata)
  - `LogitsVolume` or `SegmentationMask` (uint8/uint16)
- Implement **nnUNet-equivalent inference strategy**:
  - If the Core ML model accepts a fixed patch size: implement sliding window patch extraction + stitching.
  - Decide whether to include mirroring TTA and fold ensembling; if not, document the accuracy impact and update “Success Criteria”.
- Implement **nnUNet-equivalent postprocessing inversion**:
  - resample logits to pre-resample crop shape (Phase 1 metadata provides target shapes/spacings)
  - argmax into label map
  - reinsert into full-volume bbox
  - inverse transpose to original orientation
  - write valid NIfTI (or another format Niivue can load) with correct affine/orientation.

---

### Task 5: Create Segmentation Tab View

**Issues**

- The UI test expects a “Run Segmentation” button, but `SegmentationTabView` doesn’t contain one.
- `.task { await initialize() }` calls `await viewModel.initialize()` but `initialize()` is `async throws` → the plan’s code sample is missing `try` (`try await`).
- Model selection sheet has a Cancel button with no dismissal; selecting a model doesn’t dismiss the sheet either.
- `SegmentationViewModel` builds its own `StudyStore` and `CoreMLModelManager` rather than using the app’s instances. This breaks the shared `StudyRoots` selection logic used for UI tests and duplicates state.
- The view lists studies but doesn’t integrate with the existing “Studies” tab semantics (load/delete flows) and doesn’t pass results to the viewer.

**Fix Approach**

- Inject dependencies from `AppRootView`:
  - pass `studyStore` and `webViewManager` into `SegmentationTabView` (same pattern as `ContentView` and `StudiesTabView`).
- Add a clear call-to-action:
  - “Run Segmentation” button becomes enabled only when a study is selected and model is ready.
- Use SwiftUI modal APIs correctly:
  - alerts/sheets via modifiers per `apple-docs://swiftui/documentation_swiftui_modal-presentations`.
- Align state management with Observation guidance:
  - if using `@Observable`, align with Apple’s migration guidance (`apple-docs://swiftui/documentation_swiftui_migrating-from-the-observable-object-protocol-to-the-observable-macro`).

---

### Task 6: Integrate with WebViewManager

**Issues**

- The plan uses `updateColormap`, but the real API is `setColormap(volumeIndex:colormap:)` in `WebViewManager`.
- The plan’s extension references `logger`, but `WebViewManager` has no logger property.
- The plan’s `convertToNiivueURL` produces `niivue:<path>` without host routing; the app’s router requires `niivue://app/...` and explicitly rejects invalid hosts and path traversal.
- Loading segmentation output should be done through existing, already-tested mechanisms:
  - For persisted study items: `niivue://app/studies/<studyID>/items/<itemID>/<fileName>`
  - For imported “library files”: `niivue://app/files/<id>`

**Fix Approach**

- Do not create a generic “file path to niivue:// URL” converter.
- Instead, create a `SegmentationResultPublisher` that:
  1. writes output file into `StudyStore`’s item payload directory (or imported library)
  2. records a new `StudyItemRecord` of kind `.segmentationVolume` (requires adding “append item to existing study” support to StudyStore)
  3. generates a correct `niivue://app/studies/...` URL and calls `webViewManager.addVolumesFromUrls(...)`
- Ensure any custom-scheme expectations align with Apple’s `WKURLSchemeHandler` model:
  - see `apple-docs://webkit/documentation_webkit_wkurlschemehandler` and `apple-docs://webkit/documentation_webkit_wkwebviewconfiguration_seturlschemehandler_forurlscheme_a0b3f405`.

---

### Task 7: Add Error Handling and User Feedback

**Issues**

- `SegmentationErrorAlert: View` returns `Alert(...)` from `body`. That is not how SwiftUI presents alerts; alerts are presented via modifiers. Apple explicitly frames modals as view modifiers, and lists `Alert` under deprecated modal presentations.
- Progress overlay as a full-screen ZStack is fine, but it must be wired to actual cancellation semantics (cancelling inference and cleaning temp files).

**Fix Approach**

- Replace custom `Alert` view with:
  - `.alert(isPresented:error:...)` or `.alert(_:isPresented:presenting:...)` (see list in `apple-docs://swiftui/documentation_swiftui_modal-presentations`)
- Add a cancellation mechanism:
  - store the inference `Task` in view model and cancel it, ensuring Core ML requests stop and partial artifacts are cleaned up.

---

### Task 8: Integration Testing

**Issues**

- Tests are mostly `XCTSkip`, so they don’t validate correctness. For a medical segmentation pipeline, tests need to validate:
  - shape correctness
  - transform inversion correctness
  - deterministic file output validity (NIfTI readable by Niivue)
  - (ideally) numeric parity against a known-good python/nnUNet fixture

**Fix Approach**

- Add a minimal golden-fixture workflow:
  - Python: take a small fixture volume, run nnUNet preprocessing + inference + postprocessing; export logits or segmentation.
  - Swift: run Phase1 preprocessing + Phase2 inference + postprocessing, compare:
    - segmentation label counts and voxelwise overlap (Dice)
    - or if full parity is too strict (compute unit differences), compare coarse metrics and accept tolerance.

---

### Task 9: Documentation

**Issues**

- The documentation claims features that Phase 2 doesn’t implement (e.g., mesh generation and “2–3 seconds” performance) and doesn’t specify the model contract.
- “Model Requirements” is vague; it needs explicit input shape ordering and dtype details.

**Fix Approach**

- Update docs to:
  - clearly state whether inference is patch-based
  - declare model contract (input name, shape order, dtype, expected value range)
  - define supported max volume sizes / memory requirements
  - clarify that mesh generation is Phase 3.

---

### Task 10: Final Validation and Release

**Issues**

- “Coverage >70%” is arbitrary and likely unrealistic early, especially when Core ML models and WebView are involved.
- Tagging `v2.0.0` is fine, but the plan doesn’t mention App Store constraints (model size, on-device storage).
- Performance targets (“<3 sec”, “<1 GB”) are not derived from measured benchmarks; they must be device-class specific and verified with Instruments.

**Fix Approach**

- Replace success criteria with measurable, testable gates:
  - “Produces a valid overlay that loads in Niivue and aligns to source volume”
  - “Meets memory budget on target devices (A17/M-series) with documented fallback on lower devices”
  - “Inference correctness validated against fixture volumes”

---

## Cross-Cutting Gaps (Must Address Before Implementation)

### 1) The Model Conversion + Operator Support Gap

The plan assumes a `.mlmodel` exists and is performant on-device. In practice:

- You must prove the converted model runs with acceptable speed and memory on iOS.
- For 3D U-Nets, operator support can force CPU fallback.

**Remediation**

- Add a conversion pipeline + a “capability matrix”:
  - which layers are supported on Neural Engine vs GPU vs CPU
  - which computeUnits settings are allowed for correct/fast execution

### 2) The Inference Strategy Gap (Sliding Window)

nnUNet’s default inference is sliding-window + blending (plus optional TTA and ensembling).

**Remediation**

- If Core ML model is patch-based: implement patch extraction + stitching in Swift, ideally GPU-accelerated (Metal) or at least vectorized CPU.
- If Core ML model accepts full volume: enforce strict maximum volume sizes and verify performance; otherwise implement chunking anyway.

### 3) The Output Validity Gap (Writing `.nii.gz`)

“Write bytes to `.nii.gz`” does not create a valid NIfTI+gzip file.

**Remediation**

- Add a NIfTI writer (minimal: header + raw bytes + optional gzip) or output uncompressed `.nii` if acceptable to Niivue.
- Ensure the output orientation/affine aligns with the loaded source volume in Niivue.

### 4) The StudyStore Mutation Gap

StudyStore currently can’t append new items to an existing study (no `addItem` API).

**Remediation**

- Extend `StudyStore` actor with:
  - `appendItem(studyID:kind:displayName:payloadFileName:...) -> StudyItemRecord`
  - update `index.json` and the study’s `itemIDs`.

### 5) MainActor Saturation Risk

Core ML calls may be async, but surrounding data prep and postprocessing must be off-main.

**Remediation**

- Keep UI state on `@MainActor`, but run heavy work in detached tasks, returning progress via `AsyncStream` or callbacks onto MainActor.

---

## Recommended “Correct” Phase 2 Architecture (Actionable)

If Phase 1 is fixed, Phase 2 should be reframed around one “pipeline contract”:

1. **Input acquisition**: from `StudyStore` item → produce voxel buffer + metadata (or require that Phase1 already produced preprocessed tensor stored as a study item).
2. **Inference**:
   - Core ML model manager loads compiled model (`.mlmodelc`)
   - patch-based sliding window inference (if required), optional TTA/ensemble config
3. **Postprocessing inversion**:
   - resample logits back to crop space
   - argmax / label selection
   - reinsert crop into full image
   - inverse transpose
4. **Persistence**:
   - write segmentation volume as a `StudyItemKind.segmentationVolume` into study payload
5. **Visualization**:
   - `webViewManager.addVolumesFromUrls([(url: "niivue://app/studies/...", name: "...")])`
   - apply suitable colormap/opacity via `setColormap`/`setOpacity`

This architecture aligns with both nnUNet semantics and the iOS app’s existing storage and URL scheme routing.

---

## Suggested Follow-up Work Items (Bd Issues)

This audit strongly suggests filing (or keeping) distinct work items:

- **Model conversion + contract**: define exact Core ML IO shapes/dtypes and conversion tooling; validate on target devices.
- **Sliding window inference**: implement patch extraction + stitching (with optional TTA/ensemble flags).
- **Postprocessing inversion**: implement nnUNet-equivalent export semantics and NIfTI writer.
- **StudyStore extension**: add “append segmentation to existing study” APIs.
- **WebView overlay**: load segmentation via correct `niivue://app/studies/...` URL; add tests around URL routing/spec generation.

---

## Bottom Line

The Phase 2 plan needs a major revision before it can be executed. The highest leverage fixes are:

1. Replace placeholder `Data` flows with explicit tensor + metadata types.
2. Implement nnUNet-equivalent inference + postprocessing inversion (or explicitly document deviations and accept the accuracy impact).
3. Align WebView integration with the existing `niivue://app/...` routing and existing `WebViewManager` APIs.
4. Fix Swift compile issues and restructure concurrency to keep heavy work off the MainActor.

With those changes, Phase 2 becomes implementable and testable in a way that can plausibly meet “on-device segmentation” requirements.

