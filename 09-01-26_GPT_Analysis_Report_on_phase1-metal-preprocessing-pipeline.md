# 09-01-26 — GPT Analysis Report on `Phase 1: Metal Preprocessing Pipeline`

**Audited plan:** `docs/plans/2026-01-09-phase1-metal-preprocessing-pipeline.md`  
**Related codebases reviewed:** `nnunetv2/` (this repo) and `/Users/leandroalmeida/niivue-ios-foundation/iOS` (NiiVue iOS app)  
**Primary objective of the plan (as stated):** “Implement GPU-accelerated preprocessing pipeline for CT DICOM volumes using Metal Performance Shaders, matching nnUNet’s preprocessing behavior exactly.”

---

## Executive Summary (High-Signal Findings)

### P0 (Blocking) Issues — The plan cannot succeed “as written”

1. **The plan’s implementation details don’t match nnUNet’s actual preprocessing pipeline.**
   - nnUNet’s preprocessing order is: **transpose → crop-to-nonzero → normalize → resample** (with special-case logic for 2D configs and anisotropy).
   - The plan implements only **CT normalization** and sketches resampling, but **omits transpose + cropping** and **does not implement nnUNet’s resampling semantics**.

2. **The plan’s Swift/Metal code contains multiple compile-time and runtime bugs.**
   - `actor` isolation is used but the call sites are missing required `await`s.
   - Double/Float size mismatch when uploading constants (`Double` variables are passed with `MemoryLayout<Float>.size`).
   - Tests read `.r16Float` textures as `[Float]` with the wrong byte stride.
   - Output textures are created without `.shaderWrite` usage in tests (likely runtime validation failures).
   - The plan uses `MTLTextureDescriptor.texture3DDescriptor(...)`, which is **not present in Apple’s `MTLTextureDescriptor` doc page** (see Cupertino links below). If this API doesn’t exist in the targeted SDK, the plan won’t compile.

3. **The plan’s “MPSImageLanczosScale” resampling proposal does not match nnUNet’s resampling behavior.**
   - nnUNet uses `skimage.transform.resize(..., order=3, mode='edge', anti_aliasing=False)` for image data, with optional separate-Z resampling (nearest along the anisotropic axis via `scipy.ndimage.map_coordinates`).
   - Lanczos is a different kernel and won’t be “exact,” even if it were implementable for volumes.

4. **The plan’s “CT DICOM volumes” scope conflicts with the current iOS app architecture.**
   - The NiiVue iOS app currently serves **raw DICOM files to a WKWebView** (Niivue JS) via a custom `niivue://` scheme and does not decode DICOM into a native voxel buffer in Swift.
   - A native Metal preprocessing pipeline requires a native voxel buffer source (or a JS→native data bridge), which the plan doesn’t address.

### P1 (High Impact) Issues — Even if P0s are fixed, outcomes may still diverge

- **Cropping-to-nonzero may be a no-op for CT DICOM** if background is not zero (nnUNet’s nonzero mask is `data != 0`). This affects both “exact match” claims and pipeline design.
- **Memory and performance assumptions are unrealistic for 512×512×300 volumes** if represented as 3D textures (hundreds of MB per volume, before intermediate buffers).
- **Parameter extraction is incomplete**: nnUNet plans include resampling function names and kwargs (`resampling_fn_data`, `resampling_fn_data_kwargs`, …) that are required to match behavior but are not extracted.

---

## What I Verified (Ground Truth References)

### nnUNet source of truth (this repo)

- CT normalization semantics:
  - `nnunetv2/preprocessing/normalization/default_normalization_schemes.py` → `class CTNormalization.run(...)`
- Preprocessing order and key steps:
  - `nnunetv2/preprocessing/preprocessors/default_preprocessor.py` → `DefaultPreprocessor.run_case_npy(...)`
- Resampling semantics (including anisotropy + separate-Z logic):
  - `nnunetv2/preprocessing/resampling/default_resampling.py` → `resample_data_or_seg(...)`, `determine_do_sep_z_and_axis(...)`
- Cropping-to-nonzero semantics:
  - `nnunetv2/preprocessing/cropping/cropping.py` → `create_nonzero_mask(...)`, `crop_to_nonzero(...)`
- Plans include resampling function + kwargs:
  - `nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py` → `determine_resampling(...)`, `get_plans_for_configuration(...)`

### iOS app architecture (external path)

The iOS app at `/Users/leandroalmeida/niivue-ios-foundation/iOS/studies-tab-library/NiiVue` currently:

- Stores imported DICOM series as files and serves them to the web layer:
  - `.../Services/StudyStore.swift`
  - `.../Services/DicomSeriesStore.swift`
  - `.../Web/NiivueURLSchemeHandler.swift`
- Does **not** currently use Metal/MPS/CoreML in the Swift layer (no `MTL*`, `MPS*`, or `CoreML` usage found in Swift sources under `NiiVue/NiiVue/`).

### Apple Developer Documentation (Cupertino) I relied on

- `MTLTexture` overview (copying pixel data APIs):
  - `apple-docs://metal/documentation_metal_mtltexture`
- `MTLTextureDescriptor` overview (descriptor creation APIs listed):
  - `apple-docs://metal/documentation_metal_mtltexturedescriptor`
- `MTLTextureType` includes `.type3D`:
  - `apple-docs://metal/documentation_metal_mtltexturetype`
- Shader library creation APIs (including `makeLibrary(source:options:)`):
  - `apple-docs://metal/documentation_metal_shader-library-and-archive-creation`
- Metal in Simulator guidance:
  - `apple-docs://metal/documentation_metal_developing-metal-apps-that-run-in-simulator`
- Resource storage mode guidance:
  - `apple-docs://metal/documentation_metal_setting-resource-storage-modes`
- `MPSImageLanczosScale` API page:
  - `apple-docs://metalperformanceshaders/documentation_metalperformanceshaders_mpsimagelanczosscale`

---

## Section-by-Section Audit of the Phase 1 Plan

Below, I follow the plan’s task structure and call out: **Issue → Why it matters → Concrete fix**.

### Header / Goal / Architecture

**Issue:** The plan claims “matching nnUNet preprocessing behavior exactly,” but its described architecture doesn’t include nnUNet’s full preprocessing steps nor nnUNet’s resampling method.  
**Why it matters:** “Exact match” requires matching (at least) transpose, crop-to-nonzero behavior, normalization, and the exact resampling kernel + coordinate mapping.  
**Fix:** Re-scope Phase 1 explicitly:

- **Phase 1A (Data contract):** define the *native volume representation* (spacing, axis order, intensity units, background conventions).
- **Phase 1B (CPU reference):** implement an exact CPU reference in Swift that matches nnUNet semantics (even if slow), validated against Python nnUNet outputs.
- **Phase 1C (GPU acceleration):** port only after correctness is locked, with tolerances defined and per-step golden tests.

### Prerequisites

**Issue:** The plan’s “Setup Before Starting” clones `https://github.com/your-org/nnUNet-iOS.git` and creates a worktree, but this repo is `nnUNet` (Python).  
**Why it matters:** Execution instructions don’t apply to the current repo and will confuse automation / contributors.  
**Fix:** Move iOS-module execution steps into the iOS repo (or clarify this plan is “for a different repo” and add an explicit dependency).

**Issue:** Tech stack says “Swift 6.2, Metal 4” but the sample `Package.swift` uses `// swift-tools-version: 5.9` and `.iOS(.v17)`.  
**Why it matters:** The plan’s availability assumptions, strict concurrency rules, and APIs may differ across toolchains/targets.  
**Fix:** Align targets and tool versions explicitly (and verify against the real deployment target for the iOS app).

### Task 1: Create Project Structure

**Issue:** `Package.swift` example target platforms do not match plan assumptions (iOS 26).  
**Fix:** Decide a realistic minimum iOS version for the app, then reflect it consistently in:

- SPM manifests
- Xcode project settings (if integrating into `NiiVue.xcodeproj`)
- API choices (Metal 4 vs classic Metal)

### Task 2: Extract Preprocessing Parameters from nnUNet Model

**Issue:** The script assumes `model_dir/nnUNetPlans.json` and `model_dir/dataset_fingerprint.json` exist. In nnUNet v2, these are typically stored under `nnUNet_preprocessed/<Dataset>/` (plans and fingerprint) and not necessarily inside the trained model directory.  
**Why it matters:** Parameter extraction will fail for the most common folder layouts.  
**Fix:** Accept explicit paths:

- `--plans-json /path/to/nnUNetPlans.json`
- `--dataset-fingerprint /path/to/dataset_fingerprint.json`
- `--configuration 3d_fullres|2d|...`

…or accept `--dataset` + `--plans-identifier` and resolve the canonical locations using nnUNet conventions.

**Issue:** Missing extraction of resampling function details.
nnUNet plans include:
- `resampling_fn_data`, `resampling_fn_data_kwargs`
- `resampling_fn_seg`, `resampling_fn_seg_kwargs`
**Why it matters:** These directly define interpolation order, separate-z behavior, and other details. Without them you cannot claim “exact match.”  
**Fix:** Extend JSON export to include those fields (and version them).

### Task 3: Define Swift Models for Preprocessing Parameters

**Issue:** The proposed `AnyCodable` and `foregroundIntensityProperties` usage is internally inconsistent:
- `foregroundIntensityProperties["0"]` is `AnyCodable`, not `[String: Double]`, so the `as? [String: Double]` cast will fail.
- `AnyCodable` is not declared `Sendable`, yet it’s stored in a `Sendable` struct.  
**Fix:** Use strongly typed models for exactly what you need:

- `foregroundIntensityProperties: [String: ForegroundIntensityProperties]`
- `struct ForegroundIntensityProperties: Codable, Sendable { let mean, std, percentile_00_5, percentile_99_5: Double }`

If you truly need arbitrary JSON, make `AnyCodable: Sendable` and provide safe accessors.

### Task 4: Implement CT Normalization Metal Kernel

This task is closest to nnUNet ground truth, but the plan’s Swift and test code has multiple correctness problems.

#### 4.1 Actor usage and call sites

**Issue:** `CTNormalizer` is declared as an `actor`, but the tests call `normalizer.normalize(...)` without `await`.  
**Why it matters:** This won’t compile under Swift concurrency rules.  
**Fix:** Either:
- make `CTNormalizer` a `final class` with explicit synchronization, or
- keep it an `actor` and make the tests use `try await normalizer.normalize(...)`.

#### 4.2 Constant uploads: `Double` vs `Float`

**Issue:** The plan uploads `Double` values with `MemoryLayout<Float>.size`.  
**Why it matters:** This is a data corruption bug (wrong bytes).  
**Fix:** Convert to `Float` before uploading:

- `var mean: Float = Float(properties.mean)` etc.

#### 4.3 Texture usage flags

**Issue:** Output textures in tests are created without setting `descriptor.usage` to include `.shaderWrite`.  
**Why it matters:** Compute kernels require writable textures; you’ll get runtime failures.  
**Fix:** Set `descriptor.usage = [.shaderWrite, .shaderRead]` for outputs (and `.shaderRead` for inputs).

#### 4.4 Reading and writing `.r16Float`

**Issue:** Tests read `.r16Float` textures into `[Float]` with `bytesPerRow = width * 4`, which mismatches FP16 storage (2 bytes/element).  
**Why it matters:** Your assertions will be operating on nonsense data.  
**Fix:** Either:
- Use `.r32Float` output during testing and only switch to `.r16Float` once validated, or
- Read back into `[UInt16]` and convert half→float correctly (or use a blit to a float32 buffer/texture).

#### 4.5 3D texture creation API risk

**Issue:** The plan uses `MTLTextureDescriptor.texture3DDescriptor(...)`. In Apple’s `MTLTextureDescriptor` doc page, the listed convenience constructors are `texture2DDescriptor`, `textureCubeDescriptor`, and `textureBufferDescriptor` (no `texture3DDescriptor`).  
**Why it matters:** If the target SDK doesn’t provide `texture3DDescriptor`, this won’t compile.  
**Fix:** Create a descriptor manually:

- `let d = MTLTextureDescriptor()`
- `d.textureType = .type3D` (see `MTLTextureType.type3D` in Cupertino docs)
- Set width/height/depth/pixelFormat/mipmapLevelCount/usage/storageMode explicitly.

#### 4.6 `replace(region:...)` for 3D textures

**Issue:** The tests use `replace(region:mipmapLevel:withBytes:bytesPerRow:)` for a 3D region. Apple’s `MTLTexture` docs call out the 3D-capable copy methods that include `bytesPerImage` (see the `replace(...bytesPerRow:bytesPerImage:)` overload).  
**Fix:** For 3D textures, use the overload that specifies `bytesPerImage` (or use buffers).

#### 4.7 Redundant shader sources

**Issue:** The plan both creates `CTNormalization.metal` and compiles a shader from an inline Swift string, but never loads the `.metal` file.  
**Fix:** Choose one:

- Preferred: compile `.metal` at build time and load via `makeDefaultLibrary()` / `makeLibrary(URL:)`.
- Keep “compile-from-source” only for prototyping, and document the trade-offs (runtime compile cost, error surfaces).

### Task 5: Implement Resampling with MPSImageLanczosScale

**Issue:** This task is fundamentally incompatible with the plan’s “exact match” goal.

- nnUNet uses cubic interpolation (`order=3`) with `skimage.transform.resize(... mode='edge', anti_aliasing=False)` and special separate-Z behavior using `map_coordinates`.
- Lanczos is not cubic and differs in ringing/edge behavior and coordinate mapping.

**Fix:** To match nnUNet:

1. Implement nnUNet’s resampling semantics exactly (CPU first, then GPU).
2. Mirror nnUNet’s anisotropy decision function (`determine_do_sep_z_and_axis`) rather than using only max/min ratio.
3. Mirror `mode='edge'` boundary handling (critical for medical volumes).

If you still want to use MPS for performance, you must accept that it will be an *approximation* and update success criteria accordingly (or constrain datasets so the difference is negligible).

### Task 6: Create Integration Test with Real nnUNet Parameters

**Issue:** `Bundle.module` is used in the test target, but the plan instructs copying the JSON fixture into the **library** target’s resources. `Bundle.module` in tests refers to the **test module’s** bundle resources.  
**Fix:** Put `preprocessing_params.json` under the test target’s resources or load it via `Bundle(for:)` / explicit file paths in CI.

**Issue:** Random test volumes (`Float.random`) make the test nondeterministic.  
**Fix:** Use a fixed seed + deterministic RNG or a static fixture volume.

**Issue:** The test only checks mean near zero and value bounds; that does not validate “matches nnUNet reference.”  
**Fix:** Generate golden outputs from Python nnUNet for a small fixture input and assert elementwise equality within a defined tolerance.

### Task 7: Add Performance Benchmarks

**Issues:**
- The sample Swift has invalid syntax (`# Large`).
- `XCTAssertLessThan maxValue, 10.0` is missing parentheses.
- Performance assertions like “< 1 second” for very large volumes are not device-scoped and likely meaningless across hardware.

**Fix:** Use XCTest `measure` APIs and record baselines per device class; treat performance gating as CI optional unless you have deterministic CI hardware.

### Task 8: Documentation

**Issue:** The proposed README claims resampling is complete and correct, but Task 5 is explicitly unimplemented.  
**Fix:** Make documentation reflect actual implemented scope; mark resampling as TODO until validated.

### Task 9: Final Validation

**Issue:** References `scripts/validate_preprocessing.py` but the plan never defines it.  
**Fix:** Add a concrete validation harness spec:

- Python script that runs nnUNet preprocessing on fixture inputs and exports intermediate arrays.
- Swift test that loads those arrays and compares after each pipeline stage.

**Issue:** Tagging order: plan tags and pushes before committing `CHANGELOG.md`.  
**Fix:** Commit changelog first, then tag the commit.

---

## Cross-Cutting Gaps (Not Tied to a Single Task)

### 1) Data Contract: DICOM → voxel buffer → “nnUNet space”

The plan assumes “CT DICOM volume input” but doesn’t specify:

- How DICOM slices are sorted (InstanceNumber vs ImagePositionPatient)
- How spacing is derived (PixelSpacing + SliceThickness vs spacing between positions)
- How intensities become Hounsfield Units (RescaleSlope/RescaleIntercept)
- How orientation is standardized (patient coordinate systems, axis conventions)
- How/if background is zeroed (cropping-to-nonzero depends on `data != 0` in nnUNet)

Given the iOS app currently uses Niivue JS to read DICOM, you need an explicit decision:

- **Option A (Native pipeline):** implement DICOM decoding + HU conversion in Swift (then Metal).
- **Option B (Bridge):** extract voxel data from Niivue JS and pass to native for preprocessing (bandwidth + memory heavy).
- **Option C (Preconvert):** convert to NIfTI (or another volume format) before the iOS app sees it.

### 2) “Exact match” tolerance definition

The plan says “<1% difference” but doesn’t define the metric:

- Max absolute error? Mean absolute error? Percent of voxels differing?
- How do FP16 outputs affect tolerance?

To be auditably correct, define:

- Per-stage tolerances (normalization vs resampling)
- Required dtype (float32 vs float16) for equivalence tests
- Acceptance metrics (MAE, RMSE, max error, histogram divergence, etc.)

### 3) Memory and resource strategy

3D textures for 512×512×300 are enormous. Even with FP16, the memory footprint + intermediate buffers can exceed iOS limits and will pressure memory bandwidth.

Recommended direction:

- Keep correctness in float32 for reference.
- For GPU acceleration, consider tiled processing, slice streaming, or buffer-based linear memory (and only use textures where sampling hardware helps).
- Follow Apple’s storage mode guidance (`shared` for CPU/GPU access, `private` for GPU-only, etc.) and minimize CPU readbacks (see Cupertino storage mode article).

---

## Suggested “Fix Plan” (Concrete Next Actions)

If you want Phase 1 to be executable and truthfully validated, I recommend rewriting the Phase 1 plan into three deliverables:

1. **Golden reference harness (Python → fixtures)**
   - Export: input volume (small), nnUNet-normalized output, nnUNet-resampled output, and metadata (spacing, transpose, bbox).
   - Store fixtures in a versioned folder with checksums.

2. **Swift CPU reference implementation (correctness-first)**
   - Implement nnUNet semantics exactly:
     - transpose + spacing reorder
     - crop-to-nonzero + bbox
     - CTNormalization
     - resampling: cubic `order=3`, `mode='edge'`, separate-Z logic
   - Validate against fixtures (tight tolerance).

3. **Metal acceleration (performance pass)**
   - Port CTNormalization first (compute kernel on buffers/textures)
   - Port resampling second, ensuring coordinate mapping matches CPU reference
   - Run A/B comparisons against CPU reference and Python fixtures

This sequencing keeps correctness measurable and makes performance work safer.

---

## bd Follow-ups (Issues to File)

During this audit I created bd issue `nnUNet-f8w` for the overall work. Additional follow-ups that should be tracked as separate issues:

1. **Fix Phase 1 plan compile/runtime errors** (actors/awaits, Float vs Double, texture usage, FP16 readback)
2. **Define and implement “data contract” for CT volumes on iOS** (DICOM→HU→voxel buffer + axis/spacings)
3. **Add extraction + versioning for resampling params from `nnUNetPlans.json`** (including kwargs)
4. **Design and implement nnUNet-equivalent resampling** (cubic + edge mode + separate-Z)
5. **Build a Python↔Swift validation harness with golden fixtures**

---

## Closing Note

The current Phase 1 plan contains good intent (TDD, parameter extraction, kernel isolation), but it currently mixes “prototype scaffolding” with “claims of exact equivalence.” If you align the data contract, extract the full resampling spec from plans, and validate against real nnUNet fixtures, Phase 1 becomes both executable and auditable.

