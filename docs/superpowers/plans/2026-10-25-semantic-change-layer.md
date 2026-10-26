# Semantic Change Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the app's per-frame change masks into a live list of object events (moved with displacement and yaw, added, removed, altered as a hint) shown in the Live view, the Detect results and the 3D viewer.

**Architecture:** A pure-Swift `ObjectTracker` in OSCDCore back-projects each frame's fused mask with LiDAR depth (things that arrived) or rendered reference depth (things that left) into 1 cm voxels in the before-scan's world frame, tracks signed 3D instances across frames, matches left↔arrived instances by SAM2 features, colour and size, and fits a planar ICP for the move. The app feeds it from the fused callbacks in the Live view (own thread, back-pressure) and from a post-pass over the final masks in Detect, then draws the events as list rows and projected arrows. Methods A and B are untouched.

**Tech Stack:** Swift 6 toolchain in language mode 5, XCTest, SwiftUI `Canvas`, Apple Vision (`VNClassifyImageRequest`), ARKit LiDAR, the existing splatx `ReferenceDepth`, Core ML SAM2 encoder.

**Spec:** `docs/superpowers/specs/2026-10-25-semantic-change-layer-design.md` (§ references below are to it). Read it first.

**Starting point:** branch `oscd-mobile/phase1`. The working tree already carries uncommitted phase 2 edits under `oscd_mobile/`; commit only the files each task names. All paths below are relative to `oscd_mobile/` unless they start with `docs/`.

## Global Constraints

- OSCDCore targets iOS 17 / macOS 14 and must not import splatx, Vision, UIKit or ARKit; everything that does lives in `SpikeApp/Sources/`.
- Methods A and B keep their outputs bit for bit: `CuesATests`, `CuesBTests`, `DetectTests`, `PreviewTests` must still pass after every task; parity fixtures are not touched.
- World frame: the before-scan's ARKit world, +y up (§3). Poses are world→camera in OpenCV axes (`Pose`), images are sensor orientation 480×360 landscape, displayed rotated `.right`.
- SAM2 embeddings are `[256, 64, 64]` planar floats; a pixel (x, y) at W×H maps to cell `(y·64/H)·64 + x·64/W` (as `CuesA.feature` does).
- Defaults from the spec: voxel 1 cm; sign tolerance 5 mm; component floor 40 px; overlap 0.3; confirmation 3 frames and 15 voxels; ROI edge margin 2 cm; drop an unconfirmed instance after 10 observed frames with ≥ 5 misses; reservoir 2 000 points; match score `0.5·cos + 0.3·hist + 0.2·size`, accept ≥ 0.55, low 0.40–0.55; ICP 12 yaw candidates, 20 iterations, trim 80 %, yaw unknown when the best and median candidate RMS differ by < 10 %; Vision label kept at ≥ 0.3, hint-styled below 0.6; per-frame semantic work ≤ 60 ms median.
- Tests run with `cd Packages/OSCDCore && swift test --scratch-path ~/Library/Caches/oscd-mobile/OSCDCore-build --filter <Class>` (the checkout is in iCloud; see README).
- Phone: iPhone 17 Pro, device id `00008150-0011681A3430C01C`, bundle `com.veripheye.changes3d`, scheme `OSCDSpike` (`cd SpikeApp && xcodegen generate`, then build with `-derivedDataPath ~/Library/Caches/oscd-mobile/DerivedData`). Check device behaviour with launch arguments and pulled files, not just a build. Read the build log for warnings as well as errors.
- Commit messages: one imperative sentence, as the repo does ("Add …", "Fix …"), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A frame whose LiDAR depth is all NaN (confidence low everywhere, lens covered): the sign is 0 everywhere, no instance is created and nothing divides by zero. Test in Task 1 (`testSignIsZeroWithoutDepth`) and Task 2 (`testFrameWithoutDepthAddsNothing`).
2. Matching with no `left` or no `arrived` instance: the assignment is empty and every instance becomes `added` / `removed` without an index error. Test in Task 3 (`testEmptySideGivesNoMatches`).
3. ICP on a pair with fewer than 10 points on either side, or two identical point sets: `fit` returns nil, and the event is still `moved` with `motion == nil`. Test in Task 4 (`testTooFewPointsGivesNil`) and Task 3 (`testMovedWithoutMotionIsStillMoved`).
4. An instance centroid behind the camera or far outside the image when drawing arrows (z ≤ 0): the arrow is skipped, no NaN reaches `Canvas`. Test in Task 7 (`testProjectionBehindCameraIsNil`).
5. A saved Detect run with no `objects.json` (runs made before this feature) or a corrupt one: the results screen shows no Objects section and the viewer shows no overlay. Test in Task 3 (`testReportRoundTripsThroughJSONAndFolder`, its missing and corrupt cases) and on the phone in Task 11 step 2.

---

### Task 1: Depth sign, signed components, pooled descriptors (OSCDCore)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/FrameContext.swift`
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/ChangeComponents.swift`
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/Descriptors.swift`
- Modify: `Packages/OSCDCore/Sources/OSCDCore/Cues/CuesB.swift:67-85` (extract the gain step)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/ChangeComponentsTests.swift`

**Interfaces:**
- Produces:
  ```swift
  public struct FrameContext: Sendable {
      public var frameID: Int
      public var pose: Pose; public var intrinsics: Intrinsics
      public var changed: [Bool]        // fused mask ∧ valid, W·H
      public var valid: [Bool]
      public var dLive: [Float]         // W·H metres, NaN where untrusted
      public var dRen: [Float]          // W·H metres, NaN where the reference is uncovered
      public var live: FloatImage; public var render: FloatImage      // 3 planes, W×H
      public var embLive: [Float]; public var embRender: [Float]      // 256·64·64
      public init(...)                  // memberwise
  }
  public struct PixelRect: Equatable, Sendable { public var minX, minY, maxX, maxY: Int }
  public enum ChangeSign {
      /// d = dLive − dRen − medianOffset over pixels where both are finite and valid; NaN elsewhere.
      /// sign = +1 if d < −tolerance (arrived), −1 if d > tolerance (left), else 0.
      public static func compute(dLive: [Float], dRen: [Float], valid: [Bool], tolerance: Float = 0.005) -> (sign: [Int8], d: [Float])
  }
  public struct Component2D: Sendable {
      public var sign: Int8; public var pixels: [Int]; public var rect: PixelRect
      public var feature: [Float] = []; public var histogram: [Float] = []
  }
  public enum ChangeComponents {
      public static func extract(changed: [Bool], sign: [Int8], d: [Float], width: Int, height: Int, minPixels: Int = 40) -> [Component2D]
  }
  public enum SAM2Pool { public static func mean(embedding: [Float], pixels: [Int], width: Int, height: Int, grid: Int = 64, channels: Int = 256) -> [Float] }
  public enum LabHistogram {
      public static let bins = 256     // L 4 × a 8 × b 8; L in [0,100], a,b in [−100,100]
      public static func compute(image: FloatImage, pixels: [Int], gain: SIMD3<Float> = SIMD3(repeating: 1)) -> [Float]   // sums to 1
      public static func intersection(_ a: [Float], _ b: [Float]) -> Float
  }
  public extension CuesB {
      /// The per-channel median gain aligning the render to the live frame (the first step of colorCue).
      static func renderGain(live: FloatImage, render: FloatImage, liveDepth: [Float], renderDepth: [Float], valid: [Bool], params: Params = Params()) -> SIMD3<Float>
  }
  ```
- `extract` algorithm (the tests do not determine it): 4-connected components over `changed`. For each component, mass⁺ = Σ|d| over its sign>0 pixels, mass⁻ over sign<0, and `signedShare` = (pixels with sign ≠ 0) / pixels. If `signedShare < 0.3` → one component with sign 0. Else if `min(mass⁺, mass⁻) / (mass⁺ + mass⁻) ≥ 0.2` → re-run 4-connected components separately over the component's sign>0 and sign<0 pixels (its sign-0 pixels are dropped) and emit each. Else → one component whose sign is the larger mass. Components under `minPixels` are dropped at the end.

- [ ] **Step 1: Write the failing tests** in `ChangeComponentsTests.swift`:
  - `testSignUsesMedianOffsetAndTolerance`: 10×1 arrays, dRen all 0.400, dLive = 0.410 everywhere except pixel 3 = 0.380 and pixel 7 = 0.440 → offset 0.010, sign[3] == 1, sign[7] == −1, all others 0; d[3] ≈ −0.030 (accuracy 1e-5).
  - `testSignIsZeroWithoutDepth`: dLive all NaN → every sign 0, every d NaN.
  - `testSeparateBlobsKeepTheirSign`: 20×10 image, a 4×4 block of sign +1 (d = −0.05) at x 2–5 and a 4×4 block of sign −1 (d = 0.05) at x 12–15, both changed → 2 components, signs {+1, −1}, 16 pixels each, rects (2,2,5,5) and (12,2,15,5).
  - `testTouchingMixedBlobIsSplitBySign`: one 8×4 changed block whose left half has sign +1 and right half −1 (equal |d|) → 2 components of 16 pixels, one per sign.
  - `testMostlyFlatBlobIsFlat`: 6×6 changed block, 4 of 36 pixels with sign +1 → one component with sign 0.
  - `testSmallComponentsAreDropped`: a 5-pixel blob with `minPixels: 40` → no components; with `minPixels: 5` → one.
  - `testPoolMapsPixelsToCells`: W 480, H 360, embedding where cell (row 63, col 63) of channel 0 is 1 and everything else 0; pixels = [359·480 + 479] → feature[0] == 1, feature[1] == 0; pixels = [0] → feature[0] == 0.
  - `testHistogramSumsToOneAndIntersects`: a uniform red image → `compute` sums to 1 (accuracy 1e-5); intersection with itself 1; with a uniform blue image's histogram 0.
  - `testRenderGainMatchesColorCueStep`: live = render × (1.2, 1.0, 0.8) with agreeing depths over 600 pixels → gain ≈ (1.2, 1.0, 0.8) (accuracy 0.01); and `CuesB.colorCue` on the same inputs still returns all zeros after the refactor.
- [ ] **Step 2: Run** `swift test --filter ChangeComponentsTests` → FAIL (types not found).
- [ ] **Step 3: Implement** the types and functions above; move `colorCue`'s gain block into `renderGain` and call it from `colorCue`.
- [ ] **Step 4: Run** `swift test --filter "ChangeComponentsTests|CuesBTests"` → PASS.
- [ ] **Step 5: Commit** `Add signed change components and pooled descriptors for the object layer`.

---

### Task 2: Voxel instances and incremental association (OSCDCore)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/ObjectTracker.swift`
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/ObjectTrackerTests.swift`

**Interfaces:**
- Consumes: Task 1's `FrameContext`, `ChangeSign.compute`, `ChangeComponents.extract`, `SAM2Pool.mean`, `LabHistogram.compute`, `CuesB.renderGain`; `AABB` and `Intrinsics.unproject` / `Pose.inverse` from OSCDCore.
- Produces:
  ```swift
  public struct TrackerConfig: Sendable {
      public var voxelSize: Float = 0.01; public var minPixels = 40; public var overlapMin: Float = 0.3
      public var confirmFrames = 3; public var confirmVoxels = 15; public var roiEdgeMargin: Float = 0.02
      public var signTolerance: Float = 0.005; public var dropAfterObserved = 10; public var dropMinMisses = 5
      public var reservoir = 2000; public var matchAccept: Float = 0.55; public var matchLow: Float = 0.40
      public var roi: AABB?
      public init()
  }
  public struct VoxelKey: Hashable, Sendable { public var x, y, z: Int32; public init(_ p: SIMD3<Float>, size: Float) }
  public enum FeatureSource: String, Codable, Sendable { case render, live, beforePhoto }
  public struct TrackedInstance: Identifiable, Sendable {
      public let id: Int; public var sign: Int8
      public var voxels: [VoxelKey: Int]; public var frames: Set<Int>
      public var feature: [Float]; public var featureSource: FeatureSource; public var histogram: [Float]
      public var points: [SIMD3<Float>]                 // reservoir
      public var bestFrame: Int; public var bestRect: PixelRect; public var bestPixelCount: Int
      public var bestCrop: FloatImage?                  // live (sign ≥ 0) or render (sign < 0) crop of bestRect with 20 % margin
      public var firstSeen: Int; public var lastSeen: Int; public var misses: Int; public var observed: Int
      public var label: String?; public var labelConfidence: Float?
      public var voxelCount: Int { voxels.count }
      public var centroid: SIMD3<Float>                 // hit-weighted voxel centres
      public var bbox: AABB
      public func isConfirmed(_ c: TrackerConfig) -> Bool   // frames.count ≥ confirmFrames && voxelCount ≥ confirmVoxels
  }
  public struct IngestStats: Sendable { public var components: Int; public var merged: Int; public var created: Int; public var dropped: Int; public var milliseconds: Double }
  public final class ObjectTracker {
      public init(config: TrackerConfig = TrackerConfig())
      public let config: TrackerConfig
      public private(set) var instances: [TrackedInstance]
      @discardableResult public func ingest(_ f: FrameContext) -> IngestStats
      public func setFeature(instance id: Int, _ feature: [Float], source: FeatureSource)
      public func setLabel(instance id: Int, _ label: String, confidence: Float)
  }
  ```
- `ingest` order: sign → components → descriptors (feature from `embLive` for sign ≥ 0, `embRender` for sign < 0; histogram from `live`, or from `render` with `renderGain` for sign < 0) → back-project pixels (`dLive` for sign ≥ 0, `dRen` for sign < 0; skip non-finite) → voxel set → merge rule `|overlap| / min(|new|, |instance|) ≥ overlapMin` with instances of the **same sign** (merge all that qualify into the lowest id) → absorb `sign 0` instances overlapping a signed one → drop instances whose bbox comes within `roiEdgeMargin` of any face of `config.roi` (when set) → bookkeeping: an instance whose centroid projects inside the image at a `valid` pixel this frame is `observed += 1`, and `misses += 1` if no component merged into it; drop unconfirmed instances with `observed ≥ dropAfterObserved && misses ≥ dropMinMisses`. Reservoir sampling for `points`. `bestCrop` updated when a component's pixel count beats `bestPixelCount`.

- [ ] **Step 1: Write a tiny fixture helper** `TestFrames.swift` in the test target: `func frame(id: Int, width: 32, height: 24, pose: Pose, f: Float = 30, dLive: (Int, Int) -> Float, dRen: (Int, Int) -> Float, changed: (Int, Int) -> Bool, roi: AABB? = nil) -> FrameContext` that fills `live`/`render` with a flat grey, embeddings with zeros except a constant per-test vector, and `valid` all true.
- [ ] **Step 2: Write the failing tests** in `ObjectTrackerTests.swift`:
  - `testArrivedPixelsBackProjectWithLiveDepth`: camera at identity looking down +z, table at dRen 0.50 m, a 6×6 arrived block at the image centre with dLive 0.45 → one instance, sign +1, centroid z ≈ 0.45 (accuracy 0.01), `voxelCount ≥ 1`.
  - `testLeftPixelsBackProjectWithRenderedDepth`: same block with dLive 0.50 and dRen 0.45 → sign −1, centroid z ≈ 0.45.
  - `testSameSignOverlappingComponentsMerge`: two frames from poses 2 cm apart seeing the same arrived block → `instances.count == 1`, `frames == [0, 1]`, stats.merged == 1 on the second frame.
  - `testDifferentSignsNeverMerge`: an arrived block and a left block at the same world voxels (different frames) → 2 instances.
  - `testConfirmationNeedsThreeFramesAndFifteenVoxels`: after 2 frames `isConfirmed == false`; after the 3rd `true`; a 2×2 block (≤ 4 voxels) seen 5 times stays unconfirmed.
  - `testFlatInstanceIsAbsorbedByOverlappingSigned`: a sign-0 component over the same voxels as an arrived instance → still 1 instance, sign +1.
  - `testInstanceTouchingRoiEdgeIsDropped`: roi whose max z face lies 1 cm beyond the block → 0 instances; roi 10 cm beyond → 1.
  - `testUnconfirmedInstanceIsDroppedAfterMisses`: an instance seen once, then 10 frames that look at it (valid pixels) without change → 0 instances.
  - `testFrameWithoutDepthAddsNothing`: dLive all NaN, changed everywhere → 0 instances, stats.components == 0.
  - `testBestCropFollowsLargestView`: frame 0 shows 16 px of the object, frame 1 shows 36 px → `bestFrame == 1`, `bestCrop` non-nil with width ≥ 6.
- [ ] **Step 3: Run** `swift test --filter ObjectTrackerTests` → FAIL.
- [ ] **Step 4: Implement** `ObjectTracker.swift`.
- [ ] **Step 5: Run** `swift test --filter ObjectTrackerTests` → PASS; `swift test` (whole package, fixture tests skip) → PASS.
- [ ] **Step 6: Commit** `Track signed 3D change instances across frames in ObjectTracker`.

---

### Task 3: Events, matching and the report format (OSCDCore)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/ObjectEvents.swift`
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/Hungarian.swift`
- Modify: `Packages/OSCDCore/Sources/OSCDCore/Objects/ObjectTracker.swift` (matching and events)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/ObjectEventsTests.swift`

**Interfaces:**
- Produces:
  ```swift
  public struct Placement: Codable, Sendable { public var centroid: SIMD3<Float>; public var bbox: AABB; public var voxels: Int; public var frames: [Int] }
  public struct Motion: Codable, Sendable {
      public var translation: SIMD3<Float>; public var horizontalDistance: Float; public var yawDeg: Float?
      public var fitRMS: Float; public var inlierFraction: Float
  }
  public struct ObjectEvent: Codable, Identifiable, Sendable {
      public enum Kind: String, Codable, Sendable { case moved, added, removed, altered }
      public var id: Int               // left id for moved/removed, arrived id for added, instance id for altered
      public var kind: Kind; public var confidence: Float; public var hint: Bool      // hint == true only for altered
      public var label: String?; public var labelConfidence: Float?
      public var before: Placement?; public var after: Placement?; public var motion: Motion?
      /// "Ball · moved 14 cm · turned 30°" (rules below)
      public func summary(matchAccept: Float = 0.55) -> String
  }
  public struct ObjectReport: Codable, Sendable {
      public var beforeID: String; public var afterID: String?; public var method: String
      public var worldFrame = "before-scan ARKit world, +y up"; public var voxelSizeM: Float
      public var events: [ObjectEvent]; public var generatedAt: Date; public var timingsMs: [String: Double]
      public static let fileName = "objects.json"
      public func save(in afterScan: URL) throws                       // DetectionSummary.folder(in:beforeID:method:)/objects.json
      public static func load(in afterScan: URL, beforeID: String, method: String) -> ObjectReport?   // nil when missing or undecodable
  }
  public struct Match: Equatable, Sendable { public var leftID: Int; public var arrivedID: Int; public var score: Float }
  public enum Hungarian { /// maximises Σ score; rows → column index or nil. Empty rows or columns → all nil / [].
      public static func assign(score: [[Float]]) -> [Int?] }
  public enum InstanceMatcher {
      public static func score(_ l: TrackedInstance, _ a: TrackedInstance) -> Float   // 0.5·cos + 0.3·hist + 0.2·size
  }
  extension ObjectTracker {
      public func matches() -> [Match]                      // confirmed left × arrived, score ≥ config.matchLow
      public func setMotion(leftID: Int, arrivedID: Int, _ motion: Motion?)   // cached; cleared when either instance grows > 20 % in voxels
      public func pairsNeedingMotion() -> [Match]            // matched pairs with no cached motion
      public func events() -> [ObjectEvent]                  // ordered by firstSeen of the instance named by `id`
      public func report(beforeID: String, afterID: String?, method: String, timingsMs: [String: Double]) -> ObjectReport
  }
  ```
- Confidence: `moved` → the match score; `added` / `removed` → `min(1, frames.count / 6)`; `altered` → 0 with `hint = true`.
- `summary` rules: name = `label ?? "Object \(id)"`, with "?" appended when `labelConfidence < 0.6`; moved → `"\(name) · moved \(cm) cm"` + (`yawDeg` != nil ? `" · turned \(deg)°"` : `" · rotation unknown"`), prefixed "probably " before "moved" when `confidence < matchAccept`, and without the distance when `motion == nil`; added → `"\(name) · added"`; removed → `"\(name) · removed"`; altered → `"\(name) · look changed here?"`. Centimetres and degrees rounded to integers.

- [ ] **Step 1: Write the failing tests**:
  - `testHungarianPicksBestTotal`: score [[0.9, 0.8], [0.85, 0.1]] → [1, 0]; `testEmptySideGivesNoMatches`: `assign(score: [])` → [], `assign(score: [[]])` → [nil]; a 3×1 matrix assigns exactly one row.
  - `testScoreFormula`: two instances with identical feature, histogram and 20 voxels each → 1.0 (accuracy 1e-4); orthogonal features, disjoint histograms, 20 vs 60 voxels → 0.0 (size term: |ln 3|/ln 3 = 1 → 0).
  - `testMatchedPairBecomesMoved`: a tracker with one confirmed left and one confirmed arrived instance sharing a feature vector → `events()` has one `moved` with `before` and `after` set, `id == leftID`.
  - `testUnmatchedBecomeAddedAndRemoved`: orthogonal features → one `removed`, one `added`, no `moved`.
  - `testLowScoreIsMovedWithLowConfidence`: score 0.45 → `moved`, `confidence < 0.55`, summary starts with "Object 0 · probably moved".
  - `testMovedWithoutMotionIsStillMoved`: no `setMotion` → `motion == nil`, summary == "Object 0 · probably moved" or "Object 0 · moved" without " cm".
  - `testAlteredIsAHintOnly`: a confirmed sign-0 instance → `kind == .altered`, `hint == true`, `confidence == 0`.
  - `testSummaryStrings`: label "Ball" 0.9, motion 0.14 m, yaw 30 → "Ball · moved 14 cm · turned 30°"; label "Cup" 0.4, yaw nil → "Cup? · moved 14 cm · rotation unknown"; removed → "Glasses · removed".
  - `testReportRoundTripsThroughJSONAndFolder`: `save(in:)` to a temp "after scan" folder then `load` → equal events; `load` of an empty folder → nil; a folder with `objects.json` containing "not json" → nil.
  - `testMotionCacheClearsWhenInstanceGrows`: after `setMotion`, `pairsNeedingMotion()` is empty; after an ingest that adds > 20 % voxels to the arrived instance → it is listed again.
- [ ] **Step 2: Run** `swift test --filter ObjectEventsTests` → FAIL.
- [ ] **Step 3: Implement** the files above. Build events from `matches()` restricted to `score ≥ config.matchLow` after Hungarian on the full confirmed left × arrived score matrix.
- [ ] **Step 4: Run** `swift test --filter "ObjectEventsTests|ObjectTrackerTests"` → PASS.
- [ ] **Step 5: Commit** `Turn tracked instances into moved, added, removed and altered events`.

---

### Task 4: Planar ICP for the move (OSCDCore)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/PlanarICP.swift`
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/PlanarICPTests.swift`

**Interfaces:**
- Produces:
  ```swift
  public enum PlanarICP {
      public struct Config: Sendable {
          public var yawCandidates = 12; public var iterations = 20; public var trim: Float = 0.8
          public var minPoints = 10; public var symmetricRatio: Float = 0.1; public var nnVoxel: Float = 0.01
          public var inlierDistance: Float = 0.02
          public init()
      }
      /// Rigid planar motion (translation + yaw about +y) taking `source` (the reference surface) onto `target` (LiDAR now).
      /// nil when either set has fewer than `minPoints`.
      public static func fit(source: [SIMD3<Float>], target: [SIMD3<Float>], config: Config = Config()) -> Motion?
      public static func rotationY(_ deg: Float) -> simd_float3x3
  }
  ```
- Algorithm (§4.7): translation init = centroid(target) − R(yaw)·centroid(source); for each of 12 yaws (0°, 30°, …) run 20 trimmed point-to-point iterations (nearest neighbour through a voxel hash at `nnVoxel`, keep the best `trim` share of pairs, re-solve yaw + translation in closed form about +y); keep the lowest RMS. `yawDeg = nil` when `(medianRMS − bestRMS) / medianRMS < symmetricRatio` over the 12 candidates' final RMS. `inlierFraction` = share of source points within `inlierDistance` of a target point under the best motion. `horizontalDistance = length(translation.xz)`.

- [ ] **Step 1: Write the failing tests** with a helper `cubeSurface(side: 0.06, step: 0.005) -> [SIMD3<Float>]` and `sphereSurface(radius: 0.033, n: 800)`:
  - `testRecoversTranslationAndYawOfACube`: target = R_y(40°)·cube + (0.20, 0.0, 0.05) → `yawDeg ≈ 40` (accuracy 5), translation within 0.01 per axis, `horizontalDistance ≈ 0.206` (accuracy 0.01), `inlierFraction > 0.9`.
  - `testYawIsUnknownForASphere`: sphere moved 0.15 m → `yawDeg == nil`, translation within 0.01.
  - `testTooFewPointsGivesNil`: 5 source points → nil; empty target → nil.
  - `testPartialOverlapStillFits`: cube target with its far half removed (as LiDAR sees one side) → translation within 0.02.
- [ ] **Step 2: Run** `swift test --filter PlanarICPTests` → FAIL.
- [ ] **Step 3: Implement** `PlanarICP.swift`.
- [ ] **Step 4: Run** `swift test --filter PlanarICPTests` → PASS.
- [ ] **Step 5: Commit** `Fit a planar motion between a moved object's old and new points`.

---

### Task 5: Synthetic end-to-end tracker test (OSCDCore, spec §7.1)

**Files:**
- Create: `Packages/OSCDCore/Tests/OSCDCoreTests/SyntheticChangeScene.swift`
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/ObjectTrackerSceneTests.swift`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces (test target only):
  ```swift
  struct SyntheticChangeScene {
      enum Shape { case sphere(centre: SIMD3<Float>, radius: Float), box(centre: SIMD3<Float>, half: SIMD3<Float>, yawDeg: Float) }
      struct Object { var shape: Shape; var feature: [Float]; var colour: SIMD3<Float> }
      var before: [Object]; var after: [Object]; var tableY: Float = 0
      /// Ray-casts both scenes at a pose (160×120, f = 140): dRen from `before`, dLive from `after`, images from colours,
      /// embeddings painted per cell from the object's feature (table cells get a fixed table feature),
      /// changed = |dLive − dRen| > 0.01.
      func frame(id: Int, pose: Pose, roi: AABB) -> FrameContext
      static func orbitPoses(count: 10, radius: 0.5, height: 0.45) -> [Pose]   // looking at the origin from above
  }
  ```

- [ ] **Step 1: Write the failing test** `testThreeObjectsGiveThreeEvents`: before = {sphere A at (0,0.033,0), cube B at (0.15,0.03,0.1) yaw 0, cube C at (−0.15,0.03,−0.1)}; after = {sphere A at (0.20,0.033,0) (moved), cube B at (0.15,0.03,0.1) yaw 40 (turned in place → it may come out as `moved` with yaw ≈ 40 **or** as an `altered` hint; the test accepts either), cube D at (−0.15,0.03,0.15) (added), C absent (removed)}; `roi` = ±0.4 m box; feed 10 orbit frames; then run `PlanarICP.fit` for every `pairsNeedingMotion()` and `setMotion`; assert: exactly one `moved` with `id` of the sphere's left instance, `horizontalDistance ≈ 0.20` (accuracy 0.01), `yawDeg == nil`; exactly one `added`; exactly one `removed`; no second `moved`; total confirmed non-hint events ∈ {3, 4}.
  - `testUnchangedSceneGivesNoEvents`: before == after → `events()` filtered to `!hint` is empty.
  - `testIngestStaysUnderBudgetOnSyntheticFrames`: median `IngestStats.milliseconds` over the 10 frames < 60 (release-mode informational; `XCTSkip` in Debug via `#if DEBUG`).
- [ ] **Step 2: Run** → FAIL (scene helper missing).
- [ ] **Step 3: Implement** `SyntheticChangeScene.swift` (analytic ray–sphere and ray–box intersections).
- [ ] **Step 4: Run** `swift test --filter ObjectTrackerSceneTests` → PASS. If an assertion fails for a threshold reason, tune only the defaults in `TrackerConfig` and record the change in the spec §4.4/§4.6.
- [ ] **Step 5: Commit** `Add a synthetic scene test for the object tracker end to end`.

---

### Task 6: FrameContext plumbing, SemanticStage and the Live view data path (SpikeApp)

**Files:**
- Modify: `SpikeApp/Sources/MethodA.swift` (init, `process`, `submit`)
- Create: `SpikeApp/Sources/SemanticStage.swift`
- Modify: `SpikeApp/Sources/PreviewWorker.swift` (feed the stage, write `objects.json`, log)
- Modify: `SpikeApp/Sources/PreviewController.swift` (publish the report)
- Modify: `README.md` (Live view paragraph, `preview/objects.json`)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  ```swift
  // MethodA
  init(beforeScan: URL, beforeLabel: String, withB: Bool = false, withDepth: Bool = false, status: (String) -> Void) throws
      // referenceDepth is opened when withB || withDepth
  func process(_ f: CameraFrame,
               fused: @escaping (Output, Fused, FrameContext) -> Void = { _, _, _ in },
               fusedB: @escaping (Output, Fused, FrameContext) -> Void = { _, _, _ in }) throws -> Output
      // FrameContext.changed = fused.changed ∧ valid, built on the fusion thread; dRen is NaN-filled when no depth renderer is open
  func expectedDepth(pose: Pose) throws -> [Float]?        // nil without a depth renderer
  var beforeStore: SessionStore { get }                     // the opened before-scan (for Tasks 9–10)

  // SemanticStage: one tracker on thread "oscd.semantic"; at most one frame waits, the older one is dropped.
  final class SemanticStage {
      init(config: TrackerConfig, beforeID: String, method: String)
      var onReport: ((ObjectReport) -> Void)?
      func submit(_ context: FrameContext)
      /// Runs body on the tracker thread (naming, before-photo features). Blocks until done.
      func sync<T>(_ body: @escaping (ObjectTracker) throws -> T) throws -> T
      func close()
  }
  // PreviewResult gains:  let objects: ObjectReport?; let pose: Pose?; let intrinsics: Intrinsics
  // PreviewController gains: @Published var objects: ObjectReport?; @Published var lastPose: Pose?
  ```
- After each `ingest`, the stage calls `pairsNeedingMotion()`, runs `PlanarICP.fit` for each on a `.utility` queue and `setMotion` back on the tracker thread, then emits `report(...)` via `onReport` (also after a motion lands). `TrackerConfig.roi = method.roi`.
- `PreviewWorker`: open `MethodA(..., withDepth: true)`; create the stage with `method: "A"`; in the `fused` callback `stage.submit(context)`; on each report write `Documents/preview/objects.json` (atomic) and log one line `objects: <n> events · <summaries joined by " | ">` to `preview-log.txt`; at `stop()` close the stage.

- [ ] **Step 1: Make the signature change compile**: update `DetectionRunner`'s two closures to the three-argument form (ignore the context for now).
- [ ] **Step 2: Implement** the MethodA changes, `SemanticStage`, the PreviewWorker/Controller wiring. Build the app for the device; the build log must show no new warnings.
- [ ] **Step 3: Verify on the phone**: `xcrun devicectl device process launch --device 00008150-0011681A3430C01C --terminate-existing com.veripheye.changes3d -- -live YES` with the before-scan set, sweep the changed table for 20 s, stop. Pull `Documents/preview-log.txt` and `Documents/preview/objects.json`. Expected: `objects:` lines appear after the third placed frame; `objects.json` decodes as `ObjectReport`; the median Live update interval (frame-to-frame `Camera placed` lines) is within 10 % of the run before this task (record both numbers).
- [ ] **Step 4: Run** `swift test` in OSCDCore (unchanged) → PASS; `PreviewTests`, `DetectTests` → PASS.
- [ ] **Step 5: Commit** `Feed the Live view's fused frames to an object tracker on its own thread`.

---

### Task 7: Live view UI: event list and projected arrows (SpikeApp)

**Files:**
- Create: `SpikeApp/Sources/ObjectOverlay.swift`
- Create: `SpikeApp/Sources/ObjectEventList.swift`
- Modify: `SpikeApp/Sources/PreviewView.swift` (list under the headline; overlay on the Changes panel)
- Modify: `SpikeApp/Sources/Theme.swift` (event colours)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/ObjectProjectionTests.swift` (projection helpers live in OSCDCore so they are testable)
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/ObjectProjection.swift`

**Interfaces:**
- Produces:
  ```swift
  // OSCDCore
  public enum ObjectProjection {
      public enum Orientation: Sendable { case none, sensorRight }   // sensorRight: landscape sensor image shown rotated .right
      /// Image-space point for a world point, nil when behind the camera (z ≤ 0.01) or more than 25 % outside the image.
      public static func project(_ p: SIMD3<Float>, pose: Pose, intrinsics: Intrinsics, orientation: Orientation) -> SIMD2<Float>?
      /// sensorRight mapping: (u, v) → (H − 1 − v, u); none: identity. Output is in the displayed image's pixel frame.
      public static func displayed(_ uv: SIMD2<Float>, width: Int, height: Int, orientation: Orientation) -> SIMD2<Float>
      public static func displayedSize(width: Int, height: Int, orientation: Orientation) -> (width: Int, height: Int)
  }
  // SpikeApp
  struct ObjectOverlay: View {   // Canvas over an image of displayedSize scaled to fit
      let report: ObjectReport; let pose: Pose; let intrinsics: Intrinsics; let orientation: ObjectProjection.Orientation
      var highlighted: Int? = nil
  }
  struct ObjectEventList: View { let report: ObjectReport; var compact: Bool; @Binding var highlighted: Int? }
  extension VeriColor { static let moved = Color(hex: 0xFFB000); static let added = darkGreen; static let removed = Color(hex: 0x7A8FA6); static let altered = Color(hex: 0x9B6BD6) }
  ```
- Drawing: `moved` → arrow from `before.centroid` to `after.centroid` (dashed when `confidence < 0.55`), label at the head; `added` / `removed` → a box around the projected bbox corners' extent; `altered` → a dotted circle with the hint text. Rows show a dot in the kind's colour and `event.summary()`; order unchanged between updates.

- [ ] **Step 1: Write the failing tests** in `ObjectProjectionTests`:
  - `testProjectionBehindCameraIsNil`: point at camera z = −0.5 → nil; z = 0.005 → nil.
  - `testSensorRightMapsCornersToPortrait`: W 480, H 360; (0, 0) → (359, 0); (479, 0) → (359, 479); (0, 359) → (0, 0); `displayedSize` → (360, 480).
  - `testFarOutsidePointIsNil`: a point projecting to u = −200 → nil; u = −100 (≤ 25 % of 480 outside) → non-nil.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `ObjectProjection.swift`. **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** `ObjectOverlay`, `ObjectEventList`, the PreviewView changes (`ObjectEventList(compact: true)` under the headline, `ObjectOverlay` layered on the Changes panel with `controller.lastPose`, `orientation: .sensorRight`).
- [ ] **Step 6: Verify on the phone** (`-live YES`, same sweep as Task 6): take a screenshot with `xcrun devicectl device screenshot` (or the Xcode device window) showing at least one arrow and its row; save it as `docs/img/2026-10-25/live_objects.png`. The arrow's tail must sit on the red blob where the object was and its head on the object now.
- [ ] **Step 7: Commit** `Show object events and arrows in the Live view`.

---

### Task 8: Detect post-pass over the final masks, embedding cache, objects.json (SpikeApp)

**Files:**
- Create: `SpikeApp/Sources/EmbeddingCache.swift`
- Modify: `SpikeApp/Sources/DetectionRunner.swift` (cache in the fused callbacks; new step "Find objects"; save; `-detectObjects`)
- Modify: `SpikeApp/Sources/DetectionRunner.swift` `DetectionResult` (add `objects: ObjectReport?`, `before: ScanInfo?`)
- Modify: `SpikeApp/Sources/ScansScreen.swift:141`, `DetectScreen.swift` (pass `before`)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/Float16PackTests.swift` (the load side of `objects.json` is covered by Task 3's tests)
- Create: `Packages/OSCDCore/Sources/OSCDCore/Util/Float16Pack.swift`
- Modify: `README.md` (Detect bullet: `objects.json`; launch argument `-detectObjects YES`)

**Interfaces:**
- Consumes: Task 6's `process` signature, `expectedDepth(pose:)`, `beforeStore`; Task 3's `ObjectReport.save`.
- Produces:
  ```swift
  // OSCDCore
  public enum Float16Pack {
      public static func pack(_ v: [Float]) -> [UInt16]     // IEEE binary16 bit patterns, round to nearest
      public static func unpack(_ v: [UInt16]) -> [Float]
  }
  // SpikeApp
  final class EmbeddingCache {
      init(runID: String) throws              // Caches/detect/<runID>/, created empty
      func write(frameID: Int, embLive: [Float], embRender: [Float]) throws   // <frameID>.emb, 2 × 2 MB fp16
      func read(frameID: Int) throws -> (embLive: [Float], embRender: [Float])
      func removeAll()
  }
  // DetectionRunner
  func start(before: ScanInfo, after: ScanInfo, refine: Bool, withB: Bool = false)   // unchanged; opens MethodA with withDepth: true
  // DetectionResult: let objects: ObjectReport?  (loaded from method B's folder when B ran, else A's); let before: ScanInfo?
  ```
- The new step runs after "Final change masks" and before "Save results", titled "Find objects": for each kept frame, read the frame's depth from the after store, `dLive = CuesB.upsampleDepth`, `dRen = method.expectedDepth(pose:)`, `mask = method.changeMask(pose:, method: hasB ? "B" : "A")`, embeddings from the cache, build a `FrameContext`, `tracker.ingest`; then `PlanarICP.fit` for every `pairsNeedingMotion()`; `report(...)` saved with `method: hasB ? "B" : "A"`, `timingsMs` holding "Find objects" total and the median `IngestStats.milliseconds`. The cache is removed in a `defer`. With `-detectObjects YES` the report's JSON is written to `app-log.txt` as one `objects:` line.

- [ ] **Step 1: Write the failing test** `testFloat16RoundTrip`: [0, 1, −2.5, 0.1, 65504] → unpack(pack(v)) within relative 1e-3; NaN stays NaN.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `Float16Pack` (bit manipulation; no `Float16` type, so it also builds on Intel Macs). **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** `EmbeddingCache`, the DetectionRunner step, `DetectionResult` fields, and pass `before` from both call sites.
- [ ] **Step 6: Verify on the phone**: `... --terminate-existing com.veripheye.changes3d -- -detectBefore <Scan 4 id> -detectAfter <Scan 5 id> -detectB YES -detectObjects YES`. Pull `app-log.txt` and `Documents/sessions/<Scan 5>/detections/<Scan 4>/B/objects.json`. Expected: two `moved` events (ball, glasses) and no other non-hint event; the same on the floral-sheet pair if it is still on the phone; `Caches/detect/` is empty afterwards; "Find objects" total under 10 s for 47 frames. Record the numbers for the run report (Task 13).
- [ ] **Step 7: Verify the unchanged pair**: `-detectBefore <Scan 4> -detectAfter <Scan 4>` → `events` contains no non-hint event. If it does, raise `confirmVoxels`/`confirmFrames` or the edge margin, record the new defaults in the spec §4.4 and re-run Task 5's tests.
- [ ] **Step 8: Commit** `Find objects after a Detect run from the final masks and save objects.json`.

---

### Task 9: Before-photo features for left instances in Detect (OSCDCore + SpikeApp)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/BeforePhotoFeature.swift`
- Modify: `SpikeApp/Sources/DetectionRunner.swift` ("Find objects" step: features before matching)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/BeforePhotoFeatureTests.swift`

**Interfaces:**
- Produces:
  ```swift
  // MethodA gains:  func encodeSAM2(_ image: FloatImage) throws -> [Float]
  public enum BeforePhotoFeature {
      /// The before keyframe into which most of `points` project (in front of the camera, inside the image), with those pixel indices.
      public static func bestKeyframe(for points: [SIMD3<Float>], store: SessionStore) -> (record: FrameRecord, pixels: [Int])?
      /// Pools `encode(image)` over the projected pixels; `cache` holds one embedding per keyframe id.
      public static func pooled(for points: [SIMD3<Float>], store: SessionStore, cache: inout [Int: [Float]],
                                encode: (FloatImage) throws -> [Float]) throws -> (feature: [Float], record: FrameRecord, rect: PixelRect)?
  }
  ```
- In Detect, after all frames are ingested and before `matches()` is first used: for every confirmed `left` instance, `pooled(...)` with `encode = { try sam.encode($0) }` (expose `MethodA.encodeSAM2(_:)`), then `tracker.setFeature(instance:, feature, source: .beforePhoto)`. Store `record.id` and `rect` on the instance for Task 10 (`bestBeforeKeyframe: Int?`, `bestBeforeRect: PixelRect?` added to `TrackedInstance`).

- [ ] **Step 1: Write the failing tests** with a temp `SessionStore` of two 32×24 keyframes 0.3 m apart (as `SceneRegionTests` does):
  - `testBestKeyframeIsTheOneThatSeesThePoints`: points in front of keyframe 1 only → `record.id == 1`, `pixels.count == points.count`.
  - `testPooledUsesCachedEmbeddingOncePerKeyframe`: a counting `encode` closure returning a constant embedding → called once for two instances on the same keyframe; feature equals the constant's pooled value.
  - `testNoVisibleKeyframeGivesNil`: points behind both cameras → nil.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**. **Step 4: Run** → PASS.
- [ ] **Step 5: Wire** into DetectionRunner; build; re-run the Scan 4 → Scan 5 check of Task 8. Expected: both `moved` scores rise or stay (log them in the run report).
- [ ] **Step 6: Commit** `Match removed objects with features from the before-scan's own photos in Detect`.

---

### Task 10: Naming with Vision, behind a Settings toggle (SpikeApp)

**Files:**
- Create: `SpikeApp/Sources/ObjectNamer.swift`
- Modify: `SpikeApp/Sources/SettingsScreen.swift` (toggle "Name objects", `@AppStorage("nameObjects") = true`)
- Modify: `SpikeApp/Sources/SemanticStage.swift` (`pendingNaming()`), `PreviewWorker.swift` (one naming per frame, after `process` returns), `DetectionRunner.swift` (name all confirmed instances in "Find objects")
- Modify: `README.md` (Settings bullet)

**Interfaces:**
- Produces:
  ```swift
  final class ObjectNamer {
      static let minConfidence: Float = 0.3
      /// Top Vision label at or above minConfidence, else nil. Runs synchronously on the calling thread.
      func name(_ image: CGImage) -> (label: String, confidence: Float)?
      static func cgImage(_ crop: FloatImage) -> CGImage?
      /// Full-resolution crop of a before keyframe: `rect` is in working pixels; scaled by captureWidth / intrinsics.width, 20 % margin.
      static func beforeCrop(store: SessionStore, keyframeID: Int, rect: PixelRect) -> CGImage?
  }
  extension SemanticStage {
      /// Confirmed instances without a label, oldest first. Live: at most one is named per frame, on the preview thread.
      func pendingNaming() -> [TrackedInstance]
  }
  ```
- Live: `PreviewWorker.process`, after `method.process` returns (both SAM2 encodes done), takes `pendingNaming().first`, names `bestCrop` (live crop for sign ≥ 0, render crop for sign < 0) and calls `stage.sync { $0.setLabel(...) }`. Detect: in "Find objects", `left` instances use `beforeCrop` of Task 9's keyframe, others `bestCrop`. Both skip when the toggle is off.

- [ ] **Step 1: Implement** the files above (no unit test: Vision needs the device; the summary strings were tested in Task 3).
- [ ] **Step 2: Verify on the phone**: re-run Task 8's Scan 4 → Scan 5 check. Expected in `objects.json`: the ball's `moved` event carries a label containing "ball" (any case) with `labelConfidence ≥ 0.3`; the glasses may be unnamed. Toggle off in Settings → re-run → no labels. Live (`-live YES`): the ball row shows a name within the sweep; `objects:` log lines carry it.
- [ ] **Step 3: Commit** `Name detected objects with Vision, with a Settings toggle`.

---

### Task 11: Results screen Objects section and 3D viewer overlay (SpikeApp)

**Files:**
- Modify: `SpikeApp/Sources/DetectScreen.swift` `ResultsView` (Objects section, "Show in 3D")
- Modify: `SpikeApp/Sources/ModelViewer.swift` `ModelViewerScreen` (optional report overlay)
- Modify: `README.md` (Detect results and viewer bullets)

**Interfaces:**
- Consumes: Task 7's `ObjectOverlay`, `ObjectEventList`, `ObjectProjection.Orientation.none`; Task 8's `DetectionResult.objects` and `.before`.
- Produces: `ModelViewerScreen(scan: ScanInfo, report: ObjectReport? = nil)`; the overlay uses `viewer.camera!.pose` and `ModelViewer.intrinsics` (540×720 portrait, orientation `.none`) and redraws with the camera; a tapped row sets `highlighted`.
- `ResultsView`: an "Objects" section after "Summary" when `result.objects` is non-nil and has events: `ObjectEventList(compact: false)`, then a `NavigationLink("Show in 3D")` to `ModelViewerScreen(scan: result.before!, report: result.objects)` (hidden when `before` is nil). The footer reads "Positions are in the before-scan's frame. Dashed: probably the same object. Hints are not counted."

- [ ] **Step 1: Implement** both screens.
- [ ] **Step 2: Verify on the phone**: open Scan 5's saved result against Scan 4, then "Show in 3D"; orbit. Expected: arrows stay attached to the ball's old and new spots on the splat while orbiting; screenshot saved as `docs/img/2026-10-25/viewer_objects.png`. Open an older result that has no `objects.json` (or delete one): no Objects section, no crash.
- [ ] **Step 3: Commit** `List object events in the results and draw them over the 3D model`.

---

### Task 12: Frame dumps and the Mac `oscd-objects` tool (OSCDCore + SpikeApp)

**Files:**
- Create: `Packages/OSCDCore/Sources/OSCDCore/Objects/FrameContextIO.swift`
- Create: `Packages/OSCDCore/Sources/oscd-objects/main.swift`
- Modify: `Packages/OSCDCore/Package.swift` (add `.executableTarget(name: "oscd-objects", dependencies: ["OSCDCore"])`)
- Modify: `SpikeApp/Sources/DetectionRunner.swift` (`-dumpFrames YES`)
- Test: `Packages/OSCDCore/Tests/OSCDCoreTests/FrameContextIOTests.swift`
- Modify: `README.md` (launch-argument table row; a "Tuning the object layer on the Mac" paragraph)

**Interfaces:**
- Produces:
  ```swift
  public enum FrameContextIO {
      /// Writes <dir>/<frameID padded to 6>/ with changed.npy (b1), valid.npy (b1), dLive.npy (f4), dRen.npy (f4),
      /// live.npy and render.npy (f4, [3,H,W]), embLive.npy and embRender.npy (u2 = Float16Pack bits, [256,64,64]),
      /// and pose.json {"frameID", "viewmat": 16 row-major floats, "intrinsics"}.
      public static func write(_ f: FrameContext, to dir: URL) throws
      public static func read(frameDir: URL) throws -> FrameContext
      public static func readAll(in dir: URL) throws -> [FrameContext]      // sorted by frameID
  }
  ```
- `oscd-objects <frames-dir> [--roi minx,miny,minz,maxx,maxy,maxz] [--out <objects.json>] [--min-voxels N] [--accept S]`: ingests every frame, fits motions, prints one line per event (`event.summary()` plus score and voxel counts) and the median ingest ms, writes the report when `--out` is given (beforeID "mac", afterID nil, method "dump").
- Detect with `-dumpFrames YES` writes the post-pass frames to `<after>/detections/<before>/frames/` via `FrameContextIO.write` (≈ 4.5 MB per frame).

- [ ] **Step 1: Write the failing tests**: `testRoundTripPreservesEverything` (a Task 2 fixture frame → write → read → equal fields, embeddings within 1e-3 relative); `testReadAllSortsByFrameID` (ids 7, 2, 10 → [2, 7, 10]).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `FrameContextIO` and the tool. **Step 4: Run** `swift test --filter FrameContextIOTests` → PASS; `swift build --product oscd-objects` → builds.
- [ ] **Step 5: Verify**: run a Detect with `-dumpFrames YES` on Scan 4 → Scan 5, pull the `frames/` folder (`xcrun devicectl device copy from`), run `swift run -c release oscd-objects <pulled frames> --roi <Scan 4 roi>`. Expected: the same events as the phone's `objects.json` (same kinds, displacement within 1 cm).
- [ ] **Step 6: Commit** `Dump Detect frames and replay the object tracker on the Mac`.

---

### Task 13: Settings floor, ground-truth pair and the run report

**Files:**
- Modify: `SpikeApp/Sources/SettingsScreen.swift` (slider "Smallest object", `@AppStorage("minObjectVoxels") = 15`, range 5…60 step 5)
- Modify: `SpikeApp/Sources/SemanticStage.swift`, `DetectionRunner.swift` (read the setting into `TrackerConfig.confirmVoxels`)
- Create: `docs/2026-10-25_semantic_layer_runs.md`
- Modify: `README.md` (Settings bullet; a "Results" link to the run report)

**Interfaces:** none new.

- [ ] **Step 1: Implement** the slider and config plumbing; build.
- [ ] **Step 2: Capture the ground-truth pair** (a person must do this; the executor stops and asks): on the patterned mat, record the before-scan with a tennis ball, a box, a cup-sized marker and one object that will not move; build its model; then move the ball **20 cm** along the mat (tape measure, write the start and end positions), turn the box **90°** in place, remove the marker, add a cup, leave the last object; record the after-scan. Write the measurements in the run report under "Ground truth".
- [ ] **Step 3: Run Detect (A and B) on the pair** with `-detectObjects YES`, and the Live view over it. Fill the run report with the spec §7.3 table: event kinds, measured vs reported displacement, yaw, naming, "Find objects" time, Live median update interval before and after the layer, semantic stage per-frame median. Note any default changed and mirror it in the spec.
- [ ] **Step 4: Commit** `Add the smallest-object setting and the semantic layer run report`.
