# O-SCD Mobile: semantic change layer ("which object moved")

- **Date:** 2026-10-25
- **Status:** design agreed in conversation; builder check done 2026-10-25, its six questions settled in §5
- **Owner:** shekharashishraj (ASU applied project)
- **Builds on:** `2026-10-14-oscd-mobile-design.md` (the app, Methods A and B), `oscd_mobile/docs/2026-10-19_phase2_runs.md` (current timings)
- **Deviates from O-SCD on purpose.** O-SCD stops at per-frame change masks. Everything below is additive and
  leaves Methods A and B untouched.

---

## 1. Purpose

### 1.1 The gap
Today the app answers "where did pixels change?" with a red mask per frame. The demo audience asks a different
question: "**what** moved, from where to where, and what was added or taken away?" A moved object currently shows
up as two red blobs with nothing connecting them.

### 1.2 What this adds
A layer after fusion that turns the change into a short list of **object events**, each one of:

| Kind | Meaning | What is reported |
|---|---|---|
| `moved` | the same object is gone from one place and present at another | old and new position, displacement in cm, rotation about the table normal when it can be told |
| `added` | present now, not in the before-scan | position, size |
| `removed` | in the before-scan, gone now | position, size |
| `altered` | same place, changed look (recoloured, rotated flat, …) | position; **hint only**, never a confirmed event (§4.5) |

Each event has a confidence and, optionally, a name ("tennis ball"). The list **updates live** while the user
sweeps the phone, and is saved with Detect results and shown in the 3D viewer as arrows and boxes.

### 1.3 The demo moment (success as the user sees it)
The person sweeps the phone over the changed table. Within a few updates after the camera has seen both the old
and the new spot, the Live view says "Ball: moved 14 cm" with an arrow from where it was to where it is, and
"Glasses: removed" over the empty spot. In the 3D viewer the same arrows sit on the before-scan's model.

---

## 2. Goals, non-goals, anti-goals

### 2.1 Goals
1. Object events (moved / added / removed) with 3D positions in the before-scan's world frame; `altered` as an
   unconfirmed hint.
2. For `moved`: a measured translation (≤ 2 cm error on tape-measured test moves) and a yaw angle when the object's
   shape allows it.
3. Live: the list and arrows update in the Live view without lowering its current update rate by more than 10 %.
4. Detect: `objects.json` next to `summary.json`; events listed on the results screen; arrows and boxes drawn over the
   before-scan's model in the 3D viewer.
5. Optional names from an on-device classifier, shown as hints.
6. Honest output: an unmatched pair is reported as `removed` + `added`, never forced into a `moved`.

### 2.2 Non-goals (this version)
- Full 6-DoF rotation. Planar motion (translate on the table, rotate about its normal) is enough for the demo.
- Objects that move *during* a sweep, deformable objects, people.
- Speech output. Telling identical objects apart (two identical cups: the nearest pairing wins and is marked low
  confidence).
- Any change to Methods A or B, to fusion, or to the O-SCD parity tests.
- Objects under about 2 cm tall (flat sunglasses, a phone lying flat, paper, a flat cable) are not reported: their
  points lie within the support gate (§4.4). On Scan 4 → 5 the LiDAR puts the sunglasses only 0–1 cm above the sheet,
  the sheet's own height, so no depth rule can tell them apart from it (C1, §8).

### 2.3 Anti-goals (what would make it a failure even if it "works")
- A confident wrong story: a false `moved` arrow between two unrelated changes, or a label stated as a fact.
- Slowing the Live view so the masks themselves fall behind.
- A tabletop-only hard-coding that breaks on a room scan. Shortcuts for the tabletop stay as tunable defaults.

---

## 3. What the pipeline already provides (constraints the design builds on)

From `MethodA.process` (Detect and Live share it), per accepted frame, in the before-scan's world frame:
- the camera pose (`Pose`, world→camera, OpenCV axes) and intrinsics at 480×360;
- the live RGB, the reference **render** at that pose, and the D3 validity mask (`out.valid`);
- SAM2 embeddings of the live image and of the render (256 × 64 × 64 each);
- the LiDAR depth and confidence of the frame (`CameraFrame.depth`), upsampled by `CuesB.upsampleDepth`;
- with B: the reference's **expected depth** `D_ren` (`ReferenceDepth`) and `CuesB.depthCue`, which already
  returns a per-pixel **sign** (+1 = live surface is closer than the reference, i.e. something arrived; −1 = further,
  something left);
- the **fused change mask** at that pose, delivered one frame late on the fusion thread (`Fused.changed`), stable
  across frames where the per-frame cue flickers.

Facts that shape the design:
- The change model's Gaussians are **not** exported by the C API (`oscd_fusion_c.h` gives count, scales and
  opacities only). Also, `R_change` only has Gaussians where the reference had geometry, so an added object's change
  lands on the table behind it (spec §8.5). The layer therefore does **not** cluster Gaussians; it back-projects
  mask pixels with LiDAR / rendered depth instead, which puts added objects at their true surface. S2 (Gaussian
  insertion) is not needed for this layer.
- World frame: ARKit's world of the before-scan, gravity aligned, **+y up**. The table plane is ≈ horizontal (xz).
- Live view today runs **A only** at ≈ 1 s per frame (fusion 820 ms, SAM2 ≈ 940 ms across both images, pose
  ≈ 250 ms). B adds a second fusion of similar cost. The no-change pair flags 11.7 % (A) / 17.3 % (B), so single
  frames carry real false positives.
- splatx sessions are single-thread; the app already runs pose/cue, fusion A, fusion B and the viewer on dedicated
  threads with back-pressure (`FusionStage`).
- Before-scan keyframes (RGB at working resolution, LiDAR depth, ARKit poses in the same world) are on disk in
  `session.json` + arrays.

---

## 4. Design

### 4.1 Where it sits
```
MethodA.process ──cue──▶ FusionStage (A or B) ──fused mask──▶ SemanticStage ──ObjectReport──▶ UI / objects.json
                └──────────── FrameContext (depth, D_ren, sign, embeddings, rgb, pose) ──────▶
```
- `ObjectTracker` (new, **OSCDCore**, pure Swift, no splatx, unit-testable on the Mac) holds the state and does
  steps 4.3–4.7.
- `SemanticStage` (new, **SpikeApp**) owns one `ObjectTracker` per run, runs it on its own serial thread fed from
  the fused callbacks, and runs matching / ICP / naming on a lower-priority queue.
- `MethodA.process` gains a `FrameContext` (see 4.2) passed into the `fused` / `fusedB` callbacks alongside the
  existing `Output` snapshot. Nothing else in Method A/B changes.

### 4.2 Per-frame input: `FrameContext`
Captured in `process` after the cues and carried to the fused callback:
- `pose`, `intrinsics`, frame id (Detect: keyframe id; Live: a counter);
- `rgb` (live), `render` (reference RGB), `valid`;
- `dLive` (upsampled LiDAR, NaN where untrusted), `dRen` (expected reference depth, the model's);
- `dBefore`: the before-scan's own LiDAR splatted into the view (C1, §4.4), the reference the layer compares with;
- `sign: [Int8]` from `CuesB.depthCue` (+1 arrived, −1 left, 0 no depth evidence);
- `embLive`, `embRender` (256 × 64 × 64 floats each).

**Detect runs the tracker on the final masks (my call).** Detect's red masks and `summary.json` come from the
finished change model (`changeMask` after `finishFusion` and optional `refine`), not from the online model; the
object list must agree with the masks beside it. So in Detect the `SemanticStage` is fed by a **post-pass** over
the kept frames: the final mask per frame, with that frame's `dLive`, `dRen` and sign recomputed (cheap) and its
SAM2 embeddings read back from a per-frame **fp16 cache** written during the run (`Caches/detect/<run>/<frame>.emb`,
2 × 2 MB per frame, ≈ 190 MB for 47 frames, deleted when the run's `objects.json` is saved). Nothing is
re-encoded. The streaming path (fused callbacks) is the Live view's.

**Live view change (my call):** Live opens a `ReferenceDepth` (depth render ≈ 16 ms, already a stage in Detect)
so `dRen` and `sign` exist, but does **not** add fusion B. The semantic layer in Live uses **A's fused mask + the
depth sign**. In Detect it uses **B's fused mask** when B ran, else A's. Memory: one embedding pair ≈ 8 MB, held
only until the frame's fused callback has run (one frame of back-pressure, as today).

### 4.3 Step 1: signed 2D components (per fused frame)
1. `changed = fusedMask ∧ valid`.
2. **The layer computes its own sign (my call).** `CuesB.depthCue`'s tolerance (≥ 1 cm plus a 2 cm ramp) is tuned
   for the change cue and leaves thin objects (sunglasses, a marker, ≈ 1 cm) with no sign at all. The layer uses
   `d = dLive − dRen − o` (o = the frame's median offset, as D13) and `sign = +1` if `d < −τ_sem`, `−1` if
   `d > τ_sem`, else 0, with **τ_sem = 5 mm** (tunable). Then `arrived = changed ∧ sign>0`, `left = changed ∧ sign<0`,
   `flat = changed ∧ sign==0`. A component's sign is the |d|-weighted majority of its pixels, so LiDAR noise at a
   few pixels does not split it.
3. 4-connected components on each, dropping components under **40 px** (≈ a 1.3 cm object at 40 cm; tunable).
   A component whose back-projected points all lie on the table adds nothing (the support gate, §4.4).
4. Per component: pixel list, bounding box, pooled SAM2 feature (mean of the 64×64 cells its pixels fall in, from
   `embLive` for `arrived`/`flat`, from `embRender` for `left`), and a CIELAB colour histogram (L 4 × a 8 × b 8
   bins; from the live image for `arrived`/`flat`, from the gain-aligned render for `left`, reusing
   `CuesB.colorCue`'s gain step).

**The reference depth is the before-scan's LiDAR (C1).** Wherever this section and §4.4 say dRen for the sign, the
offset o or the left points, the layer uses `dBefore` when the frame carries it (Detect and Live always fill it): the
before-scan's high-confidence LiDAR inside the ROI, one point per 5 mm voxel (the mean), built once per run
(`BeforeDepth.points`, ≈ 58 000 points and ≈ 110 ms on Scan 4), splatted into each frame's view with a z-buffer of
3 × 3 pixel splats (`BeforeDepth.splat`, ≈ 0.8 ms per frame on the phone; 84 % of a frame's valid pixels covered).
A pixel where no point lands is no evidence: no sign, no left point, no gate pass; the model's dRen is not used there.
So a hole in dBefore (≈ 16 % of a frame's valid pixels uncovered) can leave a real change with too few signed pixels
and demote it to a `flat` hint, or keep it unconfirmed: a miss, not a wrong story.
dRen and the render stay for the SAM2 feature and the colour histogram only (and for Methods A/B). Why: the model's
depth over the sheet is off by a non-uniform 1–3 cm (per-frame median offset +4 … +162 mm on Scan 4 → 5, against
−13 … +46 mm, mostly within ±10 mm, for the before-LiDAR), and no single per-frame offset absorbs it; both LiDARs come
from the same sensor.

### 4.4 Step 2: back-projection and incremental 3D association
Each component's pixels become world points:
- `arrived` / `flat`: `world = pose⁻¹ · K.unproject(uv, dLive)` — the object's surface as LiDAR sees it now;
- `left`: `world = pose⁻¹ · K.unproject(uv, dBefore + o)` — the surface the before-scan had there, with o the frame's
  D13 offset (median of dLive − dBefore over judged pixels, returned by `ChangeSign.compute`; plain dBefore when no
  pixel is judged). (Before C1 this was dRen + o, R28; the same formula applies with dRen when no dBefore is given.)

**Left points use the D13-aligned model depth (R28; superseded by the before-LiDAR reference, C1 §4.3, whenever dBefore is given).** The splat model's depth and LiDAR disagree by a per-frame
offset: on Scan 4 the model's table sits 1–2 cm nearer than LiDAR's (left components' median y −0.434 … −0.411 m
against LiDAR's table at −0.445 m). Placed with raw dRen, the model's own sheet floated above the support gate and
became confirmed `removed` blobs (up to 3 145 voxels), while arrived points (dLive) already sat on LiDAR's table.
Adding o puts both point kinds on LiDAR's table, so one support height gates both; the `flat` gate check uses the
aligned model point too. The sign itself already subtracts o, so it is unchanged.
Pixels with no finite depth of the needed kind are skipped. Points are hashed into **1 cm voxels** (`SIMD3<Int32>`
key); each component yields a voxel set.

**Support gate (R26, height from C1).** Before voxelising, points less than **2 cm** (`minHeightAboveSupport`) above
the table are dropped: the `dLive` point for `arrived`/`flat`, the aligned reference point (dBefore + o) for `left`; a
`flat` pixel is kept when either its `dLive` or its reference point clears the gate. The table height (`supportHeight`, world y) is the mode of y over the
before-scan's high-confidence LiDAR points inside the ROI (`SupportPlane`: 1 cm histogram, refined by the median of
the winning bin; ≤ 60 keyframes), computed once per run in Detect and in Live. **It needs evidence (R37):** the
winning bin must hold **≥ 15 %** of the in-ROI points, else there is no reliable table height (a cluttered, tilted or
non-tabletop scene spreads y over many bins); the share is logged (`semantic: support y = … m (share …%)`; Scan 4:
52.5 % of 1.48 M points). **No gate, no layer (R35):** when the estimate is missing or under the floor, or the
before-scan's LiDAR cannot be read (`SupportPlane` or `BeforeDepth.points` throws), the object layer is **off for
that run**: Detect skips "Find objects" with the step saying why ("Off: no reliable table height in the before-scan
…") and writes no `objects.json`; Live runs without a `SemanticStage` and its status line says why. Methods A/B and
the rest of Detect and Live run as usual. The layer never runs without the gate, because without it the sheet
itself becomes confirmed objects (below). Why: on Scan 4 → 5 and even on the
unchanged Scan 4 → 4, Method B's final masks flag 17–37 % of the judged area, mostly the floral sheet, where LiDAR
and the model's depth disagree by 10–50 mm (both signs); those regions became confirmed `removed`/`added` instances
of up to ~2 000 voxels, which no confirmation floor removes without also removing the objects. A surface lying on
the table is the sheet, not an object. (`TrackerConfig.supportHeight = nil` still means "no gate" for tests and the
Mac replay's `--no-support`; the app never runs the layer that way, R35.)

**Calibration on real frames (C1).** Both Detect pairs were dumped (`-dumpFrames YES`) and replayed on the Mac
(`oscd-objects --sweep`) over four candidate input rules, each alone and combined: a frame gate skipping frames with
|o| > 15/20/30/50 mm (`maxFrameOffset`), the before-LiDAR reference above (splat 2/3/4), the support-gate height
1/1.5/2/3 cm, and a hit rate (changed in ≥ 40/60/80 % of the frames that observed the instance, `minHitRate`).
Acceptance was zero non-hint events on Scan 4 → 4 and exactly the ball's and the sunglasses' moves on 4 → 5.
- With the model's depth, no setting cleaned 4 → 4 (1–3 false `removed`, 43–1 087 voxels, in every row); the frame
  gate kept only 9–21 of 28 frames at 15–50 mm, and the hit rate turned false moves into false removals.
- With the before-LiDAR, a 2 cm or 3 cm gate cleaned 4 → 4 at every splat, frame gate and hit rate up to 60 %, and
  4 → 5 gave exactly one event: the tennis ball, moved 8–9 cm (scores: splat 2 0.54 "probably moved", splat 3 0.60–0.63,
  splat 4 0.51–0.54; ICP RMS ≈ 2 mm, 97 % inliers). At 1–1.5 cm the sheet (and a patch at the bed's edge) came back
  as false events on 4 → 4, and at 1 cm the bed-edge patch was paired with the sunglasses as a false 24 cm move.
- No setting reported the sunglasses (their LiDAR height is that of the sheet; §2.2, §8).
Adopted: before-LiDAR reference, 3 × 3 splats, 2 cm gate; frame gate and hit rate kept as options, off (nil / 0).
Phone result on Detect: 4 → 5 "Object · moved 9 cm" (score 0.62) and nothing else; 4 → 4 no events; the Mac replay of
the dumps is identical.

Instances are tracked per sign. A new component is merged into an existing instance of the **same sign** when
`|overlap| / min(|new|, |instance|) ≥ 0.3`; if several qualify, they and the component are merged into one. Otherwise
it starts a new instance. An instance keeps:
- `id`, `sign`, `voxels: [key: hits]`, `frames: Set<frameID>`, `bbox`, `centroid` (hit-weighted);
- running mean SAM2 feature (256), summed colour histogram;
- a reservoir of ≤ 2 000 world points for ICP;
- the frame (and 2D bbox) in which it had the most pixels, for naming crops;
- `firstSeen`, `lastSeen`.

**Confirmation and noise.** An instance is `confirmed` when `frames ≥ 3` **and** `voxels ≥ 15` (defaults; a
Settings slider "Smallest object" maps to the voxel floor). Unconfirmed instances older than 10 observed frames
whose centroid projected into judged pixels in ≥ 5 of them are dropped. `flat` instances that overlap an `arrived`
or `left` instance are absorbed into it. Only confirmed instances produce events.

**`flat` never confirms (my call).** The unchanged pair (Scan 4 → Scan 4) shows persistent bands along the sheet's
far and left edges with depth agreeing, which pass every floor above. Rather than add a colour-cue vote that Live
does not compute, `flat` instances stay **hints**: shown greyed as "look changed here?" in the Live list and the
results, never counted as events, never matched, and excluded from §7's counts. Instances whose bbox comes within
2 cm of one of the ROI's four **vertical faces** (min/max x and z) are dropped outright, for all signs. The floor and
ceiling faces are exempt (R26): the scan's ROI floor sits only 1–3 cm under the table (Scan 4: y_min −0.468 m, live
table −0.445 … −0.462 m), so every object resting on the table reached the floor band and was dropped the frame it
appeared (on Scan 4 → 5, 145 of 226 arrived components had points there, 71 entirely); the table itself is handled by
the support gate above.

### 4.5 Step 3: events
Recomputed from the confirmed instances after every frame (tens of instances; trivial cost):
- `left` instance with no match → `removed`; `arrived` with no match → `added`; `flat` → `altered` **hint**;
- a matched (`left`, `arrived`) pair → `moved`.

### 4.6 Step 4: matching `left` ↔ `arrived`
Score between a `left` instance L and an `arrived` instance A:
```
score = 0.5·cos(feature_L, feature_A) + 0.3·histIntersection(L, A) + 0.2·sizeSim
sizeSim = clamp(1 − |ln(voxels_L / voxels_A)| / ln 3, 0, 1)
```
Assignment by Hungarian matching (n ≤ ~20; a simple O(n³) is fine) **with an "unmatched" option (R36)**: the
left × arrived matrix is padded to square so every left row can take a dummy column worth `matchLow` and every
arrived column a dummy row worth 0 (dummy × dummy 0); the total is maximised and only real pairs with a finite
score ≥ `matchLow` are kept; non-finite scores are never chosen. A pair s therefore beats leaving both unmatched
exactly when s ≥ `matchLow`, and the assignment can no longer force a pair just to raise the total (without it,
L1→A1 0.72, L1→A2 0.50, L2→A1 0.60, L2→A2 0.30 gave the cross pairs, 1.10, instead of L1→A1 + L2 removed + A2
added, 0.72 + 0.40; Goal 6). Thresholds (defaults, to be calibrated in §7): `score ≥ 0.55` → `moved`,
`0.40–0.55` → `moved` with `confidence` low (drawn dashed, listed as "probably moved"), below → no match.

**Feature source for `left` (my call).** The render is a blurry splat, so its SAM2 features sit in a different
domain from the live photo. **In Detect**, once a `left` instance is confirmed, the stage projects its points into
the before-scan's keyframes, picks the keyframe with the most projected points, crops there and encodes **the real
before photo** with SAM2 (one encode, ≈ 200–500 ms, on the background queue). That feature replaces the pooled
render feature for matching and naming. **In Live** this encode is skipped: the Neural Engine is already saturated
by the per-frame SAM2 pair, and a concurrent encode would stall the frame rather than overlap it. Live matches on
render features, colour histogram and size only, so its `moved` confidences are expected to be lower than Detect's.

### 4.7 Step 5: measuring the move (planar ICP)
Source: the `left` instance's points (reference surface). Target: the `arrived` instance's points (LiDAR).
Motion model: translation in xz, free small y translation, rotation **about +y** (yaw), taken about the source
centroid (`p' = R_y(yaw)·(p − c_s) + c_s + translation`, so `translation` is the displacement of the source centroid).
1. Initialise translation from centroids; try 12 yaw candidates (0°, 30°, …) and keep the lowest RMS.
2. 20 iterations of point-to-point ICP, nearest neighbours via the voxel grid, pairing in both directions (each
   source point with its nearest target point, and each target point with its nearest source point), each set
   trimmed to its best 80 % of pairs and the union solved together, so partial views on either side and clutter
   in the target do not pull the fit. The first 10 iterations keep every pair: trimming at the start traps the fit
   on flat faces.
3. Report `translation`, `horizontalDistance`, `yawDeg`, `fitRMS`, `inlierFraction`.
4. Group the 12 converged candidates into classes whose yaws agree modulo 180° (within 15°); a class's RMS is its
   best member's. The best class gives the translation and the yaw. `yawDeg` is reported (folded into (−90°, 90°];
   a box turned 40° and one turned 220° look alike, the smaller rotation is reported) only when there is no other
   class or the next-best class's RMS is at least `max(1.6 × best, best + 1 mm)`; otherwise it is `nil` and the UI
   says "rotation unknown" (a ball, a cylinder, a cube, or a view too thin to tell). Any non-finite result makes the
   whole fit `nil`. With clutter near the object (e.g. floor bleed) the fit may report "rotation unknown" while the
   translation stays accurate. The ratio was calibrated on a seeded synthetic matrix (box, cube, sphere; full, half and
   both-halves views; 0 and 2 mm noise) whose decision boundary is narrow (ratios ≈ 1.57 must stay unknown, ≈ 1.62
   must report): a known risk to recheck on real LiDAR.
Runs on the background queue when a pair is new or either instance grew by > 20 % since its points were copied for
the last fit (the counts are taken with the points, so growth during an asynchronous fit counts); up to 2 000 points
per side, **60–350 ms per pair** measured (Release, Mac), off the per-frame path: Live fits on a `.utility` queue and
the result lands in a later report; Detect fits once after the last frame. At Live's Stop the pairs still without a
move are fitted on the tracker thread before the final report.

### 4.8 Step 6: naming (optional, behind a Settings toggle "Name objects", default on)
Apple Vision `VNClassifyImageRequest` on a crop with a 20 % margin. Keep the top label if its confidence ≥ 0.3,
else the event is "Object N". **Tie-break (R34):** Vision lists a label's ancestors at (almost) its confidence
("recreation", "sport", "ball" all 0.364 on Scan 4 → 5), so among the labels ≥ 0.3 within 0.02 of the top, the first
in Vision's order that is not generic (recreation, sport, sports_equipment, structure, material, object, indoor,
outdoor, textile, furniture, consumer_electronics, tool, container, art, people, animal, plant, food, clothing) wins;
when every tied label is generic the top stays (`LabelChoice.pick`). Scan 4 → 5 now reads "Ball?". Labels are shown as hints ("Ball?" when confidence < 0.6). MobileCLIP is a later
upgrade, not part of this.

Crop sources and scheduling (my calls):
- `arrived` / `altered`: the instance's best **working-resolution** (480×360) live frame. A ball at 40 cm is
  ≈ 60 px there, so names will be weaker than from a full-resolution photo; plumbing the 1920×1440 capture through
  `CameraFrame` is **out of scope** for this version (the best frame is gone by the time an instance confirms).
- `left`: in Detect, the before keyframe chosen in 4.6, cropped from its **full-resolution JPEG** on disk
  (`fullImageURL`); in Live, the render crop (names there are weaker still, and styled as hints).
- Vision runs once per confirmed instance (≈ 20–50 ms) **serialised with the per-frame SAM2** on the preview thread,
  right after a frame's embeddings are done and before the next frame is taken, so it never contends with SAM2 on
  the Neural Engine. It counts as a one-off burst under §4.12's budget.

### 4.9 Output
```swift
struct ObjectEvent: Codable {
    var id: Int
    var kind: Kind                      // moved, added, removed, altered
    var confidence: Float               // match score for moved; support-based for the rest
    var label: String?; var labelConfidence: Float?
    var before: Placement?              // centroid, bbox (world), voxels, frames
    var after: Placement?
    var motion: Motion?                 // translation [3], horizontalDistanceM, yawDeg?, fitRMSM, inlierFraction
}
struct ObjectReport: Codable {
    var beforeID: String; var afterID: String?; var method: String
    var worldFrame = "before-scan ARKit world, +y up"; var voxelSizeM: Float
    var events: [ObjectEvent]; var generatedAt: Date; var timingsMs: [String: Double]
}
```
- Detect: `objects.json` in `<after>/detections/<before>/` next to `summary.json`. `DetectionResult` loads it.
- Live: `Documents/preview/objects.json` rewritten on each change of the event list and at Stop (Stop first waits
  for fusion, so the last fused frame is ingested, then fits the moves still missing); one
  `objects:` line per update in `preview-log.txt` (for headless checks).
- Launch argument `-detectObjects YES` on a Detect run prints the report to `app-log.txt`.

### 4.10 Live view UI
- Under the headline, a compact list: one row per event, a coloured dot, the label and the fact:
  "Ball · moved 14 cm · turned 30°", "Glasses · removed", "Object 3 · added", "Cup · probably moved 9 cm".
- On the **Changes** panel, arrows from the old to the new centroid, projected into the current frame with the
  frame's pose, label at the arrowhead; boxes for added/removed. Dashed for low confidence. Mask stays red;
  arrow / box colours: moved amber, added green, removed grey-blue, altered violet (builder may adjust; must be
  legible over a red mask on a phone outdoors).
- The list keeps its order (by first-seen) so rows do not jump between updates.

### 4.11 Detect results and 3D viewer
- Results screen: an **Objects** section listing the events as in 4.10, with "Show in 3D" (only when the
  before-scan still has its 3D model); "No objects changed" when the report has no event; no section when the layer
  was off for the run (R35) or the run predates the layer.
- 3D viewer: opened on the **before-scan's** model with an `ObjectReport`; the world frame is the same, so centroids
  and boxes are projected with `OrbitCamera.pose` and `ModelViewer.intrinsics` and drawn as a SwiftUI `Canvas`
  overlay on the rendered image (no change to the splat renderer). Arrows old → new, boxes for added/removed,
  labels, tap a row to highlight it.

### 4.12 Threading and budget
- `SemanticStage` has its own serial thread with one-frame back-pressure like `FusionStage`; if it falls behind it
  drops the older frame, never blocks fusion.
- Per-frame work (4.3–4.5): target **≤ 60 ms median** on the phone (connected components on 480×360, back-projection,
  voxel hashing, pooling). Matching, ICP, keyframe SAM2 encode and Vision run on a `.utility` queue; their results
  land in the next report.
- UI updates are coalesced to the Live view's existing per-frame delivery.
- The Live budget ("within 10 % of today's update interval") is a **steady-state median** over a sweep. One-off
  per-object work (Vision naming) may burst above it; per-frame work may not.

---

## 5. Decisions already made

**The user's:**
- The layer runs **live**, not only after Detect.
- Output is an object list with plain-language facts plus arrows in the 3D viewer; naming is optional.
- Exact O-SCD parity does not constrain this layer.

**Mine (say so if wrong):**
- Back-project masks with depth instead of clustering `R_change` Gaussians (reasons in §3).
- Live uses A's fused mask + a depth render for the sign; no fusion B in Live.
- Planar motion model (xz translation + yaw) for the first version.
- Confirmation needs 3 frames and 15 voxels; thresholds in §4.6 are starting points to calibrate.
- Match features for `left` come from the real before-scan photo once available.
- Vision classifier for names, default on, hint-styled.
- Tabletop size floors are settings, not constants.

**Settled after the builder check (mine; say so if wrong):**
- `altered` is a hint, never a confirmed event, so the unchanged pair's edge bands cannot become events (§4.4).
- The layer computes its own depth sign with a 5 mm tolerance so thin objects get a sign (§4.3); risk in §8.
- Detect's object list is computed from the **final** masks in a post-pass, with embeddings cached to disk in fp16
  during the run; the streaming path is Live's (§4.2).
- Live skips the before-keyframe SAM2 encode and names from working-resolution crops; the full-resolution live
  image is not plumbed through (§4.6, §4.8).
- The Mac replay replays the **tracker**, not the whole pipeline, from frame contexts the phone dumps (§7.2).

---

## 6. Left to the builder
- Exact SwiftUI layout and colours (within 4.10–4.11); arrow geometry.
- Voxel key hashing, reservoir sampling, Hungarian implementation.
- Whether `ObjectTracker` processes components on the fused thread or copies the mask first (either is fine within
  the budget).
- How the Mac replay is wired (`oscd-fusion-eval` subcommand or a new tool), as long as §7.2 holds.
- Log line formats, beyond the fields named in 4.9.

---

## 7. Validation and success criteria

### 7.1 Unit tests (OSCDCore, Mac)
- Synthetic scene: two spheres on a plane rendered to signed masks and depths at 10 poses; one sphere moved 20 cm
  and turned 40°, one removed, one added. Expected: exactly those three events, displacement within 1 cm, yaw within
  5°, `yawDeg == nil` for the (symmetric) sphere handled by a cube variant of the test.
- Association: components of the same instance across frames merge; different signs never merge.
- Matching: the forced-pair anti-goal (two unrelated changes score below threshold).

### 7.2 Replay on recorded scans (Mac)
**What is replayed (my call):** the tracker, not the pipeline. There is no Mac path that runs XFeat pose, SAM2,
cues and fusion over a recorded pair (`oscd-fusion-eval` consumes Python-made cues; `MethodA` lives in the app).
Instead, a Detect run on the phone with `-dumpFrames YES` writes one file per kept frame (`frames/<id>.ctx`:
final mask, sign, `dLive`, `dRen`, pose, intrinsics, and the fp16 SAM2 embeddings from the §4.2 cache) into the
run's folder. The Mac tool `oscd-objects` (new, in OSCDCore's executables) reads those and writes `objects.json`,
so thresholds in §4.4–4.6 can be tuned without the phone. A full Mac pipeline is out of scope. Pairs:
- Scan 4 → Scan 5 and the floral-sheet pair (ball and glasses moved): both must report two `moved` events and
  nothing else confirmed.
- Scan 4 → Scan 4 (no change): zero confirmed events.
- **New ground-truth pair** to capture: ball moved 20 cm (tape), a box turned 90° in place, a cup added, a marker
  removed, one object untouched; measurements recorded in `oscd_mobile/docs/`.

### 7.3 On the phone
| Criterion | Target |
|---|---|
| Events on the ground-truth pair | moved, added and removed all correct (the box turned in place may be `moved` with ≈ 90° yaw or an `altered` hint; both pass); no confirmed event on the untouched object or the table |
| Displacement error | ≤ 2 cm; yaw ≤ 15° where defined |
| Live list stabilises | within 3 updates after the camera has seen both positions |
| Semantic stage, per frame | ≤ 60 ms median; Live update interval within 10 % of today's |
| Naming | in Detect, ball and cup named correctly on the ground-truth pair; in Live, at least the ball; wrong names never shown above "?" styling |

Results go into a run report `oscd_mobile/docs/<date>_semantic_layer_runs.md`, as the phase 2 report did.

---

## 8. Risks
| Risk | Mitigation |
|---|---|
| Render-vs-photo domain gap makes SAM2 features poor for matching | before-keyframe features (4.6); colour histogram and size in the score; thresholds calibrated on §7.2 |
| Depth sign noisy at object edges (LiDAR bleed) | components under 40 px dropped; `flat` absorbed; confirmation over 3 frames |
| A small move (< object size) splits into two touching blobs with a mixed sign | signs are separated **before** connected components, so the two surfaces form two instances by construction |
| False positives (11–17 % flagged on the unchanged pair) become spurious `added`/`removed` | confirmation floors; the unchanged-pair test must report zero events |
| Live memory: embeddings per frame | one frame in flight; release after the fused callback |
| Identical objects | nearest pairing, low confidence shown; stated non-goal |
| Thin objects (≈ 1 cm) still get no sign at 5 mm tolerance because of LiDAR noise or the median offset | they fall to an `altered` hint, not a wrong event; the ground-truth pair's marker and glasses are the test; if they fail, the fallback is a presence test (live vs render SAM2 feature distance to the surrounding table) in a later version. C1: the sunglasses failed; see the thin-object row below |
| Live matching without before-photo features is weaker | expected and shown as lower confidence; Detect gives the strong answer on the same scans |
| Two objects that exchange places (Scan 4 → 5: ball and sunglasses swapped, ≈ 8 cm each) | their old footprints touch and merge into one `left` instance, so the left side mixes both and matching may prefer the wrong arrived partner; reported confidence stays low. Not solved in this version |
| Thin objects (sunglasses, a phone lying flat) are invisible to the depth layer: LiDAR puts them 0–1 cm above the sheet, under the 2 cm gate (C1) | stated non-goal (§2.2); follow-up: a depth-free presence cue (live vs render SAM2 feature or colour distance against the surrounding sheet), or a local table height instead of one global support y (the after-frames' table varies by ±1–2 cm in world y) |
| A confirmed object that gains a voxel within 2 cm of a vertical ROI face is dropped and reappears under a new id (list flicker) | known; revisit with ground truth |
| Detect embedding cache (≈ 190 MB) on a full phone | written to `Caches`, deleted after the run; the run fails cleanly if the write fails |

## 9. Relation to O-SCD
O-SCD has no object layer. This design adds one on top of its masks without altering its pose, cues or fusion, so
the parity tests of the base spec still hold. The only upstream change is carrying a `FrameContext` into the fused
callbacks and opening a depth renderer in the Live view.
