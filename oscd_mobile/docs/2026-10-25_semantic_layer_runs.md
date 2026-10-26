# O-SCD Mobile, semantic change layer: runs, 2026-10-25

Device: iPhone 17 Pro (iPhone18,1) · app "3D Changes" (Release) · branch `oscd-mobile/semantic-layer-app`, commits `38979ca6` (before-LiDAR reference, 2 cm gate), `74b9b1cb` (splat in Detect and Live), `4c25f287` (before-photo features), `027a2366` (naming) and the commit that adds this report · spec `docs/superpowers/specs/2026-10-25-semantic-change-layer-design.md`

## Summary

- **The layer reports objects, not just masks.** On Scan 4 → Scan 5 (tennis ball and sunglasses moved) it reports exactly one event: the **tennis ball, moved 8.6 cm**, score **0.722**. On the unchanged Scan 4 → Scan 4 it reports **nothing**.
- **The sunglasses are not reported.** LiDAR sees them 0 to 1 cm above the sheet, under the 2 cm height gate (a stated non-goal, spec §2.2).
- **Before-photo features lifted the ball's match score from 0.624 to 0.722** (0.722 to 0.723 in four runs): a removed object is now described by the before-scan's own photo of it instead of the model's render.
- **Naming works but is not good yet.** Vision's top label for the ball is "Recreation" (36 %), a parent of "ball" at the same confidence. The event shows "Recreation?". See Naming.
- **Cost in Detect is small.** "Find objects" takes 2.7 s of a 40 to 42 s run; the tracker's median ingest is 9 to 13 ms per frame.
- **Two checks need a person.** The Live view sweep and the ground-truth pair are marked **Pending** below. Nothing in those sections is measured.

## What the layer does now

Spec §4 in short; the app version as of this report.

1. **Frames in.** For each placed frame the app builds a `FrameContext`: the final change mask, the live LiDAR depth, the before-scan's own LiDAR splatted into this camera (`dBefore`, the reference depth), the live image and the before model's render, and both SAM2 embeddings.
2. **Sign.** Each changed pixel is `arrived` (live depth nearer than the reference by more than 5 mm), `left` (farther) or `flat`. A per-frame median offset is removed first (D13).
3. **Instances.** Signed pixels become 2D components (≥ 40 px), then 1 cm voxels in the before-scan's world. Components overlapping an instance by ≥ 30 % merge into it. Points less than **2 cm** above the table (its height from the before-scan's LiDAR) are ignored. An instance is **confirmed** after 3 frames and **15 voxels** (Settings → Smallest object).
4. **Matching.** Each confirmed `left` is paired with a confirmed `arrived` by Hungarian assignment on `0.5·cos(SAM2) + 0.3·colour histogram + 0.2·size`. The score decides `moved` (≥ 0.55) or "probably moved" (0.40 to 0.55). A `left` feature comes from the before-scan's photo (Detect only), an `arrived` one from the live frames.
5. **Measuring.** A planar ICP (12 yaw candidates, 20 iterations, 80 % trim) fits each pair for the horizontal distance and the yaw. Yaw is reported only when it clearly beats every other fit.
6. **Events.** Unmatched `left` = `removed`, unmatched `arrived` = `added`, a flat change = an `altered` hint. They are written to `objects.json` (Detect) or `Documents/preview/objects.json` (Live), listed in the results and drawn over the 3D model.
7. **Names.** Vision's image classifier names each confirmed object. A `left` is read from the full-size before photo, the rest from the best crop. A label needs 30 % confidence and shows with "?" below 60 %.

## Detect, Scan 4 → Scan 5 and Scan 4 → Scan 4

Final build, `-detectB YES -detectObjects YES`, Scan 4 (`2026-10-14T15-45-15Z`) as the before-scan. Files pulled from the phone: `B/objects.json`, `B/summary.json`, `summary.json` and `app-log.txt`.

| | Scan 4 → Scan 5 | Scan 4 → Scan 4 |
|---|---|---|
| Camera placed | 28 / 47 | 47 / 47 |
| A flagged / B flagged (judged area) | 16.7 % / 37.1 % | 11.7 % / 17.3 % |
| Events (non-hint) | **1: Recreation? moved 9 cm** | **none** |
| Score | **0.722** | – |
| Voxels, left → arrived | 205 → 170 | – |
| ICP translation (m) | (−0.086, −0.008, −0.006) | – |
| ICP horizontal distance | 8.6 cm | – |
| ICP RMS / inliers | 2.1 mm / 97 % | – |
| Yaw | not reported ("rotation unknown") | – |
| Whole Detect run | 42.4 s | 40.2 s |

Expected: Scan 4 → 5 has two moves (ball and sunglasses), Scan 4 → 4 none. **Got:** one of two, and none. No tape measure was used on these scans, so the 8.6 cm is not checked against a measured distance; the crops in C1 show the ball moving about 8 cm.

### Before and after the before-photo features (Task 9)

Same pair, same build otherwise, naming off:

| | Score | Event |
|---|---|---|
| Without (commit `74b9b1cb`) | 0.624 | moved 9 cm |
| With (`4c25f287`) | 0.723 | moved 9 cm |
| Final build (four runs) | 0.722 to 0.723 | moved 9 cm |

Scan 4 → 4 still has no event. The `left` feature is pooled from the before keyframe into which most of the ball's old points project, encoded once with SAM2 (111 to 115 ms for the one photo, logged as `detect: before-photo features`).

### Stage times in "Find objects"

| Stage | Scan 4 → 5 (28 frames) | Scan 4 → 4 (47 frames) |
|---|---|---|
| Support height (once) | 150 ms | 136 ms |
| Before-scan LiDAR cloud, 57 685 points (once) | 121 ms | 116 ms |
| Before-LiDAR splat, per frame (median / max) | 0.8 / 1.1 ms | 0.9 / 1.3 ms |
| Tracker ingest, median per frame | 13.3 ms | 9.4 ms |
| Before-photo features | 115 ms | 0 ms (no `left` object) |
| Naming (Vision) | 50 ms for 2 objects | 0 ms |
| **Find objects, total** | **2.66 s** | **2.76 s** |

"Find objects" includes the `-detectObjects` diagnostics, so it is a slight overestimate.

## Naming (Task 10)

Scan 4 → 5, naming on:

| Object | Crop | Vision's top labels (confidence) |
|---|---|---|
| arrived (the ball, live) | 91 × 82 px | recreation 0.364, sport 0.364, ball 0.364, sports_equipment 0.364, food 0.061 |
| left (the ball, before photo) | 154 × 76 px | recreation 0.337, sport 0.337, ball 0.334, sports_equipment 0.334, golf 0.141 |

- **The event is labelled "Recreation", confidence 0.36**, shown as "Recreation? · moved 9 cm". The event takes the higher of its two objects' confidences (the arrived one).
- Vision lists a label's ancestors with almost the same confidence as the label ("recreation" and "sport" above "ball"), and the code takes the top one. "Ball" is in the list, three places down, 0.001 lower. I did not add a mapping or a most-specific rule; that is a decision for the spec owner.
- **Run-to-run variation:** in three earlier runs the left crop scored 0.29 to 0.30 and got no label (under the 30 % floor), while the arrived one got "Recreation" 0.36. Fusion is not deterministic, so the best crop differs a little.
- **Toggle off** (`-nameObjects NO`): the same run gives the same event with no label (shown as "Object N"), and "Naming" is 0 ms. The Settings switch writes the same key.
- **Unchanged pair:** no event, so nothing to name.
- **Live naming is untested on the phone** (it needs a person sweeping, see below). It names at most one object per frame after both SAM2 encodes, and retries an object that Vision cannot name up to 3 times.

## Calibration of the input rules (Task C1)

Replayed on the Mac on dumped frames (`oscd-objects --sweep`, 400 settings of reference depth, support gate, frame gate and hit rate on both pairs). Acceptance: Scan 4 → 4 has no non-hint event and Scan 4 → 5 has exactly two `moved` events.

| Reference depth | Support gate | Scan 4 → 4 | Scan 4 → 5 |
|---|---|---|---|
| Before model's rendered depth (the old reference), any frame gate or hit rate | swept | 1 to 3 false `removed` events of 43 to 1087 voxels in every row | no row meets the acceptance |
| Before-scan LiDAR | 1 to 1.5 cm | false events at splat 2 and 4; at splat 3 clean only with hit rate ≥ 40 % | a false `removed` at the bed edge beside the ball's move; once a false 24 cm "move" of bed edge → sunglasses (score 0.44) |
| Before-scan LiDAR | **2 cm** | **clean at every splat and frame gate** | **exactly one event, the ball: `moved` 0.50 to 0.63** (0.543 at splat 2, 0.624 at splat 3) |
| Before-scan LiDAR | 3 cm | clean | the same ball move, 0.51 to 0.61 |

- **0 of 400 settings meet the full acceptance**, because the sunglasses are never found.
- **Adopted:** before-scan LiDAR as the reference depth, splat 3 × 3, gate 2 cm, no frame gate, no hit rate (`38979ca6`, `74b9b1cb`).
- Before the change, the same Scan 4 → 5 run on the phone gave 4 events in 6.9 s (median ingest 16.3 ms). The old reference with only the 2 cm gate still gives a `removed` blob of 673 voxels at +2.4 cm on Scan 4 → 4, so the gate alone would not have been enough.
- The per-frame offset against the before-LiDAR is within ±10 mm on 21 of 28 frames (−13 to +46 mm); against the model it is +4 to +162 mm. The large model offsets are the model's depth, not the pose.

## The swap: ball and sunglasses exchanged places

Between Scan 4 and Scan 5 the ball and the sunglasses swapped places, each moving about 8 cm. Their old footprints touch, so their `left` regions merge into one instance (its 2D box spans both).

- **The ball** is found and matched (above).
- **The sunglasses are not reported** as `moved`, `added` or `removed`. LiDAR puts their pixels at a median 0.3 cm above the support height (99th percentile 1.3 to 1.5 cm), the same as the sheet's own pixels (median 0.25 to 0.4 cm), and the table's height moves ±1 to 2 cm from frame to frame. Any gate of 1 cm or more removes them with the sheet. With a 1 cm gate a few of their pixels survive, but so do sheet blobs of the same height on the unchanged pair, which is a worse trade.
- **Risk kept in the spec (§8):** the merged `left` can make the matcher prefer the wrong arrived partner (one candidate pairing scored 0.707 and lost to a worse assignment). Here the anti-goal holds only because the sunglasses never confirm.
- **Follow-ups** (not done): a depth-free presence cue (live versus render SAM2 or colour distance against the surrounding sheet), or a local table height in place of one global support height.

## On-phone per-frame budget

Detect, medians per placed frame (Scan 4 → 5, 28 frames; this is the pose and change-mask pipeline the layer sits behind):

| Stage | ms |
|---|---|
| XFeat | 21.7 |
| Find the closest before frames | 213.5 |
| Match to 3D anchors | 53.0 |
| PnP-RANSAC | 69.3 |
| Bundle adjustment | 7.0 |
| Render | 16.0 |
| SAM2 (render) | 255.0 |
| SAM2 (live, alongside pose) | 266.6 |
| Change cue | 131.1 |
| B: depth render + cues | 26.2 + 70.8 |
| Cache the SAM2 features (fp16), fusion thread | 15.0 |
| Fusion A / B (16 steps, own threads) | 712.3 / 772.4 |
| **End to end per placed frame** | **878.4** |

The object layer adds, per frame: splat 0.8 ms + tracker ingest 13.3 ms = **about 14 ms** (Scan 4 → 4: 0.9 + 9.4 ms), against the 60 ms budget. In Detect the tracker runs in a post-pass after fusion, so these times are measured with nothing else competing. **In the Live view the layer runs next to ARKit, XFeat, SAM2 and fusion; that figure is Pending (below).**

## Settings

- **Smallest object** (`minObjectVoxels`, 5 to 60 in steps of 5, default 15): the voxel floor for confirming an object (`TrackerConfig.confirmVoxels`), read at the start of each Detect run and each Live session. `-minObjectVoxels N` sets it for one launch. Check: Scan 4 → 4 with `-minObjectVoxels 5` gives one false `removed` event of 7 voxels that the default of 15 does not. The ball covers 170 to 205 voxels, so the whole range keeps it.
- **Name objects** (`nameObjects`, default on): Vision naming. No default was changed by this batch, so the spec is unchanged.

## Viewer check

`-viewObjects 2026-10-14T15-45-52Z` opens the 3D viewer over Scan 4's model with the latest saved detection: the log reads `viewer: objects 1 events`, which is the one event in `B/objects.json`. Touch interactions (tapping a row, "Show in 3D" from the results, how the arrow sits on the ball's old and new spots in Walk and Orbit mode) were not exercised.

## Pending: needs a person

### Live view sweep

**Pending — needs a person.** Run `-live YES` (or Settings → Live view) with Scan 4 as the before-scan, and sweep the phone slowly over the table so that both the ball's old and new position (Scan 5's layout) are seen. Pull `Documents/preview-log.txt` and `Documents/preview/objects.json`. Check:

| What to check | Target | Measured |
|---|---|---|
| An objects line (`objects: n events · …`) appears after the third placed frame | yes | – |
| An arrow for the ball (moved) in the overlay, dashed under 0.55 | yes | – |
| Event list settles | within 3 updates after both positions have been seen | – |
| Update interval, median | within 10 % of 0.724 s (the baseline from 2026-10-19, 35 placed frames, before the layer) | – |
| `semantic: frame N · before-LiDAR splat … · ingest … ms` per frame | ingest ≤ 60 ms median | – |
| `semantic: n frames ingested, m dropped · median ingest` at Stop | few dropped | – |
| A name for the ball within the sweep (`naming:` lines) | at least the ball is named; a wrong name is shown with "?" | – |

### Ground-truth pair

**Pending — needs a person.** Capture protocol (spec §7.2):

1. On the patterned mat, record the **before-scan** with a tennis ball, a box, a cup-sized marker and one object that will not move. Build its model on the phone.
2. Move the **ball 20 cm along the mat** with a tape measure; write down its start and end positions.
3. **Turn the box 90° in place.**
4. **Remove the marker.**
5. **Add a cup.**
6. Leave the last object untouched.
7. Record the **after-scan**.
8. Run Detect (A and B) with `-detectObjects YES -detectB YES`, and the Live view over it.

| Item | Measured truth | Event expected | Reported |
|---|---|---|---|
| Ball | 20 cm: start –, end – | `moved`, 20 cm ± 2 cm | – |
| Box | turned 90° in place | `moved` with yaw ≈ 90° (± 15°) or an `altered` hint | – |
| Marker | removed | `removed` | – |
| Cup | added | `added`, named a cup | – |
| Untouched object, table | – | no confirmed event | – |
| Naming (Detect: ball, cup) | – | correct, or shown with "?" | – |
| "Find objects" time | – | – | – |
| Live update interval, before / after the layer | – | within 10 % | – |
| Semantic stage per frame, median | – | ≤ 60 ms | – |

If a default changes after this pair, mirror it in the spec.

## Launch arguments

| Argument | Effect |
|---|---|
| `-detectBefore <scan> -detectAfter <scan> [-detectB YES] [-detectObjects YES]` | Run Detect; the second adds the object diagnostics to `app-log.txt` |
| `-nameObjects NO` | Run with naming off |
| `-minObjectVoxels N` | Run with a different smallest object |
| `-viewObjects <after-scan>` | Open the 3D viewer with that scan's latest saved events |
| `-live YES` | Open the Live view and start it |
