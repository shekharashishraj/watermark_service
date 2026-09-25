# Results

All numbers come from the synthetic harness (160×120 RGB-D walkthroughs rendered on CPU,
five-room apartments). Scores follow the PASLCD protocol used by O-SCD: change-class
IoU and F1 per inspection frame, averaged over frames that show a change. False alarms
are the share of pixels flagged on frames without any change. Values are comparable
between methods here, not with numbers reported on PASLCD.

Regenerate with `python -m scene_change.harness.benchmark --out runs/bench`.

## Held-out benchmark

Seeds 0 to 4 were never used during development; each seed has a `mixed` (8 changes),
`clean` (no changes) and `partial` (skipped room, half-panned room) scenario, 15 in all.
IoU and F1 average the 10 scenarios with changes; false alarms average all 15.

| Method | Frame IoU | Frame F1 | False-alarm pixels | Frames/s (4 CPU cores) |
|---|---|---|---|---|
| This prototype, all flags | 0.609 | 0.728 | 0.20% | 5.5 |
| This prototype, confirmed items only | 0.517 | 0.613 | 0.01% | |
| 2D video comparison | 0.152 | 0.215 | 2.11% | 1.3 |
| O-SCD as published (10 scenes, see below) | 0.138 | 0.219 | 22.61% | 0.33 |

O-SCD ran on the 10 `mixed` and `clean` scenarios only (about 9 minutes per scene on CPU,
most of it in the SAM 2.1 encoder). On exactly those 10 scenes:

| Method | Frame IoU | Frame F1 | False-alarm pixels | Frames/s |
|---|---|---|---|---|
| This prototype, all flags | 0.628 | 0.744 | 0.25% | 5.8 |
| 2D video comparison | 0.162 | 0.224 | 2.08% | 1.3 |
| O-SCD as published (own PnP poses, SAM 2.1) | 0.138 | 0.219 | 22.61% | 0.33 |

O-SCD localised 100 to 160 frames per walk with PnP (median pose error 2.7 cm) and
bridged the rest with odometry. Its false alarms stay between 15% and 28% of pixels on
every scene, with or without changes.

Object level, this prototype:

| Scenario kind | Changed objects found | Confirmed false positives / visit | Wrong review flags / visit | Median pose error |
|---|---|---|---|---|
| mixed (5) | 95% | 0.4 | 5.6 | 1.7 cm |
| partial (5) | 90% | 0.0 | 5.4 | 1.9 cm |
| clean (5) | no changes | 0.0 | 7.6 | 1.5 cm |

Every object the second walkthrough never saw is reported as not checked. Review flags
are mostly restyle-only detections (colour changed, geometry did not), which always go
to a person with before/after images; lamps switched on or off between visits cause
most of the wrong ones.

Since these numbers, the robustness study (`ROBUSTNESS.md`) added two capture-quality
safeguards: a clean verdict is withheld when the depth or the exposure could not have
shown a change. Re-run on the same 15 scenes with both on, frame IoU, F1, false alarms,
object recall and confirmed false positives match the tables above to within 0.001, no
safeguard fires, the verdicts are the same, and wrong review flags per visit drop slightly
(5.6 / 5.4 / 7.6 to 5.4 / 5.2 / 7.0 for mixed / partial / clean).

Missed changes are mostly small objects: a trash bin, a bottle, a toiletry item, a toy, a
backpack, and a coffee stain in a half-panned room.

### Fixes found on the held-out scenes

1. **Localization**: some walks drift 25-30 cm partway through. Chunk re-alignment now
   runs in walking order, starting from the previous chunk's correction.
2. **Silhouette edges**: Gaussians bleed a few pixels past door jambs and frame edges,
   so rays grazing those edges looked like removed surfaces. See-through evidence within
   a Gaussian's footprint of an observed depth edge no longer counts.

| This prototype, held-out | Frame IoU | Frame F1 | False-alarm pixels | Recall | Confirmed false positives / visit |
|---|---|---|---|---|---|
| Before either fix | 0.550 | 0.659 | 2.10% | 91.1% | 3.3 |
| Localization fix | 0.608 | 0.725 | 0.27% | 91.1% | 0.6 |
| Both fixes | 0.609 | 0.728 | 0.20% | 92.5% | 0.13 |

The worst scene, s002_mixed, went from F1 0.173 (median pose error 6.9 cm, 25 confirmed
false positives) to 0.684 (1.3 cm, none).

## O-SCD configuration study (development scene s100_mixed)

| Configuration | Frame IoU | Frame F1 | False-alarm pixels | Median pose error |
|---|---|---|---|---|
| As published: own PnP poses, depth-fused reference render | 0.112 | 0.189 | 22.5% | 3.1 cm |
| Own PnP poses, exposure matching | 0.151 | 0.242 | 17.7% | 3.1 cm |
| Our localization poses | 0.112 | 0.190 | 22.5% | 1.5 cm |
| True poses | 0.118 | 0.198 | 20.9% | 0 |
| True poses, exposure matching | 0.160 | 0.253 | 15.8% | 0 |
| True poses, perfect reference render | 0.235 | 0.339 | 63.0% | 0 |
| True poses, perfect render, exposure matching | 0.275 | 0.381 | 63.7% | 0 |
| This prototype on the same scene | 0.591 | 0.705 | 0.05% | 1.5 cm |

Pose accuracy is not what limits O-SCD here: its own PnP poses and our 1.5 cm poses
give the same score. The change cues are. The second visit is 23% darker with
different room lamps, and the paper corrects lighting only in its scene-update step,
so both the pixel cue and the SAM 2.1 feature cue fire on unchanged surfaces. Our
depth-fused reference also renders blurrier than a photometrically trained 3DGS. A
perfect render raises F1 but not the false alarms, because the per-frame min-max
normalisation of the cues turns lighting and sensor-noise differences into strong
responses.

## O-SCD scene update

Fidelity of the map to the second visit (PSNR on its frames):

| Map | Changed pixels | Unchanged pixels |
|---|---|---|
| Reference, not updated | 12.9 dB | 15.8 dB |
| Updated with true change masks | 21.2 dB | 15.7 dB |
| Updated with O-SCD masks | 21.7 dB | 15.5 dB (58% of the map rebuilt) |
