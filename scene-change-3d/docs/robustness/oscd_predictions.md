# O-SCD: predicted failure modes against measured results

Checks the predictions in section 3.7 of the paper explainer ("How your stressors should
hit it") against two sets of runs. Written 12 Oct 2026.

| Source | What | Weight |
|---|---|---|
| A100 | Official O-SCD on PASLCD Garden, Instance 1 (group progress report, CSE 598 Group 13) | Real method on real data, but one scene and one seed |
| Synthetic | `scene_change/oscd.py` on the held-out harness scenes (`results/robustness/results_oscd*.jsonl.gz`) | 2 to 5 scenes per level (sweep at 96 of 150 levels); SIFT instead of XFeat, so pose is weaker than in the official code |

Changes are paired per scene against that scene's unperturbed run. Where the two
sources disagree, the A100 run reflects the real method.

## Verdicts

| Prediction | Verdict | Evidence |
|---|---|---|
| Blur: weaker keypoints, fewer inliers, rejected or wrong pose | Partly right; not the first failure | A100: XFeat matches −66% at 64 px, yet the pose moves only 0.14° and every frame still localises; matching fails at 128 px (−91%). Synthetic (SIFT): localised frames −13 points at 1% blur, −47 at 4% (median rotation error 32°), none from 7%, so no output. |
| Blur: misaligned render, L1/SSIM fire along edges | Right effect, wrong cause | A100: with the pose held at the clean value, the pixel cue still falls 0.56 → 0.25 at 64 px. A blurred frame differs from a sharp render at every edge even when aligned. No-change pair: false alarms ×1.9 at 32 px. Whether the false alarms sit on edges has not been measured. |
| Blur: SAM2 cue somewhat robust | Confirmed, stronger than predicted | A100: SAM2 cue 0.72 → 0.69 at 64 px; SAM2 alone beats the combined cue on 15 of 25 frames at 128 px. |
| Exposure: min-max normalisation partly cancels a uniform offset | Right for smooth shifts, not once pixels clip | A100: sensor noise, colour cast, local gain fields and a cast shadow keep 98–104% of mIoU. At +2 EV SAM2 alone beats the combined cue on 23 of 25 frames; −2 EV alone costs 12%. |
| Exposure: clipped or crushed regions remain | Confirmed; dark is the problem | Synthetic: online F1 −11% at −2 EV and −27% at −4 EV, false-alarm pixels +10 and +21 points. Overexposure up to +2 EV did not hurt (F1 +10% to +29%, fewer false alarms). |
| Exposure: very dark frames break the pose | Not observed at tested levels | Synthetic: −4 EV costs 11 points of localised frames; median rotation error stays at 0.76°. The loss is in the comparison. The A100 run did not separate pose from cue under exposure. |
| Fewer views: less evidence to outvote distractors | Weakly supported | Synthetic: false alarms +2 to +6 points, F1 barely moves (−2% to +4% online at 50% and 75% of frames removed). |
| Fewer views: refined masks recover more than online | Contradicted | Synthetic, 75% removed: refined −13%, online +4%. A100: under all 7 stressors costing more than 5%, refined loses more than online. |
| Early online masks rest on little evidence | Contradicted | Synthetic, clean: online F1 by thirds of the walk 0.20 → 0.13 → 0.10; refinement helps the last third (0.10 → 0.13), not the first. Errors accumulate rather than evidence. Partly confounded by which changes fall in which third. |
| Frame order: online depends on it, refinement removes most of it | Not tested | Neither source shuffles frame order. |

## What to change in the explainer

1. Blur breaks the comparison before the pose. The pixel cue (0.8·L1 + 0.2·(1 − SSIM))
   collapses under a correct pose because the render is sharp and the frame is not. Pose
   failure is a second, later mechanism (128 px with XFeat). It comes first in the
   re-implementation only because SIFT on low-texture synthetic boxes is weaker.
2. The cue sum is the weak point. C = C_pixel + C_feature lets a broken pixel cue swamp
   a healthy SAM2 cue (15/25 and 23/25 frames where SAM2 alone wins). Worth testing:
   weight the cues by frame quality, or fall back on SAM2 alone for blurred or clipped
   frames.
3. Exposure matters through clipping and crushing only; smooth gain changes are absorbed.
   Underexposure raises false alarms more than it lowers recall.
4. Refinement is not a safety net under stress. The planned online arm with the offline
   optimisation budget separates hindsight from budget.
5. Online error accumulation is the clearer story than weak early evidence.
6. Stressors compound: blur 32 px with −2 EV costs 48% where each alone costs 12%.
7. Missing map coverage causes false alarms: r = 0.97 between refined false-alarm area
   and uncovered area on the no-change pair. In synthetic map compression, false alarms
   rise 6 points at 10 cm and 17 points at 20 cm.

## Open checks (cheap on the A100)

- Frame order: the same frames in 3 random orders, online against refined.
- Edge false alarms: split false-alarm pixels under blur by distance to image edges.
