> **Interim, 25 Sep 2026.** Our detector, its confirmed-only variant and the 2D video
> comparison are complete on all 5 held-out scenes. O-SCD is partial (96 of 150 levels):
> scenes 0 and 3 complete, scenes 1 and 4 partial, scene 2 not started. O-SCD numbers
> below will move. The main sweep evaluates our detector as it was before the two safeguards at the
> end of this summary.

## Findings so far

Against the proposal's success criteria: (1) failure boundaries, (2) whether change
scores stay calibrated, (3) offline against online behaviour.

**1. Where each method breaks** (first level with a significant 20% drop in frame F1):

- **2D video comparison** breaks first. It fails at 1% motion blur (F1 −21%, near zero from
  4%), at −2 EV underexposure (−34%) and at a 2× lighting change (−28%). It fails
  quietly: it can no longer align frames, so it stops flagging anything.
- **Our 3D detector** holds against image degradation until it is severe: 10% blur
  (−22%) and −3 EV (−30%). Its confirmed detections never fail under blur, exposure or
  lighting, because they come from depth. It breaks early where depth or views matter: at
  the first level of added depth noise (−36%), with half of the frames removed (−45%),
  and with the baseline map coarsened to 10 cm Gaussians (−40%).
- **O-SCD** (interim) collapses at 7% blur (−100%). PnP localises no frame, so there is
  no output at all. It loses about a quarter of its F1 at −4 EV and holds against lighting
  changes and view removal. Its F1 is low throughout (about 0.24 online, 0.27 refined), with
  23–28% of pixels flagged on frames without change.

**2. Do the scores say when to worry?**

- On clean captures, our per-detection scores are well ordered: detections scored
  0.85–1 are right 100% of the time, those scored 0.4–0.55 only 15%. Under the most
  severe levels the top band drops to 51%. At −4 EV, 45% of false-positive pixels carry a
  score of 0.9 or more, so errors become confident. A calibration map fitted on clean
  scenes barely helps (ECE 0.38 → 0.33). The review flag holds up better: confirmed
  detections stay 85–100% correct under every image stressor.
- The 2D method's scores recalibrate well under every stressor (ECE about 0.02 after the
  clean-fitted map), but they stop ranking changes under blur (AP 0.13 → 0.03).
- O-SCD's change scores are poorly calibrated even on clean captures (ECE 0.30–0.38).
- Run-level warning signals: the 2D method's share of aligned frames tracks its accuracy
  (rank correlation 0.64). Our depth agreement tracks depth failures (0.61) but not
  appearance failures. O-SCD's share of PnP-localised frames tracks weakly (0.25–0.36).

**3. Offline against online** (interim):

- O-SCD's live masks get worse along the walkthrough (F1 0.27 in the first third, 0.18
  in the last). Refinement over the whole walk holds up better late (0.24).
- Removing frames *helps* O-SCD online on the frames that remain (+53% F1 under lost
  coverage, fewer frames to accumulate error), while our offline detector needs the
  views (−69% with 75% removed).
- Blur defeats both O-SCD modes alike: refinement cannot recover from failed
  localisation.

**Unobserved changes.** When stretches of the walk are removed, our detector reports every
change that no kept frame shows as not checked (23 of 23). The other methods report no
change there.

**Map compression** (proposal's extension): small changes fail first. At 7.5 cm
Gaussians (76k instead of about 190k) nothing is lost. At 10 cm, 6% of small changes are
still found against 71% of large ones.

## Safeguards added from these results

Both thresholds were set on the development scenes (s100, s101) and then checked on the
held-out ones. On the standard benchmark neither fires, and scores are unchanged to
within 0.001 (`RESULTS.md`).

- **No all-clear from noisy depth.** Before: with heavy depth noise, our detector found
  nothing and reported "Guest-ready" on all 5 held-out scenes that had 8 changes. Now a
  clean verdict needs at least 80% of the depth to agree with the baseline. Wrong
  all-clears fell from 6 to 0, and the safeguard never fired on unmodified captures
  (`depth_gate.json`).
- **No restyle flood from bad exposure.** Clipped or crushed pixels, and frames whose
  colour fit hits its gain limits, no longer count as appearance evidence. At −4 EV, frame
  F1 went from 0.18 to 0.56 and false-alarm pixels from 39% to 0.4%, with confirmed
  detections unchanged. Unmodified captures were unaffected (`exposure_gate.json`).

**When not to trust a result:** motion blur beyond about 1% of the frame width for 2D
comparison, and beyond about 4% for O-SCD's localisation. A walkthrough with fewer than
about half its frames, or with depth noise above the sensor's own, for our detector. Any
restyle flag from a strongly under- or overexposed visit.
