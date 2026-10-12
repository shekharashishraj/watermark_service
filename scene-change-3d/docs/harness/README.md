# Harness page

A single HTML page that shows how the synthetic walkthrough harness works, using one
held-out `mixed` scenario: floor plan with both walks and pose drift, the anatomy of one
RGB-D frame, the eight ground-truth changes, the stress ladder and the partial scenario.
Published (private) at https://claude.ai/artifact/JGn31q3qPWjt4HntBstKqv.

```bash
# needs the benchmark scenarios in runs/bench/scenarios (python -m scene_change.harness.benchmark ...)
python docs/harness/assets.py /tmp/assets.json   # renders every image the page shows (~1.5 MB)
python docs/harness/build.py /tmp/assets.json /tmp/harness.html
```
