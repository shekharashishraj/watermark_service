"""Expand an experiment YAML into manifests: variants.jsonl and one runs_<method>.jsonl per method."""
from __future__ import annotations

from pathlib import Path

from . import SCENES
from .config import variant_id, write_jsonl
from .variants import LAYOUT, pair_dirname, parse_pair, variant_dir

# Released settings (run_oscd.sh / MV3DCD run.sh). Arms in the experiment YAML override these.
METHOD_DEFAULTS = {
    "oscd": {
        "resolution": 4, "test_hold": 5, "refine": True, "steps_per_frame": 16, "refine_total": 3000,
        "save_scores": True, "frame_log": True,
    },
    "mv3dcd": {
        "resolution": 8, "t": 0.5, "reference": "cache", "save_scores": True,
        "ref_iters": 7000, "change_iters": 10000, "aug_iters": 13000,
    },
}

EVAL_OUTPUTS = {
    "oscd": [
        {"name": "online", "pred": "renders/change_mask", "score": "renders/change_score", "score_kind": "oscd_raw"},
        {"name": "refined", "pred": "renders/change_mask_refined", "score": "renders/change_score_refined",
         "score_kind": "oscd_raw", "requires": "refine"},
    ],
    "mv3dcd": [
        {"name": "final", "pred": "renders/binary_masks", "score": "renders/change_score", "score_kind": "mv3dcd"},
    ],
}


def _scenes(value):
    if value in (None, "all"):
        return list(SCENES)
    unknown = [s for s in value if s not in SCENES]
    if unknown:
        raise ValueError("unknown scenes %s; valid: %s" % (unknown, SCENES))
    return list(value)


def expand(exp: dict):
    """Return (variants, runs_by_method, refcache_runs)."""
    paths, catalog = exp["_paths"], exp["_perturbations"]["stressors"]
    name = exp["name"]
    variants, runs, refcache = {}, {}, {}

    def add_variant(method, scene, pair, stressor, severity, trial):
        vid = variant_id(stressor, severity, trial)
        out = variant_dir(paths, method, vid, pair, scene)
        variants[str(out)] = {
            "method": method, "scene": scene, "pair": pair, "stressor": stressor, "severity": severity,
            "trial": trial, "variant_id": vid, "out_dir": str(out),
        }
        return vid, out

    # keys under `methods:` are labels; `method:` names the method (defaults to the label), so one
    # experiment can hold several blocks for the same method (e.g. extra seeds on a scene subset)
    for label, mcfg in exp["methods"].items():
        method = mcfg.get("method", label)
        if method not in METHOD_DEFAULTS:
            raise ValueError("unknown method %r in block %r (expected one of %s)"
                             % (method, label, list(METHOD_DEFAULTS)))
        sweep = mcfg.get("sweep", exp.get("sweep"))
        trials = dict(exp.get("trials", {}), **mcfg.get("trials", {}))
        arms = mcfg.get("arms") or {"default": {}}
        seeds = mcfg.get("seeds", [0])
        scenes = _scenes(mcfg.get("scenes", "all"))
        pairs = mcfg.get("pairs", ["Instance_1", "Instance_2"])
        code_dir = paths["code_%s" % method]
        for pair in pairs:
            kind, a, b = parse_pair(pair)
            if kind == "null" and LAYOUT[method] == "offline":
                raise ValueError("no-change pairs are only implemented for O-SCD (MV3DCD needs protocol B)")
        for stressor, levels in sweep.items():
            if stressor not in catalog:
                raise ValueError("stressor %r is not defined in perturbations.yaml" % stressor)
            for severity in levels:
                for trial in range(int(trials.get(stressor, 1))):
                    for scene in scenes:
                        for pair in pairs:
                            vid, vdir = add_variant(method, scene, pair, stressor, severity, trial)
                            null = parse_pair(pair)[0] == "null"
                            for arm, overrides in arms.items():
                                margs = dict(METHOD_DEFAULTS[method], **(overrides or {}))
                                for seed in seeds:
                                    out = (Path(paths["runs"]) / name / method / arm / vid / pair_dirname(pair)
                                           / scene / ("s%d" % seed))
                                    outputs = [o for o in EVAL_OUTPUTS[method]
                                               if not o.get("requires") or margs.get(o["requires"])]
                                    spec = {
                                        "run_id": "%s/%s/%s/%s/%s/%s/s%d" % (name, method, arm, vid,
                                                                            pair_dirname(pair), scene, seed),
                                        "experiment": name, "method": method, "arm": arm, "scene": scene,
                                        "pair": pair, "null": null, "stressor": stressor, "severity": severity,
                                        "trial": trial, "seed": seed, "variant_id": vid,
                                        "data_dir": str(vdir), "out_dir": str(out), "code_dir": code_dir,
                                        "method_args": margs, "reference_cache": None,
                                        "eval": {
                                            "gt_dir": None if null else str(vdir / "gt_mask"),
                                            "image_dir": str(vdir / ("inference_scene/images"
                                                                     if LAYOUT[method] == "online" else "images")),
                                            "work_scale": margs["resolution"],
                                            "frame_log": "frame_log.jsonl" if method == "oscd" else None,
                                            "outputs": outputs,
                                        },
                                    }
                                    if method == "mv3dcd" and margs.get("reference") == "cache":
                                        ref_out = (Path(paths["runs"]) / "_refcache" / "mv3dcd"
                                                   / ("r%s_i%s" % (margs["resolution"], margs["ref_iters"]))
                                                   / scene / pair / ("s%d" % seed))
                                        spec["reference_cache"] = str(ref_out)
                                        _, clean_dir = add_variant(method, scene, pair, "clean", 0, 0)
                                        refcache[str(ref_out)] = {
                                            "run_id": "_refcache/mv3dcd/r%s_i%s/%s/%s/s%d" % (
                                                margs["resolution"], margs["ref_iters"], scene, pair, seed),
                                            "experiment": "_refcache", "method": "mv3dcd", "arm": "refcache",
                                            "scene": scene, "pair": pair, "null": False, "stressor": "clean",
                                            "severity": 0, "trial": 0, "seed": seed, "variant_id": "clean-0_t0",
                                            "data_dir": str(clean_dir), "out_dir": str(ref_out), "code_dir": code_dir,
                                            "method_args": dict(margs, reference="train", refcache_only=True),
                                            "reference_cache": None, "eval": None,
                                        }
                                    runs.setdefault(method, {})[spec["run_id"]] = spec

    variant_list = sorted(variants.values(), key=lambda v: v["out_dir"])
    runs_by_method = {m: sorted(r.values(), key=lambda s: s["run_id"]) for m, r in runs.items()}
    return variant_list, runs_by_method, sorted(refcache.values(), key=lambda s: s["run_id"])


def write_manifests(exp: dict) -> dict:
    variants, runs, refcache = expand(exp)
    mdir = Path(exp["_paths"]["manifests"]) / exp["name"]
    files = {"variants": mdir / "variants.jsonl"}
    write_jsonl(files["variants"], variants)
    for method, specs in runs.items():
        files["runs_%s" % method] = mdir / ("runs_%s.jsonl" % method)
        write_jsonl(files["runs_%s" % method], specs)
    if refcache:
        files["runs_mv3dcd_refcache"] = mdir / "runs_mv3dcd_refcache.jsonl"
        write_jsonl(files["runs_mv3dcd_refcache"], refcache)
    counts = {"variants": len(variants), **{"runs_%s" % m: len(s) for m, s in runs.items()}}
    if refcache:
        counts["runs_mv3dcd_refcache"] = len(refcache)
    return {"dir": str(mdir), "files": {k: str(v) for k, v in files.items()}, "counts": counts}
