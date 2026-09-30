"""YAML config loading with ${name} interpolation (other keys of the same file, then environment)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path):
    import yaml  # only planning / variant generation need PyYAML

    with open(path) as f:
        return yaml.safe_load(f) or {}


def interpolate(mapping: dict) -> dict:
    """Resolve ${name} inside string values of a flat mapping."""
    out = dict(mapping)
    for _ in range(32):
        changed = False
        for key, value in out.items():
            if not isinstance(value, str) or "${" not in value:
                continue

            def sub(m):
                name = m.group(1)
                ref = out.get(name)
                if isinstance(ref, str) and "${" not in ref:
                    return ref
                if ref is not None and not isinstance(ref, str):
                    return str(ref)
                if name in os.environ:
                    return os.environ[name]
                return m.group(0)

            new = _REF.sub(sub, value)
            if new != value:
                out[key] = new
                changed = True
        if not changed:
            break
    unresolved = {k: v for k, v in out.items() if isinstance(v, str) and "${" in v}
    if unresolved:
        raise ValueError(
            "Unresolved variables in config: %s. Define the key or export the environment variable "
            "(e.g. `source setup/sol.env` sets SCD_ROOT)." % unresolved
        )
    return out


def load_experiment(path) -> dict:
    """Load an experiment YAML plus the paths and perturbation catalogs it points to."""
    path = Path(path).resolve()
    exp = load_yaml(path)
    base = path.parent
    exp["_file"] = str(path)
    exp["_paths"] = interpolate(load_yaml((base / exp.get("paths", "../paths.yaml")).resolve()))
    exp["_perturbations"] = load_yaml((base / exp.get("perturbations", "../perturbations.yaml")).resolve())
    if "name" not in exp:
        exp["name"] = path.stem
    return exp


def stable_hash(obj) -> str:
    """Short content hash of a JSON-serialisable object (used to detect stale variants)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def seed_from(key: str) -> int:
    """Deterministic 32-bit seed from a string. Never use hash(): it is salted per process."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def fmt_severity(value) -> str:
    """Filesystem-safe severity label: 32 -> '32', -2 -> 'm2', 0.6 -> '0p6'."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return ("%g" % value).replace("-", "m").replace(".", "p")


def variant_id(stressor: str, severity, trial: int) -> str:
    return "%s-%s_t%d" % (stressor, fmt_severity(severity), trial)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
