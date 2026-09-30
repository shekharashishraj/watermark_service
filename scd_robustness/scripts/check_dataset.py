#!/usr/bin/env python
"""Step S3 of the plan: record the dataset facts the design depends on (scd-tools env).

  python scripts/check_dataset.py --paths configs/paths.yaml --out $SCD_ROOT/results/dataset_facts.json

Answers: do both instances share the post-change images? how many frames / GT masks per scene?
is the prebuilt O-SCD reference present? what are the native image size and format? do the
online and offline zips use the same frame names? are all post-change images in the COLMAP model?
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd import INSTANCES, SCENES  # noqa: E402
from scd.config import interpolate, load_yaml  # noqa: E402
from scd.imageio import list_images  # noqa: E402


def md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def image_info(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if img is None else {"shape": list(img.shape), "dtype": str(img.dtype), "ext": path.suffix}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = interpolate(load_yaml(args.paths))
    online, offline = Path(paths["data_online"]), Path(paths["data_offline"])
    facts = {"data_online": str(online), "data_offline": str(offline), "scenes": {}, "problems": []}

    for scene in SCENES:
        s = facts["scenes"][scene] = {}
        hashes = {}
        for inst in INSTANCES:
            d = online / inst / scene
            e = s.setdefault(inst, {})
            if not d.exists():
                facts["problems"].append("online missing: %s" % d)
                continue
            inf = d / "inference_scene" / "images"
            ref = d / "reference_scene" / "images"
            gt = d / "gt_mask"
            inf_imgs = list_images(inf) if inf.exists() else []
            e["online_inference_frames"] = len(inf_imgs)
            e["online_reference_frames"] = len(list_images(ref)) if ref.exists() else 0
            e["online_gt_masks"] = len(list(gt.iterdir())) if gt.exists() else 0
            e["online_gt_without_frame"] = sorted({p.stem for p in gt.iterdir()} - {Path(f).stem for f in inf_imgs})[:5] \
                if gt.exists() else []
            e["online_reference_ply"] = (d / "reference_reconstruction" / "point_cloud" / "iteration_30000"
                                         / "point_cloud.ply").exists()
            e["online_reference_sparse"] = (d / "reference_scene" / "sparse" / "0").exists()
            if inf_imgs:
                e["online_first_image"] = image_info(inf / inf_imgs[0])
            hashes[inst] = {f: md5(inf / f) for f in inf_imgs}
            if not e["online_reference_ply"]:
                facts["problems"].append("no prebuilt O-SCD reference: %s" % d)

            o = offline / scene / inst
            if not o.exists():
                facts["problems"].append("offline missing: %s" % o)
                continue
            imgs = list_images(o / "images")
            test = [f for f in imgs if "test" in Path(f).stem]
            e["offline_test_frames"] = len(test)
            e["offline_reference_frames"] = len(imgs) - len(test)
            e["offline_gt_masks"] = len(list((o / "gt_mask").iterdir())) if (o / "gt_mask").exists() else 0
            e["offline_first_test_image"] = image_info(o / "images" / test[0]) if test else None
            e["offline_same_names_as_online"] = sorted(Path(f).stem for f in test) == sorted(Path(f).stem for f in inf_imgs)
            try:
                from scd import colmap_rw
                cams, ims, pts = colmap_rw.read_model(str(o / "sparse" / "0"))
                names = {Path(im.name).stem for im in ims.values()}
                e["colmap_images"] = len(ims)
                e["colmap_points"] = len(pts)
                e["colmap_camera_models"] = dict(Counter(c.model for c in cams.values()))
                e["colmap_missing_test_frames"] = sorted({Path(f).stem for f in test} - names)[:5]
            except Exception as err:  # noqa: BLE001
                e["colmap_error"] = str(err)
        if len(hashes) == 2:
            a, b = hashes["Instance_1"], hashes["Instance_2"]
            common = set(a) & set(b)
            s["instances_share_inference_images"] = bool(common) and all(a[f] == b[f] for f in common) \
                and set(a) == set(b)
            s["identical_inference_files"] = sum(a[f] == b[f] for f in common)

    shared = [sc for sc, v in facts["scenes"].items() if v.get("instances_share_inference_images")]
    facts["summary"] = {
        "scenes_with_shared_inference_images": shared,
        "n_problems": len(facts["problems"]),
    }
    text = json.dumps(facts, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
