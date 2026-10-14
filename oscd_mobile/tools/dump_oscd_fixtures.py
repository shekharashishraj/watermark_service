#!/usr/bin/env python
"""Dump O-SCD intermediate tensors as golden fixtures for the Swift port (oscd_mobile).

Run on a GPU node inside the scd_robustness `oscd` env (see dump_fixtures.sbatch). The script
calls O-SCD's own functions (commit 3abfeaa, patched by scd_robustness) on one PASLCD instance
and saves each component's inputs and outputs as .npy files. Random index choices (RANSAC
samples, subsets) are generated here and saved, so the Swift tests can replay them.
"""
import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np


def save(d, name, x):
    import torch
    os.makedirs(d, exist_ok=True)
    if isinstance(x, torch.Tensor):
        x = x.detach()
        x = x.float().cpu().numpy() if x.dtype.is_floating_point else x.cpu().numpy()
    x = np.asarray(x)
    if x.dtype in (np.float64, np.float16):
        x = x.astype(np.float32)
    if x.dtype == np.int32:
        x = x.astype(np.int64)
    np.save(os.path.join(d, name + ".npy"), x)


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd-root", required=True, help="O-SCD checkout (scd_robustness code/O-SCD)")
    ap.add_argument("--instance", required=True, help=".../PASLCD/Instance_1/<Scene>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    instance = os.path.abspath(a.instance)

    # O-SCD opens poses/ransac.cu with a relative path, so run from its root.
    os.chdir(a.oscd_root)
    sys.path.insert(0, a.oscd_root)
    sys.argv = ["oscd.py", "-s", instance, "-m", os.path.join(out, "_oscd_model"), "--resolution", "4"]

    import torch
    import torch.nn.functional as F
    from arguments.config_args import parse_args
    from dataloaders.image_dataset import ImageDataset
    from gaussian_renderer import render
    from oscd import generate_candidate_map
    from poses.feature_detector import Detector
    from poses.matcher import Matcher, match
    from poses.pose_initializer import PoseInitializer
    from poses.triangulator import Triangulator
    from scene import GaussianModel
    from scene.cameras import Camera
    from scene.dense_extractor import DenseExtractor
    from scene.keyframe import Keyframe
    from transformers import Sam2Model
    from utils.loss_utils import ssim

    args, lp, op, pp = parse_args()
    dataset_args, pipe = lp.extract(args), pp.extract(args)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); np.random.seed(0); random.seed(0)

    ref = ImageDataset(args, instance="ref")
    inf = ImageDataset(args, instance="inf")
    height, width = inf.get_image_size()
    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(args.num_kpts, args.num_keyframes_for_triangulation, max_error)
    pose_init = PoseInitializer(width, height, triangulator, matcher, max_error, args)
    dense = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)

    refs = []
    for i in range(len(ref)):
        image, info = ref.getnext()
        kp = detector(image)
        refs.append(Keyframe(image, info, kp, info["Rt"], i, info["focal"], dense, triangulator, args))
        f_ref, fovx, fovy = info["focal"], info["FovX"], info["FovY"]
    pose_init.init_focal(f_ref)
    f = float(torch.as_tensor(f_ref).reshape(-1)[0])
    centre = pose_init.centre

    # ---- xfeat: raw network outputs and the Detector's final keypoints for reference image 0
    xf = torch.hub.load(os.environ["SCD_XFEAT_REPO"], "XFeat", source="local", pretrained=True,
                        top_k=args.num_kpts).cuda().eval().half()
    img0 = refs[0].image_pyr[0]
    H32, W32 = (height // 32) * 32, (width // 32) * 32
    with torch.no_grad():
        x = F.interpolate(img0[None].half(), (H32, W32), mode="bilinear", align_corners=False)
        M1, K1, H1 = xf.net(x)
        M1 = F.normalize(M1, dim=1)
        K1h = xf.get_kpts_heatmap(K1)
    d = os.path.join(out, "xfeat")
    save(d, "image", img0.permute(1, 2, 0))
    save(d, "net_feats", M1[0]); save(d, "net_heatmap", K1h[0, 0]); save(d, "net_reliability", H1[0, 0])
    save(d, "kpts", refs[0].desc_kpts.kpts); save(d, "feats", refs[0].desc_kpts.feats)

    # ---- mutual-NN matching between reference images 0 and 1
    ka, kb = refs[0].desc_kpts, refs[1].desc_kpts
    idx0, m12, mask = match(ka.feats.cuda(), kb.feats.cuda())
    d = os.path.join(out, "match")
    save(d, "kpts_a", ka.kpts); save(d, "feats_a", ka.feats); save(d, "kpts_b", kb.kpts); save(d, "feats_b", kb.feats)
    save(d, "match12", m12); save(d, "mask", mask)

    # ---- 8-point F-RANSAC with recorded samples
    k1, k2 = ka.kpts[idx0[mask]].contiguous(), kb.kpts[m12[mask]].contiguous()
    est = matcher.fundmat_estimator
    idxs = torch.topk(torch.rand(est.N, k1.shape[0], device="cuda"), est.m, dim=1)[1]
    est.estimate(k1, k2, idxs, None, None, None, None)
    inl = est.get_inlier_mask(k1, k2, None, None)
    d = os.path.join(out, "fransac")
    save(d, "kpts1", k1); save(d, "kpts2", k2); save(d, "idxs", idxs); save(d, "models", est.models)
    save(d, "inliers", inl); save(d, "best_id", torch.argmax(inl.sum(dim=1))); save(d, "max_error", np.float32(max_error))

    # ---- reference anchors exactly as oscd.py builds them
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            matcher(refs[i].desc_kpts, refs[j].desc_kpts, remove_outliers=True,
                    update_kpts_flag="inliers", kID=i, kID_other=j)
    kf = refs[0]
    uv, uvs_others, chosen = triangulator.prepare_matches(kf.desc_kpts)
    Rts = torch.stack([refs[c].get_Rt() for c in chosen])
    n_cams = triangulator.n_cams
    Rts_pad = torch.cat([Rts, torch.eye(4, device="cuda")[None].repeat(n_cams - len(Rts), 1, 1)]) if len(Rts) < n_cams else Rts
    pts, depth, best, valid = triangulator(uv, uvs_others, kf.get_Rt(), Rts_pad, kf.f, kf.centre)
    d = os.path.join(out, "triangulate")
    save(d, "uv", uv); save(d, "uvs_others", uvs_others[:len(chosen)]); save(d, "Rt", kf.get_Rt())
    save(d, "Rts_others", Rts); save(d, "f", np.float32(f)); save(d, "centre", kf.centre)
    save(d, "max_error", np.float32(max_error)); save(d, "min_dis", np.float32(30 * max_error))
    save(d, "pts3d", pts); save(d, "depth", depth); save(d, "best_dis", best); save(d, "valid", valid)
    for k in refs:
        k.update_3dpts(refs)
    d = os.path.join(out, "anchors")
    for i, k in enumerate(refs):
        dk = k.desc_kpts
        save(d, f"kf_{i:04d}_kpts", dk.kpts); save(d, f"kf_{i:04d}_feats", dk.feats)
        save(d, f"kf_{i:04d}_has_pt3d", dk.has_pt3d); save(d, f"kf_{i:04d}_pts3d", dk.pts3d)
        save(d, f"kf_{i:04d}_Rt", k.get_Rt())
    with open(os.path.join(d, "info.json"), "w") as fh:
        json.dump({"count": len(refs), "f": f, "cx": float(centre[0]), "cy": float(centre[1]),
                   "width": width, "height": height, "max_error": max_error}, fh, indent=2)

    # ---- per-frame pose pipeline with recorded randomness, then O-SCD's own end-to-end pose
    gaussians = GaussianModel(dataset_args.sh_degree, dataset_args.sh_degree)
    gaussians.load_ply(os.path.join(instance, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().to("cuda")
    bg = torch.zeros(3, device="cuda")
    for k in range(a.frames):
        image, info = inf.getnext()
        kp = detector(image)
        d = os.path.join(out, "pose", f"frame_{k:03d}")
        save(d, "kpts", kp.kpts); save(d, "feats", kp.feats)
        counts = torch.tensor([float(matcher.evaluate_match(r.desc_kpts, kp)) for r in refs])
        top = torch.topk(counts, args.num_reference_keyframes).indices
        save(d, "retrieval_counts", counts); save(d, "retrieved", top)
        kfs = [refs[i] for i in top.tolist()]
        xyz, uvs, confs = [], [], []
        for kfr in kfs:
            m = matcher(kp, kfr.desc_kpts, remove_outliers=True)
            msk = kfr.desc_kpts.has_pt3d[m.idx_other]
            xyz.append(kfr.desc_kpts.pts3d[m.idx_other[msk]]); uvs.append(m.kpts[msk])
            confs.append(kfr.desc_kpts.pts_conf[m.idx_other[msk]])
        xyz, uvs, confs = torch.cat(xyz), torch.cat(uvs), torch.cat(confs)
        if len(xyz) > pose_init.num_pts_pnpransac:
            sel = torch.multinomial(confs, pose_init.num_pts_miniba_incr, replacement=False)
            xyz, uvs, confs = xyz[sel], uvs[sel], confs[sel]
        save(d, "corr_xyz", xyz); save(d, "corr_uv", uvs); save(d, "corr_conf", confs)
        est = pose_init.PnPRANSAC
        p_idxs = torch.topk(torch.rand(est.N, uvs.shape[0], device="cuda"), est.m, dim=1)[1]
        r6_init, t_init = kfs[0].rW2C, kfs[0].tW2C
        est.estimate(uvs, xyz, p_idxs, pose_init.f, centre, r6_init, t_init)
        p_inl = est.get_inlier_mask(uvs, xyz, pose_init.f, centre)
        best_h = torch.argmax((p_inl * confs[None]).sum(dim=1))
        save(d, "p4p_idxs", p_idxs); save(d, "p4p_init_r6", r6_init); save(d, "p4p_init_t", t_init)
        save(d, "p4p_models", est.models); save(d, "p4p_inliers", p_inl); save(d, "p4p_best", best_h)
        Rt = est.models[best_h].clone()
        sel_in = p_inl[best_h] > 0
        xyz_i, uv_i = xyz[sel_in], uvs[sel_in]
        N = pose_init.num_pts_miniba_incr
        if len(xyz_i) >= N:
            sel = torch.topk(torch.rand_like(xyz_i[..., 0]), N, dim=0, largest=False)[1]
            xyz_ba, uv_ba = xyz_i[sel], uv_i[sel]
        else:
            xyz_ba = torch.cat([xyz_i, torch.zeros(N - len(xyz_i), 3, device="cuda")])
            uv_ba = torch.cat([uv_i, -torch.ones(N - len(uv_i), 2, device="cuda")])
        R6, t = Rt[:3, :2][None], Rt[:3, 3][None]
        R6o, to, _, _, r, r_init, ba_mask = pose_init.miniBA(R6, t, pose_init.f, xyz_ba, centre, uv_ba.view(-1))
        save(d, "ba_in_xyz", xyz_ba); save(d, "ba_in_uv", uv_ba); save(d, "ba_in_r6", R6[0]); save(d, "ba_in_t", t[0])
        save(d, "ba_out_r6", R6o.reshape(-1, 3, 2)[0]); save(d, "ba_out_t", to.reshape(-1, 3)[0])
        save(d, "ba_r", r); save(d, "ba_r_init", r_init); save(d, "ba_mask", ba_mask)
        save(d, "accepted", np.bool_(bool(ba_mask.sum() > args.min_num_inliers)))
        res = pose_init.init_inference_pose(kfs, kp, k, False, image)
        save(d, "oscd_Rt", res[0] if res is not None else torch.zeros(4, 4))
        save(d, "oscd_accepted", np.bool_(res is not None))

        # ---- change cues at O-SCD's pose, against the real reference render
        if res is None:
            continue
        Rw = res[0]
        view = Camera(colmap_id=f"{k:04d}", R=np.transpose(Rw[:3, :3].cpu().numpy()), T=Rw[:3, 3].cpu().numpy(),
                      FoVx=fovx, FoVy=fovy, image=image[:3], gt_alpha_mask=None,
                      image_name=info["name"].split(".")[0], uid=f"{k:04d}")
        with torch.no_grad():
            ren = render(view, gaussians, pipe, bg)["render"]
            live = view.original_image[:3]
            C = generate_candidate_map(live, ren, sam, 14, height, width)
            ssim_map = ssim(ren, live, window_size=11, map=True).mean(dim=0)
            l1 = torch.abs(ren - live).mean(dim=0)
            delta = 0.8 * l1 + 0.2 * (1 - ssim_map)
            r1024 = F.interpolate(ren[None], size=(1024, 1024), mode="nearest")[0]
            l1024 = F.interpolate(live[None], size=(1024, 1024), mode="nearest")[0]
            stack = torch.stack([r1024, l1024]).half()
            emb = sam.get_image_embeddings(stack)[-1]
        d = os.path.join(out, "cues", f"frame_{k:03d}")
        save(d, "live", live); save(d, "render", ren); save(d, "ssim_map", ssim_map); save(d, "l1", l1)
        save(d, "delta_raw", delta); save(d, "vmin", delta.min()); save(d, "vmax", delta.max())
        save(d, "emb", emb); save(d, "C", C[0])
        np.save(os.path.join(d, "sam_input.npy"), stack.cpu().numpy().astype(np.float16))

    import transformers
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"width": width, "height": height, "f": f, "cx": float(centre[0]), "cy": float(centre[1]),
                   "max_error": max_error, "n_refs": len(refs), "instance": instance,
                   "xfeat_commit": git_head(os.environ["SCD_XFEAT_REPO"]), "oscd_commit": git_head(a.oscd_root),
                   "torch": torch.__version__, "transformers": transformers.__version__}, fh, indent=2)
    print("fixtures written to", out)


if __name__ == "__main__":
    main()
