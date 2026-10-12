"""Render everything the harness page shows from a real held-out scenario."""
import base64, json, sys
import cv2
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scene_change.harness.scenario import load_scenario
from scene_change.harness.raycast import BoxRenderer
from scene_change.harness.evaluate import gt_masks
from scene_change.harness.robustness import change_masks
from scene_change import stress
from scene_change.geometry import invert_pose

OUT = sys.argv[1]
B = str(ROOT / 'runs' / 'bench' / 'scenarios')
sc = load_scenario(f'{B}/s000_mixed')
scene, iscene = sc['scene'], sc['inspection_scene']
insp, base = sc['inspection'], sc['baseline']
A = sc['gt_arrays']
K, W, H = insp.K, insp.width, insp.height

def png(rgb):
    ok, b = cv2.imencode('.png', cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR))
    return 'data:image/png;base64,' + base64.b64encode(b).decode()

def depth_img(d, dmax=5.0):
    v = np.clip(d / dmax, 0, 1)
    c = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[d <= 0] = (20, 20, 20)
    return c

def oid_img(o):
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 230, (int(o.max()) + 2, 3)).astype(np.uint8)
    pal[0] = (15, 15, 15)
    return pal[o]

def overlay(rgb, m, col=(255, 70, 40)):
    out = rgb.astype(np.float32).copy()
    out[m] = 0.45 * out[m] + 0.55 * np.array(col)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = out.astype(np.uint8)
    cv2.drawContours(out, cnts, -1, col, 1)
    return out

def hexc(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return '#%02x%02x%02x' % tuple((c * 255).astype(int))

def footprint(obj):
    polys = []
    for ctr, half, yaw, mat, _ in obj.world_boxes():
        cs, sn = np.cos(yaw), np.sin(yaw)
        pts = []
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = dx * half[0], dy * half[1]
            pts.append([round(float(ctr[0] + cs * x - sn * y), 3), round(float(ctr[1] + sn * x + cs * y), 3)])
        polys.append({'p': pts, 'z': round(float(ctr[2] + half[2]), 2), 'c': hexc(scene_mat(obj, mat).color)})
    return polys

def scene_mat(obj, mat):
    return MATS[mat]

def plan_of(s, mats):
    global MATS
    MATS = mats
    objs = []
    for o in s.objects.values():
        if o.category == 'structure' and o.tags.get('surface') in ('floor', 'ceiling'):
            continue
        objs.append({'oid': o.oid, 'name': o.name, 'cat': o.category, 'room': o.room, 'boxes': footprint(o)})
    return objs

T_bw = np.asarray(sc['gt']['T_baseline_world'])
T_iw = np.asarray(sc['gt']['T_inspection_world'])
true_i = A['insp_poses_world']
true_b = A['base_poses_world']
noisy_i = np.stack([invert_pose(T_iw) @ T for T in insp.poses])
drift_cm = np.linalg.norm(noisy_i[:, :3, 3] - true_i[:, :3, 3], axis=1) * 100

data = {
    'scene': {'width': scene.width, 'depth': scene.depth, 'height': scene.height,
              'rooms': [{'name': r.name, 'x0': r.x0, 'y0': r.y0, 'x1': r.x1, 'y1': r.y1} for r in scene.rooms],
              'objects_before': plan_of(scene, scene.materials),
              'objects_after': plan_of(iscene, iscene.materials),
              'counts': {}},
    'paths': {'baseline': true_b[:, :2, 3].round(3).tolist(), 'inspection': true_i[:, :2, 3].round(3).tolist(),
              'inspection_device': noisy_i[:, :2, 3].round(3).tolist(),
              'insp_dir': [[round(float(t[0, 2]), 3), round(float(t[1, 2]), 3)] for t in true_i],
              'kinds': insp.meta['kinds'], 'rooms': insp.meta['rooms'],
              'drift_cm_max': round(float(drift_cm.max()), 1), 'drift_cm_end': round(float(drift_cm[-1]), 1)},
    'frames': {'baseline': len(base), 'inspection': len(insp), 'size': [W, H]},
}
from collections import Counter
data['scene']['counts'] = dict(Counter(o.category for o in scene.objects.values()))

# --- what one frame carries -------------------------------------------------
gt = gt_masks(A, sc['gt']['changes'])
cm = change_masks(A, sc['gt']['changes'])
per = gt.sum(axis=(1, 2))
f = int(np.argmax(per))
lb, li = sc['lighting']['baseline'], sc['lighting']['inspection']
rb = BoxRenderer(scene)
before_render = rb.render(K, true_i[f], W, H, lb)
data['anatomy'] = {
    'frame': f, 'room': insp.meta['rooms'][f],
    'rgb': png(insp.rgb[f]), 'depth': png(depth_img(insp.depth[f])), 'oid': png(oid_img(A['insp_oid'][f])),
    'before': png((np.clip(before_render['rgb'], 0, 1) * 255).astype(np.uint8)),
    'mask': png(overlay(insp.rgb[f], gt[f])),
    'dropout': round(float(np.mean(insp.depth[f] == 0)), 3),
}

# --- each change: before / after / ground truth -----------------------------
changes = []
for c in sc['gt']['changes']:
    m = cm[c['id']]
    n = m.sum(axis=(1, 2))
    if n.max() == 0:
        continue
    k = int(np.argmax(n))
    br = rb.render(K, true_i[k], W, H, lb)
    changes.append({'id': c['id'], 'type': c['type'], 'name': c['name'].replace('_', ' '), 'room': c['room'],
                    'detail': c.get('detail', ''), 'frame': k, 'frames_visible': int((n > 0).sum()),
                    'before': png((np.clip(br['rgb'], 0, 1) * 255).astype(np.uint8)),
                    'after': png(insp.rgb[k]), 'mask': png(overlay(insp.rgb[k], m[k]))})
data['changes'] = changes

# --- stressor ladder on one frame -------------------------------------------
g = f
sub = insp.subset([max(g - 1, 0), g, min(g + 1, len(insp) - 1)])
ladder = []
for name, lv in [('blur', 0.02), ('blur', 0.07), ('dark', -2.0), ('dark', -4.0), ('bright', 2.0)]:
    s2, _ = stress.apply_stressor(name, lv, sub)
    ladder.append({'name': name, 'level': lv, 'img': png(s2.rgb[1])})
for lv in (0.0, 3.0):
    s2 = stress.relight_inspection(sc, lv, frames=[g])
    ladder.append({'name': 'relight', 'level': lv, 'img': png(s2.rgb[0])})
for lv in (2.0, 8.0):
    s2 = stress.depth_session(insp.subset([g]), lv)
    ladder.append({'name': 'depth', 'level': lv, 'img': png(depth_img(s2.depth[0]))})
data['ladder'] = {'clean': png(insp.rgb[g]), 'clean_depth': png(depth_img(insp.depth[g])), 'items': ladder}

# --- partial scenario: what the walk skipped --------------------------------
sp = load_scenario(f'{B}/s000_partial')
data['partial'] = {'skipped': sp['gt']['skipped_rooms'], 'half': sp['gt']['partial_rooms'],
                   'path': sp['gt_arrays']['insp_poses_world'][:, :2, 3].round(3).tolist(),
                   'rooms': [{'name': r.name, 'x0': r.x0, 'y0': r.y0, 'x1': r.x1, 'y1': r.y1} for r in sp['scene'].rooms],
                   'n_changes': len(sp['gt']['changes'])}
data['lighting'] = {'exposure_base': round(lb.exposure, 3), 'exposure_insp': round(li.exposure, 3),
                    'lamps_insp': len(li.lamps), 'lamps_base': len(lb.lamps)}
json.dump(data, open(OUT, 'w'))
print('ok', len(changes), 'changes; frame', f, round(len(json.dumps(data)) / 1e6, 2), 'MB')
