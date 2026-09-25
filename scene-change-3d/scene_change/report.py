"""Self-contained HTML inspection report with a 3D viewer (three.js).

    write_report(model, session, result, "report.html", title="Unit 4B turnover")

The page embeds everything it needs except three.js and its orbit controls (loaded
from public CDNs): a subsampled baseline point cloud coloured by verification status,
change markers, unverified regions, the inspection path, and per-change image pairs
(baseline render vs. inspection frame from the same viewpoint). Pass an O-SCD result
as ``oscd`` to add its per-Gaussian change field as a layer.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .detect import STATUS, InspectionResult
from .geometry import invert_pose
from .model import BaselineModel
from .render import render_gaussians
from .session import Session

TYPE_LABEL = {"missing": "Missing", "added": "Added", "moved": "Moved", "appearance": "Restyled / stained"}
TYPE_RGB = {"missing": (196, 61, 40), "added": (46, 139, 58), "moved": (44, 95, 196), "appearance": (183, 128, 15)}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _png(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _project(P, K, T_wc, W, H):
    T = invert_pose(T_wc)
    pc = P @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    ok = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def best_view(points: np.ndarray, session: Session, poses: np.ndarray, tol: float = 0.08):
    """Frame that sees most of ``points`` unoccluded (ties: closer to the image centre)."""
    H, W = session.depth.shape[1:]
    best, best_score = -1, 0.0
    for i in range(len(session)):
        u, v, z, ok = _project(points, session.K, poses[i], W, H)
        if ok.sum() == 0:
            continue
        d = session.depth[i][v[ok].astype(int), u[ok].astype(int)]
        vis = (d <= 0) | (z[ok] <= d + tol + 0.02 * z[ok])
        n = float(vis.sum())
        if n == 0:
            continue
        cu, cv_ = u[ok][vis].mean(), v[ok][vis].mean()
        centred = 1.0 - 0.5 * (abs(cu - W / 2) / W + abs(cv_ - H / 2) / H)
        score = n / len(points) * centred
        if score > best_score:
            best, best_score = i, score
    return best


def _outline(img: np.ndarray, pts: np.ndarray, K, T_wc, color, scale: int) -> np.ndarray:
    H, W = img.shape[:2]
    out = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    u, v, _, ok = _project(pts, K, T_wc, W, H)
    if ok.sum() >= 3:
        x0, x1 = np.percentile(u[ok], [2, 98]) * scale
        y0, y1 = np.percentile(v[ok], [2, 98]) * scale
        pad = 3 * scale
        cv2.rectangle(out, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), color, 2)
    return out


def _change_images(ch, model: BaselineModel, session: Session, poses: np.ndarray, scale: int = 2):
    pts = [p for p in (ch.points, ch.points_to) if p is not None and len(p)]
    if not pts:
        return None
    P = np.concatenate(pts).astype(np.float64)
    i = best_view(P, session, poses)
    if i < 0:
        return None
    H, W = session.depth.shape[1:]
    ref = render_gaussians(model.gaussians, session.K, poses[i], W, H)["rgb"]
    ref = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
    col = TYPE_RGB[ch.type]
    return {"frame": int(i),
            "baseline": _png(_outline(ref, P, session.K, poses[i], col, scale)),
            "inspection": _png(_outline(session.rgb[i], P, session.K, poses[i], col, scale))}


def report_data(model: BaselineModel, session: Session, result: InspectionResult, oscd=None,
                max_points: int = 90_000, seed: int = 0) -> dict:
    g = model.gaussians
    rng = np.random.default_rng(seed)
    n = len(g)
    keep = np.arange(n)
    # keep every non-verified point; subsample verified ones
    special = result.status != STATUS["verified"]
    if n > max_points:
        ver = np.nonzero(~special)[0]
        k = max(max_points - int(special.sum()), 0)
        keep = np.sort(np.concatenate([np.nonzero(special)[0], rng.choice(ver, size=min(k, len(ver)), replace=False)]))
    pos = g.means[keep].astype(np.float32)
    col = (np.clip(g.colors[keep], 0, 1) * 255).astype(np.uint8)
    status = result.status[keep].astype(np.uint8)
    data = {
        "points": {"n": int(len(keep)), "pos": _b64(pos), "col": _b64(col), "status": _b64(status)},
        "status_codes": STATUS,
        "floor_z": float(model.floor_z),
        "added": {"n": int(len(result.added_points)), "pos": _b64(result.added_points.astype(np.float32)),
                  "label": _b64(result.added_labels.astype(np.int16))},
        "path": np.round(result.poses[:, :3, 3], 3).tolist(),
        "summary": result.summary(),
        "changes": [],
    }
    for ch in result.changes:
        d = ch.summary()
        d["label"] = TYPE_LABEL[ch.type]
        imgs = _change_images(ch, model, session, result.poses)
        if imgs:
            d.update(imgs)
        data["changes"].append(d)
    if oscd is not None:
        on = np.nonzero(oscd.change_colors > 0.5 + 1e-6)[0]
        data["oscd"] = {"n": int(len(on)), "pos": _b64(g.means[on].astype(np.float32)),
                        "val": _b64(np.clip((oscd.change_colors[on] - 0.5) * 2, 0, 1).astype(np.float32)),
                        "backbone": oscd.backbone}
    return data


def write_report(model: BaselineModel, session: Session, result: InspectionResult, path, title: str | None = None,
                 subtitle: str = "", oscd=None, max_points: int = 90_000) -> Path:
    path = Path(path)
    data = report_data(model, session, result, oscd=oscd, max_points=max_points)
    title = title or f"{session.name} changes"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__SUBTITLE__", html.escape(subtitle))
            .replace("__DATA__", payload))
    path.write_text(page, encoding="utf-8")
    return path


_TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #F2F4F3; --surface: #FFFFFF; --sunken: #E7ECEA; --ink: #18211E; --muted: #5A6964; --rule: #D3DBD8;
  --accent: #1F6F66; --accent-ink: #FFFFFF;
  --missing: #C43D28; --added: #2E8B3A; --moved: #2C5FC4; --appearance: #B7800F; --unverified: #8B919B;
  --ok: #1F7A52; --warn: #A87308; --crit: #C43D28;
  --viewer-bg: #E4EAE7;
  --display: "Archivo", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0F1513; --surface: #17201D; --sunken: #121A17; --ink: #E3EBE8; --muted: #97A8A2; --rule: #2A3632;
    --accent: #5CC3B5; --accent-ink: #0F1513;
    --missing: #F07A62; --added: #6CCB76; --moved: #7EA5F2; --appearance: #E5B046; --unverified: #8E96A1;
    --ok: #62C995; --warn: #E5B046; --crit: #F07A62; --viewer-bg: #141C19;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0F1513; --surface: #17201D; --sunken: #121A17; --ink: #E3EBE8; --muted: #97A8A2; --rule: #2A3632;
  --accent: #5CC3B5; --accent-ink: #0F1513;
  --missing: #F07A62; --added: #6CCB76; --moved: #7EA5F2; --appearance: #E5B046; --unverified: #8E96A1;
  --ok: #62C995; --warn: #E5B046; --crit: #F07A62; --viewer-bg: #141C19;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); padding-inline: 16px; padding-block: 28px 56px; }
.wrap { max-width: 1280px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); font-stretch: 85%; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
h1 { font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 750; line-height: 1.1; }
h2 { font-size: 1.25rem; font-weight: 700; }
h3 { font-size: 1.02rem; font-weight: 650; }
.eyebrow { font: 500 0.72rem/1.2 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.mono, .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
header { display: grid; gap: 14px; }
.headline { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; }
.verdict { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; font: 600 0.95rem/1 var(--body); border: 1.5px solid currentColor; }
.verdict::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: currentColor; }
.verdict.ok { color: var(--ok); } .verdict.warn { color: var(--warn); } .verdict.crit { color: var(--crit); }
.reasons { color: var(--muted); margin: 0; }
.tally { display: flex; flex-wrap: wrap; gap: 10px 26px; padding: 14px 0; border-block: 1px solid var(--rule); }
.tally div { display: grid; gap: 2px; }
.tally b { font: 600 1.45rem/1.1 var(--mono); font-variant-numeric: tabular-nums; }
.main { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 24px; align-items: start; }
@media (max-width: 920px) { .main { grid-template-columns: minmax(0, 1fr); } }
.viewer { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 12px); height: min(72vh, 660px); min-height: 360px; background: var(--viewer-bg); border-radius: 12px; overflow: hidden; border: 1px solid var(--rule); }
@media (max-width: 920px) { .viewer { position: relative; top: 0; height: 58vh; } }
.viewer canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.layers { position: absolute; left: 10px; top: 10px; display: flex; flex-wrap: wrap; gap: 6px; max-width: calc(100% - 20px); }
.layers label { display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px; background: color-mix(in srgb, var(--surface) 88%, transparent); border: 1px solid var(--rule); font-size: 0.8rem; cursor: pointer; user-select: none; }
.layers input { accent-color: var(--accent); margin: 0; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.hint { position: absolute; right: 10px; bottom: 8px; font: 0.72rem var(--mono); color: var(--muted); }
.nowebgl { position: absolute; inset: 0; display: grid; place-items: center; padding: 24px; text-align: center; color: var(--muted); }
.list { display: grid; gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 14px; display: grid; gap: 10px; cursor: pointer; text-align: left; font: inherit; color: inherit; width: 100%; }
.card:hover { border-color: color-mix(in srgb, var(--accent) 45%, var(--rule)); }
.card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.card-top { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.chip { display: inline-flex; align-items: center; gap: 6px; font: 600 0.78rem/1 var(--body); padding: 4px 8px; border-radius: 6px; color: var(--c); background: color-mix(in srgb, var(--c) 13%, transparent); }
.chip::before { content: ""; width: 8px; height: 8px; border-radius: 2px; background: var(--c); }
.flag { font: 500 0.72rem/1 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; padding: 4px 7px; border-radius: 5px; border: 1px dashed var(--warn); color: var(--warn); }
.meta { display: flex; flex-wrap: wrap; gap: 4px 14px; color: var(--muted); font-size: 0.84rem; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.pair figure { margin: 0; display: grid; gap: 4px; }
.pair img { width: 100%; height: auto; border-radius: 6px; image-rendering: pixelated; background: var(--sunken); }
.pair figcaption { font: 0.72rem var(--mono); color: var(--muted); }
.why { margin: 0; font-size: 0.84rem; color: var(--muted); }
.empty { padding: 18px; border: 1px dashed var(--rule); border-radius: 10px; color: var(--muted); }
.lower { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); gap: 24px; }
.panel { display: grid; gap: 12px; align-content: start; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th, td { text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font: 500 0.72rem/1.2 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
td.num { text-align: right; }
.bar { display: inline-block; width: 90px; height: 7px; background: var(--sunken); border-radius: 4px; overflow: hidden; vertical-align: middle; margin-right: 8px; }
.bar > i { display: block; height: 100%; background: var(--ok); }
.bar.low > i { background: var(--warn); }
footer { color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--rule); padding-top: 14px; display: grid; gap: 4px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Walkthrough comparison against baseline</div>
    <h1 id="title">__TITLE__</h1>
    <div class="headline"><span id="verdict" class="verdict">…</span><p id="reasons" class="reasons"></p></div>
    <p class="reasons">__SUBTITLE__</p>
    <div class="tally" id="tally"></div>
  </header>

  <section class="main">
    <div class="viewer" id="viewer" aria-label="3D view of the baseline with detected changes">
      <div class="layers" id="layers"></div>
      <div class="hint">drag to orbit · right-drag to pan · scroll to zoom</div>
    </div>
    <div class="panel">
      <h2>Changes</h2>
      <div class="list" id="list"></div>
    </div>
  </section>

  <section class="lower">
    <div class="panel">
      <h2>Room coverage</h2>
      <div class="tablewrap"><table id="rooms"></table></div>
    </div>
    <div class="panel">
      <h2>Not checked this visit</h2>
      <div class="tablewrap"><table id="unverified"></table></div>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script id="report-data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const S = D.summary;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const bytes = (b64) => { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; };
  const f32 = (b64) => new Float32Array(bytes(b64).buffer);
  const i16 = (b64) => new Int16Array(bytes(b64).buffer);
  const TYPE_VAR = {missing: "--missing", added: "--added", moved: "--moved", appearance: "--appearance"};

  // ---------- summary ----------
  const v = document.getElementById("verdict");
  v.textContent = S.verdict;
  v.className = "verdict " + (S.verdict.startsWith("Guest-ready") ? "ok" : (S.verdict.startsWith("Review") || S.verdict.startsWith("Not verified")) ? "warn" : "crit");
  document.getElementById("reasons").textContent = S.verdict_reasons.join(" · ");
  const bt = S.counts.by_type;
  const tally = [["Confirmed", S.counts.confirmed], ["Needs review", S.counts.needs_review], ["Missing", bt.missing],
                 ["Added", bt.added], ["Moved", bt.moved], ["Restyled", bt.appearance], ["Unchecked areas", S.counts.unverified_regions]];
  document.getElementById("tally").innerHTML = tally.map(([k, n]) => `<div><span class="eyebrow">${k}</span><b>${n}</b></div>`).join("");

  const rooms = document.getElementById("rooms");
  rooms.innerHTML = "<tr><th>Room</th><th>Checked surface</th><th>Changes</th><th>Review</th></tr>" + S.rooms.map((r) =>
    `<tr><td>${esc(r.name)}</td><td><span class="bar ${r.coverage < 0.6 ? "low" : ""}"><i style="width:${Math.round(100 * r.coverage)}%"></i></span><span class="num">${Math.round(100 * r.coverage)}%</span> <span class="num" style="color:var(--muted)">of ${r.surface_m2} m²</span></td><td class="num">${r.changes}</td><td class="num">${r.review}</td></tr>`).join("");
  const unv = document.getElementById("unverified");
  unv.innerHTML = S.unverified.length
    ? "<tr><th>Area</th><th>Room</th><th>Surface</th><th>Size</th></tr>" + S.unverified.slice(0, 14).map((r) =>
        `<tr><td class="num">U${r.id + 1}</td><td>${esc(r.room)}</td><td>${esc(r.surface)}</td><td class="num">${r.area_m2.toFixed(2)} m²</td></tr>`).join("")
      + (S.unverified.length > 14 ? `<tr><td colspan="4">${S.unverified.length - 14} smaller areas not listed</td></tr>` : "")
    : "<tr><td>Every baseline surface was seen again.</td></tr>";

  const reg = S.registration, tm = S.timings;
  document.getElementById("footer").innerHTML =
    `<div>Alignment to baseline: ${Math.round(100 * reg.inlier_ratio)}% of surface points agree (RMSE <span class="num">${(100 * reg.rmse_m).toFixed(1)} cm</span>)${reg.confident ? "" : " · low confidence"}.</div>` +
    `<div>Processing: <span class="num">${tm.frames}</span> frames in <span class="num">${tm.inspection_total_s ?? tm.detect_total_s} s</span>.` +
    (D.oscd ? ` O-SCD change field (${esc(D.oscd.backbone)}) available as a layer.` : "") + `</div>`;

  // ---------- change list ----------
  const list = document.getElementById("list");
  if (!D.changes.length) list.innerHTML = '<div class="empty">No changes found in the areas that were checked.</div>';
  D.changes.forEach((c, k) => {
    const b = document.createElement("button");
    b.className = "card"; b.type = "button"; b.id = "change-" + c.id;
    const where = c.type === "moved" ? `moved ${c.moved_distance} m` : `${c.area_m2} m² ${esc(c.surface)}`;
    b.innerHTML = `<div class="card-top"><span class="chip" style="--c:var(${TYPE_VAR[c.type]})">${esc(c.label)}</span>` +
      (c.review ? '<span class="flag">Needs review</span>' : "") + `</div>` +
      `<div class="meta"><span>${esc(c.room)}</span><span>${where}</span><span class="mono">score ${c.score.toFixed(2)}</span><span class="mono">${c.n_views} views</span></div>` +
      (c.baseline ? `<div class="pair"><figure><img alt="Baseline view" src="${c.baseline}"><figcaption>Baseline</figcaption></figure>` +
                    `<figure><img alt="Inspection frame ${c.frame}" src="${c.inspection}"><figcaption>This visit · frame ${c.frame}</figcaption></figure></div>` : "") +
      (c.reasons.length ? `<p class="why">${c.reasons.map(esc).join(" · ")}</p>` : "");
    b.addEventListener("click", () => focusChange(k));
    list.appendChild(b);
  });

  // ---------- 3D viewer ----------
  const host = document.getElementById("viewer");
  let focusChange = (k) => { document.querySelectorAll(".card").forEach((el, j) => el.classList.toggle("active", j === k)); };
  if (!window.THREE || !THREE.OrbitControls) {
    host.insertAdjacentHTML("beforeend", '<div class="nowebgl">The 3D view needs three.js from a CDN, which did not load. The list and tables still work.</div>');
    return;
  }
  let renderer;
  try { renderer = new THREE.WebGLRenderer({antialias: true}); } catch (e) {
    host.insertAdjacentHTML("beforeend", '<div class="nowebgl">This browser cannot draw WebGL, so the 3D view is off. The list and tables still work.</div>');
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
  camera.up.set(0, 0, 1);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.12;
  const color = (name) => new THREE.Color(css(name) || "#888");

  // baseline points
  const P = f32(D.points.pos), C = bytes(D.points.col), St = bytes(D.points.status), n = D.points.n;
  const code = D.status_codes;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(P, 3));
  const colAttr = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
  geo.setAttribute("color", colAttr);
  let ceiling = D.floor_z + 2.1;
  const baseMat = new THREE.PointsMaterial({size: 0.045, vertexColors: true, sizeAttenuation: true});
  const cloud = new THREE.Points(geo, baseMat);
  scene.add(cloud);

  function paintCloud() {
    const miss = color("--missing"), app = color("--appearance"), unv = color("--unverified");
    const dark = (css("color-scheme") || "").includes("dark");
    for (let i = 0; i < n; i++) {
      let r = C[3 * i] / 255, g = C[3 * i + 1] / 255, b = C[3 * i + 2] / 255;
      const s = St[i];
      if (s === code.missing) { r = miss.r; g = miss.g; b = miss.b; }
      else if (s === code.appearance) { r = app.r; g = app.g; b = app.b; }
      else if (s === code.unverified) { r = unv.r; g = unv.g; b = unv.b; }
      else { const k = dark ? 0.62 : 0.8; r = r * k + (1 - k) * (dark ? 0.1 : 0.9); g = g * k + (1 - k) * (dark ? 0.1 : 0.9); b = b * k + (1 - k) * (dark ? 0.1 : 0.9); }
      colAttr.setXYZ(i, r, g, b);
    }
    colAttr.needsUpdate = true;
  }
  // hide ceilings so rooms read from above
  const idx = []; for (let i = 0; i < n; i++) if (P[3 * i + 2] < ceiling) idx.push(i);
  geo.setIndex(idx);

  // added evidence
  const layers = {};
  const addedGeo = new THREE.BufferGeometry();
  addedGeo.setAttribute("position", new THREE.BufferAttribute(f32(D.added.pos), 3));
  const addedMat = new THREE.PointsMaterial({size: 0.05, sizeAttenuation: true});
  const added = new THREE.Points(addedGeo, addedMat);
  scene.add(added);

  // change boxes
  const markers = new THREE.Group(); scene.add(markers);
  const boxMats = {};
  function boxLines(lo, hi, mat) {
    const g = new THREE.BoxGeometry(hi[0] - lo[0] + 0.06, hi[1] - lo[1] + 0.06, hi[2] - lo[2] + 0.06);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), mat);
    e.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    return e;
  }
  D.changes.forEach((c) => {
    const m = boxMats[c.type] || (boxMats[c.type] = new THREE.LineBasicMaterial({}));
    markers.add(boxLines(c.bbox_min, c.bbox_max, m));
    if (c.type === "moved" && c.moved_to) {
      const a = new THREE.Vector3(...c.centroid), b = new THREE.Vector3(...c.moved_to);
      const dir = b.clone().sub(a); const len = dir.length();
      if (len > 0.05) markers.add(new THREE.ArrowHelper(dir.normalize(), a, len, 0xffffff, Math.min(0.25, len * 0.3), 0.12));
    }
  });
  // unverified regions
  const unvGroup = new THREE.Group(); scene.add(unvGroup);
  const unvMat = new THREE.LineDashedMaterial({dashSize: 0.08, gapSize: 0.06});
  S.unverified.forEach((r) => { const e = boxLines(r.bbox_min, r.bbox_max, unvMat); e.computeLineDistances(); unvGroup.add(e); });
  // path
  const pathGeo = new THREE.BufferGeometry().setFromPoints(D.path.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
  const pathMat = new THREE.LineBasicMaterial({});
  const path = new THREE.Line(pathGeo, pathMat); scene.add(path);
  // O-SCD field
  let oscd = null;
  if (D.oscd && D.oscd.n) {
    const og = new THREE.BufferGeometry();
    og.setAttribute("position", new THREE.BufferAttribute(f32(D.oscd.pos), 3));
    oscd = new THREE.Points(og, new THREE.PointsMaterial({size: 0.06, sizeAttenuation: true, color: 0xd6336c}));
    oscd.visible = false; scene.add(oscd);
  }

  function paintAll() {
    scene.background = color("--viewer-bg");
    paintCloud();
    addedMat.color = color("--added");
    Object.entries(boxMats).forEach(([t, m]) => { m.color = color(TYPE_VAR[t]); });
    markers.children.forEach((o) => { if (o.type === "ArrowHelper") o.setColor(color("--moved")); });
    unvMat.color = color("--unverified");
    pathMat.color = color("--accent");
  }
  paintAll();
  new MutationObserver(paintAll).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintAll);

  // layer toggles
  const defs = [["Baseline", cloud, "--muted"], ["Added", added, "--added"], ["Change boxes", markers, "--missing"],
                ["Not checked", unvGroup, "--unverified"], ["Walk path", path, "--accent"]];
  if (oscd) defs.push(["O-SCD field", oscd, null]);
  const lay = document.getElementById("layers");
  defs.forEach(([name, obj, sw], k) => {
    const id = "layer-" + k;
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" id="${id}" ${obj.visible ? "checked" : ""}>` +
      `<span class="swatch" style="background:${sw ? `var(${sw})` : "#d6336c"}"></span>${name}`;
    l.querySelector("input").addEventListener("change", (e) => { obj.visible = e.target.checked; });
    lay.appendChild(l);
  });

  // framing
  const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute("position"));
  const centre = box.getCenter(new THREE.Vector3()); const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y);
  function home() {
    // back off further on narrow viewers so the whole floor plan fits
    const k = 1 / Math.min(1, Math.max(camera.aspect, 0.45));
    controls.target.copy(centre);
    camera.position.set(centre.x - 0.15 * span * k, centre.y - 0.75 * span * k, centre.z + 0.95 * span * k);
    controls.update();
  }
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let anim = null;
  focusChange = (k) => {
    document.querySelectorAll(".card").forEach((el, j) => el.classList.toggle("active", j === k));
    const c = D.changes[k];
    const tgt = new THREE.Vector3(...c.centroid);
    if (c.type === "moved" && c.moved_to) tgt.add(new THREE.Vector3(...c.moved_to)).multiplyScalar(0.5);
    const off = camera.position.clone().sub(controls.target).normalize().multiplyScalar(3.2);
    off.z = Math.max(off.z, 1.8);
    const to = tgt.clone().add(off);
    if (reduce) { controls.target.copy(tgt); camera.position.copy(to); controls.update(); return; }
    anim = {t0: performance.now(), fromT: controls.target.clone(), fromP: camera.position.clone(), toT: tgt, toP: to};
  };

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host); resize(); home();
  (function loop(now) {
    if (anim) {
      const u = Math.min(1, (now - anim.t0) / 700), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      controls.target.lerpVectors(anim.fromT, anim.toT, e); camera.position.lerpVectors(anim.fromP, anim.toP, e);
      if (u >= 1) anim = null;
    }
    controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop);
  })(performance.now());
})();
</script>
"""
