import Foundation
import OSCDCore
import simd

// oscd-objects <frames-dir> [--roi minx,miny,minz,maxx,maxy,maxz] [--support <y> | --no-support] [--out <objects.json>]
//              [--min-voxels N] [--accept S] [--low S] [--verbose]
//              [--before-session <scan-dir> [--splat N] | --no-before] [--max-offset <m>] [--min-height <m>]
//              [--min-hit-rate <f>] [--crops <dir>] [--offsets] [--sweep]
//
// Replays the object layer on the Mac over the frames a Detect run dumped (`-dumpFrames YES`, FrameContextIO):
// every frame is ingested in ID order, the matched pairs get their ICP motion, and each event is printed with its
// score, voxel counts and centroid height above the support. The ROI and support height come from the first frame's
// pose.json (the run's TrackerConfig) unless given; --no-support turns the support gate off. A dump's own dBefore
// (the before-scan's LiDAR in each view) is used unless --no-before; --before-session computes it from a pulled
// before-scan instead. --sweep replays a grid of the calibration candidates (reference depth × frame gate ×
// support-gate height × hit rate) and prints one line of non-hint events per setting (no ICP).
let usage = """
usage: oscd-objects <frames-dir> [--roi minx,miny,minz,maxx,maxy,maxz] [--support <y> | --no-support] [--out <objects.json>]
                    [--min-voxels N] [--accept S] [--low S] [--verbose]
                    [--before-session <scan-dir> [--splat N] | --no-before] [--max-offset <m>] [--min-height <m>]
                    [--min-hit-rate <f>] [--crops <dir>] [--offsets] [--sweep]
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

var args = Array(CommandLine.arguments.dropFirst())
guard !args.isEmpty, !args.contains("--help"), !args.contains("-h") else { print(usage); exit(args.isEmpty ? 2 : 0) }
let framesDir = URL(fileURLWithPath: args.removeFirst())

var roiArg: AABB?, supportArg: Float?, noSupport = false, outURL: URL?
var minVoxels: Int?, accept: Float?, low: Float?, verbose = false
var beforeSession: URL?, splatSize: Int?, noBefore = false, maxOffset: Float?, minHeight: Float?, minHitRate: Float?
var cropsDir: URL?, showOffsets = false, sweep = false
func value(_ i: inout Int, _ flag: String) -> String {
    i += 1
    guard i < args.count else { fail("\(flag) needs a value\n\(usage)") }
    return args[i]
}
func number(_ i: inout Int, _ flag: String) -> Float {
    guard let v = Float(value(&i, flag)) else { fail("\(flag) needs a number") }
    return v
}
var i = 0
while i < args.count {
    switch args[i] {
    case "--roi":
        let v = value(&i, "--roi").split(separator: ",").compactMap { Float($0) }
        guard v.count == 6 else { fail("--roi needs minx,miny,minz,maxx,maxy,maxz") }
        roiArg = AABB(min: SIMD3(v[0], v[1], v[2]), max: SIMD3(v[3], v[4], v[5]))
    case "--support": supportArg = number(&i, "--support")
    case "--no-support": noSupport = true
    case "--out": outURL = URL(fileURLWithPath: value(&i, "--out"))
    case "--min-voxels":
        guard let n = Int(value(&i, "--min-voxels")) else { fail("--min-voxels needs an integer") }
        minVoxels = n
    case "--accept": accept = number(&i, "--accept")
    case "--low": low = number(&i, "--low")
    case "--verbose": verbose = true
    case "--before-session": beforeSession = URL(fileURLWithPath: value(&i, "--before-session"))
    case "--splat":
        guard let n = Int(value(&i, "--splat")), n >= 1 else { fail("--splat needs a positive integer") }
        splatSize = n
    case "--no-before": noBefore = true
    case "--max-offset": maxOffset = number(&i, "--max-offset")
    case "--min-height": minHeight = number(&i, "--min-height")
    case "--min-hit-rate": minHitRate = number(&i, "--min-hit-rate")
    case "--crops": cropsDir = URL(fileURLWithPath: value(&i, "--crops"))
    case "--offsets": showOffsets = true
    case "--sweep": sweep = true
    default: fail("unknown argument \(args[i])\n\(usage)")
    }
    i += 1
}

let frameDirs: [(id: Int, url: URL)]
do { frameDirs = try FrameContextIO.frameDirs(in: framesDir) } catch { fail("cannot read \(framesDir.path): \(error.localizedDescription)") }
guard let first = frameDirs.first else { fail("no frame folders (with a pose.json) under \(framesDir.path)") }
let firstMeta = try FrameContextIO.readMeta(frameDir: first.url)

var config = TrackerConfig()
config.roi = roiArg ?? firstMeta.roi
config.supportHeight = noSupport ? nil : (supportArg ?? firstMeta.supportHeight)
if let minVoxels { config.confirmVoxels = minVoxels }
if let accept { config.matchAccept = accept }
if let low { config.matchLow = low }
if let maxOffset { config.maxFrameOffset = maxOffset }
if let minHeight { config.minHeightAboveSupport = minHeight }
if let minHitRate { config.minHitRate = minHitRate }

func fmt(_ v: Float?) -> String { v.map { String(format: "%.3f", $0) } ?? "none" }
func fmt(_ b: AABB?) -> String {
    b.map { String(format: "(%.3f, %.3f, %.3f)–(%.3f, %.3f, %.3f)", $0.min.x, $0.min.y, $0.min.z, $0.max.x, $0.max.y, $0.max.z) } ?? "none"
}

// Frames, with the reference depth the replay uses.
var frames = try frameDirs.map { try FrameContextIO.read(frameDir: $0.url) }
var beforeNote = frames.contains { $0.dBefore != nil } ? "from the dump" : "none"
if let beforeSession {
    let t = DispatchTime.now().uptimeNanoseconds
    let store = try SessionStore(open: beforeSession)
    let points = try BeforeDepth.points(of: store, roi: config.roi)
    let ptsMs = Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
    var splatMs: [Double] = [], coverage: [Double] = []
    for k in frames.indices {
        let ts = DispatchTime.now().uptimeNanoseconds
        let d = splatSize.map { BeforeDepth.splat(points: points, pose: frames[k].pose, intrinsics: frames[k].intrinsics, splat: $0) }
            ?? BeforeDepth.splat(points: points, pose: frames[k].pose, intrinsics: frames[k].intrinsics)
        splatMs.append(Double(DispatchTime.now().uptimeNanoseconds - ts) / 1e6)
        frames[k].dBefore = d
        let valid = frames[k].valid.filter { $0 }.count
        let covered = zip(d, frames[k].valid).filter { $0.0.isFinite && $0.1 }.count
        coverage.append(Double(covered) / Double(max(valid, 1)))
    }
    beforeNote = String(format: "%@ (%d points, %.0f ms; splat %d, %.1f ms per frame, %.0f %% of valid pixels covered)",
                        store.root.lastPathComponent, points.count, ptsMs, splatSize ?? 3,
                        splatMs.reduce(0, +) / Double(max(splatMs.count, 1)), 100 * coverage.reduce(0, +) / Double(max(coverage.count, 1)))
    // The table estimate the phone makes from the same scan (R37: off below a 15 % share).
    if let band = try SupportPlane.densestBand(of: store, roi: config.roi) {
        print(String(format: "before-scan support band: y = %.3f m · share %.1f %% of %d points · %@", band.height, band.share * 100,
                     band.points, band.isReliable ? "reliable" : "below the 15 % floor: the object layer would be off"))
    } else {
        print("before-scan support band: no LiDAR points in the ROI: the object layer would be off")
    }
} else if noBefore {
    for k in frames.indices { frames[k].dBefore = nil }
    beforeNote = "off"
}
let hasBefore = frames.contains { $0.dBefore != nil }

print("frames \(frames.count) (\(first.id)–\(frameDirs.last!.id)) · roi \(fmt(config.roi)) · support y \(fmt(config.supportHeight)) · confirm \(config.confirmFrames) frames, \(config.confirmVoxels) voxels · accept \(config.matchAccept) low \(config.matchLow)")
print("reference depth \(hasBefore ? "before-scan LiDAR" : "model") · dBefore \(beforeNote) · frame gate \(config.maxFrameOffset.map { String(format: "%.0f mm", $0 * 1000) } ?? "off") · min height \(String(format: "%.1f cm", config.minHeightAboveSupport * 100)) · min hit rate \(String(format: "%.2f", config.minHitRate))")

let support = config.supportHeight
func height(_ y: Float) -> String { support.map { String(format: "%+.1f cm", (y - $0) * 100) } ?? "n/a" }

func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.isEmpty ? 0 : (s[(s.count - 1) / 2] + s[s.count / 2]) / 2
}

if showOffsets {
    print("per-frame median offset o = median(dLive − reference) over judged pixels (mm):")
    for f in frames {
        let o = ChangeSign.compute(dLive: f.dLive, dRen: f.dRen, valid: f.valid).offset
        var line = String(format: "  frame %3d · model %+7.1f", f.frameID, o * 1000)
        if let b = f.dBefore {
            line += String(format: " · before-LiDAR %+7.1f", ChangeSign.compute(dLive: f.dLive, dRen: b, valid: f.valid).offset * 1000)
        }
        print(line + " · changed \(f.changed.filter { $0 }.count) px")
    }
}

// MARK: - Sweep

/// The non-hint events of one replay, compactly: kind#id(voxels, centroid y − support in cm, score).
func compact(_ events: [ObjectEvent]) -> String {
    let shown = events.filter { !$0.hint }
    guard !shown.isEmpty else { return "none" }
    func h(_ y: Float) -> String { support.map { String(format: "%+.1f", (y - $0) * 100) } ?? "?" }
    return shown.map { e -> String in
        switch e.kind {
        case .moved:
            return String(format: "moved#%d(%d→%d v, %@→%@ cm, %.2f)", e.id, e.before!.voxels, e.after!.voxels,
                          h(e.before!.centroid.y), h(e.after!.centroid.y), e.confidence)
        default:
            let p = (e.before ?? e.after)!
            return String(format: "%@#%d(%d v, %@ cm, %.2f)", e.kind.rawValue, e.id, p.voxels, h(p.centroid.y), e.confidence)
        }
    }.joined(separator: " ")
}

if sweep {
    print("sweep (no ICP; events: kind#id(voxels, centroid y − support, score)):")
    let refs: [(String, Bool)] = hasBefore ? [("model", false), ("before", true)] : [("model", false)]
    for (refName, useBefore) in refs {
        for gate in [nil, 0.015, 0.02, 0.03, 0.05] as [Float?] {
            for h in [0.01, 0.015, 0.02, 0.03] as [Float] {
                for hit in [0, 0.4, 0.6, 0.8] as [Float] {
                    var c = config
                    c.maxFrameOffset = gate; c.minHeightAboveSupport = h; c.minHitRate = hit
                    let t = ObjectTracker(config: c)
                    var kept = 0
                    for var f in frames {
                        if !useBefore { f.dBefore = nil }
                        if !t.ingest(f).skipped { kept += 1 }
                    }
                    let events = t.events()
                    print(String(format: "  ref %@ · gate %@ · h %.1f cm · hit %2.0f %% · frames %2d/%2d · %d non-hint · %@",
                                 refName, gate.map { String(format: "%2.0f mm", $0 * 1000) } ?? "off  ", h * 100, hit * 100,
                                 kept, frames.count, events.filter { !$0.hint }.count, compact(events)))
                }
            }
        }
    }
    exit(0)
}

// MARK: - One replay

let tracker = ObjectTracker(config: config)
var ingestMs: [Double] = [], skipped: [Int] = []
let started = DispatchTime.now().uptimeNanoseconds
for f in frames {
    let s = tracker.ingest(f)
    ingestMs.append(s.milliseconds)
    if s.skipped { skipped.append(f.frameID) }
}
let fits = tracker.fitMotions()
let totalMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
let medianMs = median(ingestMs)
if !skipped.isEmpty { print("skipped by the frame gate: \(skipped.count) frames (\(skipped.map(String.init).joined(separator: ", ")))") }

let report = tracker.report(beforeID: "mac", afterID: nil, method: "dump",
                            timingsMs: ["Find objects": totalMs, "Ingest (median per frame)": medianMs])

for e in report.events {
    var line = "\(e.summary(matchAccept: config.matchAccept)) · score \(String(format: "%.3f", e.confidence))"
    switch e.kind {
    case .moved:
        let b = e.before!, a = e.after!
        line += " · \(b.voxels) → \(a.voxels) voxels · y−support \(height(b.centroid.y)) → \(height(a.centroid.y))"
        if let m = e.motion {
            line += String(format: " · Δ (%.3f, %.3f, %.3f) m · fit RMS %.1f mm · inliers %.0f %%", m.translation.x, m.translation.y, m.translation.z, m.fitRMS * 1000, m.inlierFraction * 100)
        }
    default:
        let p = (e.before ?? e.after)!
        line += " · \(p.voxels) voxels · \(p.frames.count) frames · centroid (\(String(format: "%.3f, %.3f, %.3f", p.centroid.x, p.centroid.y, p.centroid.z))) · y−support \(height(p.centroid.y))"
    }
    print(line)
}
if report.events.isEmpty { print("no events") }
print(String(format: "%d events · %d frames · %d fits · median ingest %.1f ms · total %.0f ms", report.events.count, ingestMs.count, fits, medianMs, totalMs))

if verbose {
    print("instances:")
    for i in tracker.instances {
        print(String(format: "  id %d · sign %+d · %d voxels · %d frames (%d–%d) · observed %d · %@ · centroid (%.3f, %.3f, %.3f) · y−support %@",
                     i.id, Int(i.sign), i.voxelCount, i.frames.count, i.firstSeen, i.lastSeen, i.observed,
                     i.isConfirmed(config) ? "confirmed" : "unconfirmed", i.centroid.x, i.centroid.y, i.centroid.z, height(i.centroid.y)))
    }
    print("match scores (left × arrived; * = chosen by the assignment):")
    let chosen = Set(tracker.matches().map { [$0.leftID, $0.arrivedID] })
    for l in tracker.instances where l.sign < 0 {
        for a in tracker.instances where a.sign > 0 {
            print(String(format: "  left %d → arrived %d · %.3f%@%@", l.id, a.id, InstanceMatcher.score(l, a),
                         chosen.contains([l.id, a.id]) ? " *" : "",
                         l.isConfirmed(config) && a.isConfirmed(config) ? "" : " (not both confirmed)"))
        }
    }
}

// MARK: - Crops

if let cropsDir {
    try FileManager.default.createDirectory(at: cropsDir, withIntermediateDirectories: true)
    let byID = Dictionary(uniqueKeysWithValues: frames.map { ($0.frameID, $0) })
    var written = 0
    for inst in tracker.instances where inst.isConfirmed(config) {
        guard let crop = inst.bestCrop else { continue }
        let kind = inst.sign > 0 ? "arrived" : inst.sign < 0 ? "left" : "flat"
        let c = inst.centroid
        let name = String(format: "id%03d_%@_%dvox_c%+.3f_%+.3f_%+.3f", inst.id, kind, inst.voxelCount, c.x, c.y, c.z)
        // The best crop (live for arrived/flat, render for left), enlarged 4×.
        try PNG.write(crop, to: cropsDir.appendingPathComponent(name + ".png"), scale: 4)
        // Context: the best frame's live image | render, with the component's box on both.
        if let f = byID[inst.bestFrame] {
            let colour: SIMD3<UInt8> = inst.sign < 0 ? SIMD3(0, 120, 255) : SIMD3(255, 40, 40)
            let r = inst.bestRect, shift = f.live.width + 4
            let other = PixelRect(minX: r.minX + shift, minY: r.minY, maxX: r.maxX + shift, maxY: r.maxY)
            try PNG.write(PNG.sideBySide(f.live, f.render), to: cropsDir.appendingPathComponent(name + "_frame\(f.frameID)_live-render.png"),
                          scale: 2, boxes: [(r, colour), (other, colour)])
        }
        written += 1
    }
    print("crops: \(written) confirmed instances → \(cropsDir.path)")
}

if let outURL {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    enc.dateEncodingStrategy = .iso8601
    try enc.encode(report).write(to: outURL, options: .atomic)
    print("report → \(outURL.path)")
}
