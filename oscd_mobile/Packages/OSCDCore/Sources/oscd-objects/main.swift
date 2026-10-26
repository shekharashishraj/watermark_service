import Foundation
import OSCDCore
import simd

// oscd-objects <frames-dir> [--roi minx,miny,minz,maxx,maxy,maxz] [--support <y> | --no-support] [--out <objects.json>]
//              [--min-voxels N] [--accept S] [--low S] [--verbose]
//
// Replays the object layer on the Mac over the frames a Detect run dumped (`-dumpFrames YES`, FrameContextIO):
// every frame is ingested in ID order, the matched pairs get their ICP motion, and each event is printed with its
// score, voxel counts and centroid height above the support. The ROI and support height come from the first frame's
// pose.json (the run's TrackerConfig) unless given; --no-support turns the support gate off.
let usage = """
usage: oscd-objects <frames-dir> [--roi minx,miny,minz,maxx,maxy,maxz] [--support <y> | --no-support] [--out <objects.json>]
                    [--min-voxels N] [--accept S] [--low S] [--verbose]
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
func value(_ i: inout Int, _ flag: String) -> String {
    i += 1
    guard i < args.count else { fail("\(flag) needs a value\n\(usage)") }
    return args[i]
}
var i = 0
while i < args.count {
    switch args[i] {
    case "--roi":
        let v = value(&i, "--roi").split(separator: ",").compactMap { Float($0) }
        guard v.count == 6 else { fail("--roi needs minx,miny,minz,maxx,maxy,maxz") }
        roiArg = AABB(min: SIMD3(v[0], v[1], v[2]), max: SIMD3(v[3], v[4], v[5]))
    case "--support":
        guard let y = Float(value(&i, "--support")) else { fail("--support needs a number") }
        supportArg = y
    case "--no-support": noSupport = true
    case "--out": outURL = URL(fileURLWithPath: value(&i, "--out"))
    case "--min-voxels":
        guard let n = Int(value(&i, "--min-voxels")) else { fail("--min-voxels needs an integer") }
        minVoxels = n
    case "--accept":
        guard let s = Float(value(&i, "--accept")) else { fail("--accept needs a number") }
        accept = s
    case "--low":
        guard let s = Float(value(&i, "--low")) else { fail("--low needs a number") }
        low = s
    case "--verbose": verbose = true
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

func fmt(_ v: Float?) -> String { v.map { String(format: "%.3f", $0) } ?? "none" }
func fmt(_ b: AABB?) -> String {
    b.map { String(format: "(%.3f, %.3f, %.3f)–(%.3f, %.3f, %.3f)", $0.min.x, $0.min.y, $0.min.z, $0.max.x, $0.max.y, $0.max.z) } ?? "none"
}
print("frames \(frameDirs.count) (\(first.id)–\(frameDirs.last!.id)) · roi \(fmt(config.roi)) · support y \(fmt(config.supportHeight)) · confirm \(config.confirmFrames) frames, \(config.confirmVoxels) voxels · accept \(config.matchAccept) low \(config.matchLow)")

let tracker = ObjectTracker(config: config)
var ingestMs: [Double] = []
let started = DispatchTime.now().uptimeNanoseconds
for d in frameDirs {
    let f = try FrameContextIO.read(frameDir: d.url)
    ingestMs.append(tracker.ingest(f).milliseconds)
}
var fits = 0
for pair in tracker.pairsNeedingMotion() {
    guard let l = tracker.instances.first(where: { $0.id == pair.leftID }),
          let a = tracker.instances.first(where: { $0.id == pair.arrivedID }) else { continue }
    tracker.setMotion(leftID: l.id, arrivedID: a.id, PlanarICP.fit(source: l.points, target: a.points))
    fits += 1
}
let totalMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
let sorted = ingestMs.sorted()
let median = sorted.isEmpty ? 0 : (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2

let report = tracker.report(beforeID: "mac", afterID: nil, method: "dump",
                            timingsMs: ["Find objects": totalMs, "Ingest (median per frame)": median])
let support = config.supportHeight
func height(_ y: Float) -> String { support.map { String(format: "%+.1f cm", (y - $0) * 100) } ?? "n/a" }

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
print(String(format: "%d events · %d frames · %d fits · median ingest %.1f ms · total %.0f ms", report.events.count, ingestMs.count, fits, median, totalMs))

if verbose {
    print("instances:")
    for i in tracker.instances {
        print(String(format: "  id %d · sign %+d · %d voxels · %d frames (%d–%d) · %@ · centroid (%.3f, %.3f, %.3f) · y−support %@",
                     i.id, Int(i.sign), i.voxelCount, i.frames.count, i.firstSeen, i.lastSeen,
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

if let outURL {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    enc.dateEncodingStrategy = .iso8601
    try enc.encode(report).write(to: outURL, options: .atomic)
    print("report → \(outURL.path)")
}
