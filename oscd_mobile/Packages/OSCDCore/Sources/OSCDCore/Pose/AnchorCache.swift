import Foundation
import simd

/// A scan's built anchors on disk, so the anchor step runs once per scan. Stores each keyframe's 3D anchors
/// (not the matches, which pose estimation does not use) with a fingerprint of the keypoints they index:
/// a cache written for other keypoints, or other build settings (`key`), is ignored.
public enum AnchorCache {
    static let magic = "OSCDANC1"

    public static func save(_ anchors: AnchorSet, key: String, to url: URL) throws {
        var d = Data()
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { d.append(contentsOf: $0) } }
        func putString(_ s: String) { let b = Data(s.utf8); put(Int64(b.count)); d.append(b) }
        putString(magic)
        putString(key)
        put(Int64(anchors.keyframes.count)); put(anchors.maxError); put(Int64(anchors.anchorCount)); put(anchors.yield)
        for kf in anchors.keyframes {
            put(Int64(kf.keypoints.count)); put(fingerprint(kf.keypoints))
            d.append(contentsOf: kf.hasPt3d.map { $0 ? UInt8(1) : 0 })
            kf.pts3d.withUnsafeBytes { d.append(contentsOf: $0) }
            kf.conf.withUnsafeBytes { d.append(contentsOf: $0) }
        }
        try d.write(to: url, options: .atomic)
    }

    /// The cached anchors for these keyframes, or nil when there is no cache or it does not match them.
    public static func load(from url: URL, key: String, keyframes input: [(pose: Pose, keypoints: DescribedKeypoints)],
                            intrinsics: Intrinsics) -> AnchorSet? {
        guard let d = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        var at = 0
        func get<T>(_: T.Type) -> T? {
            let n = MemoryLayout<T>.size
            guard at + n <= d.count else { return nil }
            defer { at += n }
            return d.subdata(in: at..<(at + n)).withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        }
        func getString() -> String? {
            guard let n = get(Int64.self), n >= 0, at + Int(n) <= d.count else { return nil }
            defer { at += Int(n) }
            return String(decoding: d.subdata(in: at..<(at + Int(n))), as: UTF8.self)
        }
        func getArray<T>(_: T.Type, _ count: Int) -> [T]? {
            let n = count * MemoryLayout<T>.stride
            guard at + n <= d.count else { return nil }
            defer { at += n }
            return d.subdata(in: at..<(at + n)).withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
        }
        guard getString() == magic, getString() == key, get(Int64.self) == Int64(input.count),
              let maxError = get(Float.self), let anchorCount = get(Int64.self), let yield = get(Double.self) else { return nil }
        var kfs: [KeyframeAnchors] = []
        for (i, v) in input.enumerated() {
            guard get(Int64.self) == Int64(v.keypoints.count), get(Double.self) == fingerprint(v.keypoints),
                  let has = getArray(UInt8.self, v.keypoints.count),
                  let pts = getArray(SIMD3<Float>.self, v.keypoints.count),
                  let conf = getArray(Float.self, v.keypoints.count) else { return nil }
            var kf = KeyframeAnchors(index: i, pose: v.pose, keypoints: v.keypoints)
            kf.hasPt3d = has.map { $0 != 0 }
            kf.pts3d = pts
            kf.conf = conf
            kfs.append(kf)
        }
        return AnchorSet(keyframes: kfs, intrinsics: intrinsics, maxError: maxError, anchorCount: Int(anchorCount), yield: yield)
    }

    static func fingerprint(_ k: DescribedKeypoints) -> Double {
        k.kpts.enumerated().reduce(0) { $0 + Double($1.element.x) * Double($1.offset % 7 + 1) + Double($1.element.y) }
    }
}
