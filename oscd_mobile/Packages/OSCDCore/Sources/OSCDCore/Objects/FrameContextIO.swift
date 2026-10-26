import Foundation
import simd

/// What `pose.json` holds besides the pose: the tracker settings the frame was ingested with.
public struct FrameDumpMeta: Codable, Equatable, Sendable {
    public var frameID: Int
    /// The session keyframe the frame came from (equal to `frameID` for Detect dumps).
    public var keyframeID: Int
    /// 4×4 world→camera, row-major (OpenCV axes).
    public var viewmat: [Float]
    public var intrinsics: Intrinsics
    /// `TrackerConfig.supportHeight` of the run (nil: unknown).
    public var supportHeight: Float?
    /// `TrackerConfig.roi` of the run.
    public var roi: AABB?
    /// `TrackerConfig.confirmVoxels` of the run (the Smallest object setting; nil in dumps written before it was kept).
    public var confirmVoxels: Int? = nil
    /// `TrackerConfig.minHeightAboveSupport` of the run (nil in older dumps).
    public var minHeightAboveSupport: Float? = nil
}

/// A `FrameContext` on disk, one folder per frame, so the Mac can replay exactly what the phone's object post-pass ingested
/// (`oscd-objects`). Embeddings are stored as binary16 bits, as the phone caches them.
public enum FrameContextIO {
    public static func folderName(_ frameID: Int) -> String { String(format: "%06d", frameID) }

    /// Writes `<dir>/<frameID padded to 6>/` with changed.npy and valid.npy (b1), dLive.npy and dRen.npy (f4, [H,W]),
    /// live.npy and render.npy (f4, [3,H,W]), embLive.npy and embRender.npy (u2 = `Float16Pack` bits, [256,64,64]),
    /// dBefore.npy (f4, [H,W]) when the frame has one, and pose.json. `supportHeight`, `roi`, `confirmVoxels` and
    /// `minHeightAboveSupport` are the run's `TrackerConfig` values; `keyframeID` defaults to the frame ID.
    public static func write(_ f: FrameContext, to dir: URL, supportHeight: Float? = nil, roi: AABB? = nil,
                             keyframeID: Int? = nil, confirmVoxels: Int? = nil, minHeightAboveSupport: Float? = nil) throws {
        let folder = dir.appendingPathComponent(folderName(f.frameID), isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let w = f.intrinsics.width, h = f.intrinsics.height
        func save(_ a: NPYArray, _ name: String) throws { try a.write(to: folder.appendingPathComponent(name + ".npy")) }
        try save(NPYArray(bools: f.changed, shape: [h, w]), "changed")
        try save(NPYArray(bools: f.valid, shape: [h, w]), "valid")
        try save(NPYArray(floats: f.dLive, shape: [h, w]), "dLive")
        try save(NPYArray(floats: f.dRen, shape: [h, w]), "dRen")
        try save(NPYArray(floats: f.live.data, shape: [f.live.channels, f.live.height, f.live.width]), "live")
        try save(NPYArray(floats: f.render.data, shape: [f.render.channels, f.render.height, f.render.width]), "render")
        try save(NPYArray(uint16: Float16Pack.pack(f.embLive), shape: embShape), "embLive")
        try save(NPYArray(uint16: Float16Pack.pack(f.embRender), shape: embShape), "embRender")
        if let b = f.dBefore { try save(NPYArray(floats: b, shape: [h, w]), "dBefore") }
        let m = f.pose.matrix
        let viewmat = (0..<4).flatMap { r in (0..<4).map { c in m[c][r] } }
        let meta = FrameDumpMeta(frameID: f.frameID, keyframeID: keyframeID ?? f.frameID, viewmat: viewmat,
                                 intrinsics: f.intrinsics, supportHeight: supportHeight, roi: roi,
                                 confirmVoxels: confirmVoxels, minHeightAboveSupport: minHeightAboveSupport)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(meta).write(to: folder.appendingPathComponent("pose.json"), options: .atomic)
    }

    public static func readMeta(frameDir: URL) throws -> FrameDumpMeta {
        try JSONDecoder().decode(FrameDumpMeta.self, from: Data(contentsOf: frameDir.appendingPathComponent("pose.json")))
    }

    /// One frame folder written by `write`.
    public static func read(frameDir: URL) throws -> FrameContext {
        let meta = try readMeta(frameDir: frameDir)
        let k = meta.intrinsics, n = k.width * k.height
        func load(_ name: String, count: Int) throws -> NPYArray {
            let a = try NPYArray.load(frameDir.appendingPathComponent(name + ".npy"))
            guard a.count == count else { throw NPYError.badHeader("\(name).npy holds \(a.count) values, expected \(count)") }
            return a
        }
        func image(_ name: String) throws -> FloatImage {
            FloatImage(width: k.width, height: k.height, channels: 3, data: try load(name, count: 3 * n).floats())
        }
        guard meta.viewmat.count == 16 else { throw NPYError.badHeader("viewmat needs 16 values") }
        let v = meta.viewmat
        let pose = Pose(matrix: simd_float4x4(rows: [SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                                                     SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15])]))
        let embCount = embShape.reduce(1, *)
        return FrameContext(
            frameID: meta.frameID, pose: pose, intrinsics: k,
            changed: try load("changed", count: n).bools(), valid: try load("valid", count: n).bools(),
            dLive: try load("dLive", count: n).floats(), dRen: try load("dRen", count: n).floats(),
            live: try image("live"), render: try image("render"),
            embLive: Float16Pack.unpack(try bits(load("embLive", count: embCount))),
            embRender: Float16Pack.unpack(try bits(load("embRender", count: embCount))),
            dBefore: FileManager.default.fileExists(atPath: frameDir.appendingPathComponent("dBefore.npy").path)
                ? try load("dBefore", count: n).floats() : nil)
    }

    /// Every frame folder under `dir` (those holding a pose.json), sorted by frame ID.
    public static func readAll(in dir: URL) throws -> [FrameContext] {
        try frameDirs(in: dir).map { try read(frameDir: $0.url) }
    }

    /// Frame folders under `dir`, sorted by frame ID.
    public static func frameDirs(in dir: URL) throws -> [(id: Int, url: URL)] {
        try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .compactMap { u in
                guard let id = Int(u.lastPathComponent),
                      FileManager.default.fileExists(atPath: u.appendingPathComponent("pose.json").path) else { return nil }
                return (id, u)
            }
            .sorted { $0.id < $1.id }
    }

    /// The binary16 bits of a u2 array.
    private static func bits(_ a: NPYArray) throws -> [UInt16] {
        guard a.dtype == .u2 else { throw NPYError.unsupportedDType(a.dtype.rawValue) }
        return [UInt16](unsafeUninitializedCapacity: a.count) { buffer, initialized in
            _ = a.payload.copyBytes(to: buffer)
            initialized = a.count
        }
    }

    private static let embShape = [256, 64, 64]
}
