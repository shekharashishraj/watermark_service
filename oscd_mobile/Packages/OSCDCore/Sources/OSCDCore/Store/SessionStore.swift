import Foundation
import simd

public struct AABB: Codable, Equatable, Sendable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>
    public init(min: SIMD3<Float>, max: SIMD3<Float>) { self.min = min; self.max = max }
    public func contains(_ p: SIMD3<Float>) -> Bool { all(p .>= min) && all(p .<= max) }
}

public struct FrameRecord: Codable, Equatable, Sendable {
    public var id: Int
    public var timestamp: Double
    public var arkitCameraToWorld: [Float]           // 16 values, column-major
    public var pnpWorldToCamera: [Float]?            // 16 values when a PnP pose was accepted
    public var stats: [String: Double]

    public init(id: Int, timestamp: Double, arkitCameraToWorld: [Float], pnpWorldToCamera: [Float]? = nil,
                stats: [String: Double] = [:]) {
        self.id = id; self.timestamp = timestamp; self.arkitCameraToWorld = arkitCameraToWorld
        self.pnpWorldToCamera = pnpWorldToCamera; self.stats = stats
    }
}

public struct SessionManifest: Codable, Equatable, Sendable {
    public var version = 1
    /// Shown in the scans list; nil means "Scan N" by capture order.
    public var name: String?
    public var deviceModel: String
    public var createdAt: Date
    public var captureWidth: Int
    public var captureHeight: Int
    public var depthWidth: Int
    public var depthHeight: Int
    public var intrinsics: Intrinsics                // at working resolution
    public var roi: AABB?
    public var keyframes: [FrameRecord] = []
    public var sweep: [FrameRecord] = []

    public init(deviceModel: String, captureWidth: Int, captureHeight: Int, depthWidth: Int, depthHeight: Int,
                intrinsics: Intrinsics, createdAt: Date = Date()) {
        self.deviceModel = deviceModel; self.captureWidth = captureWidth; self.captureHeight = captureHeight
        self.depthWidth = depthWidth; self.depthHeight = depthHeight; self.intrinsics = intrinsics; self.createdAt = createdAt
    }
}

/// A recorded session folder. Frame arrays are written before the manifest (atomically), so a crash
/// mid-frame leaves orphan files that no manifest entry points to.
public final class SessionStore {
    public enum Kind: String, Sendable { case keyframes, sweep }
    public enum StoreError: Error, Equatable { case notASession(String), missingFile(String) }

    public let root: URL
    public private(set) var manifest: SessionManifest
    static let manifestName = "session.json"

    public init(create root: URL, manifest: SessionManifest) throws {
        self.root = root
        self.manifest = manifest
        for kind in [Kind.keyframes, .sweep] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(kind.rawValue), withIntermediateDirectories: true)
        }
        try saveManifest()
    }

    public init(open root: URL) throws {
        self.root = root
        let url = root.appendingPathComponent(Self.manifestName)
        guard FileManager.default.fileExists(atPath: url.path) else { throw StoreError.notASession(root.path) }
        manifest = try Self.decoder.decode(SessionManifest.self, from: Data(contentsOf: url))
    }

    func url(_ kind: Kind, _ id: Int, _ name: String) -> URL {
        root.appendingPathComponent(kind.rawValue).appendingPathComponent(String(format: "%06d_%@.npy", id, name))
    }

    public func append(_ kind: Kind, record: FrameRecord, rgb: [UInt8], depth: [Float], confidence: [UInt8],
                       extra: [String: NPYArray] = [:]) throws {
        let K = manifest.intrinsics
        try NPYArray(uint8: rgb, shape: [K.height, K.width, 3]).write(to: url(kind, record.id, "rgb"))
        try NPYArray(floats: depth, shape: [manifest.depthHeight, manifest.depthWidth]).write(to: url(kind, record.id, "depth"))
        try NPYArray(uint8: confidence, shape: [manifest.depthHeight, manifest.depthWidth]).write(to: url(kind, record.id, "conf"))
        for (name, array) in extra { try array.write(to: url(kind, record.id, name)) }
        switch kind {
        case .keyframes: manifest.keyframes.append(record)
        case .sweep: manifest.sweep.append(record)
        }
        try saveManifest()
    }

    /// A keyframe's image at the full capture size (JPEG), for training the model at full resolution.
    public func fullImageURL(_ kind: Kind, id: Int) -> URL {
        root.appendingPathComponent(kind.rawValue).appendingPathComponent(String(format: "%06d_full.jpg", id))
    }

    /// True when every keyframe has its full-size image.
    public var hasFullImages: Bool {
        !manifest.keyframes.isEmpty
            && manifest.keyframes.allSatisfy { FileManager.default.fileExists(atPath: fullImageURL(.keyframes, id: $0.id).path) }
    }

    public func setROI(_ roi: AABB) throws {
        manifest.roi = roi
        try saveManifest()
    }

    public func rgb(_ kind: Kind, id: Int) throws -> FloatImage {
        let a = try load(kind, id, "rgb")
        return FloatImage(rgb8: a.uint8(), width: a.shape[1], height: a.shape[0])
    }

    public func depth(_ kind: Kind, id: Int) throws -> (depth: [Float], confidence: [UInt8]) {
        (try load(kind, id, "depth").floats(), try load(kind, id, "conf").uint8())
    }

    public func array(_ kind: Kind, id: Int, name: String) throws -> NPYArray { try load(kind, id, name) }

    func load(_ kind: Kind, _ id: Int, _ name: String) throws -> NPYArray {
        let u = url(kind, id, name)
        guard FileManager.default.fileExists(atPath: u.path) else { throw StoreError.missingFile(u.lastPathComponent) }
        return try NPYArray.load(u)
    }

    func saveManifest() throws {
        try Self.encoder.encode(manifest).write(to: root.appendingPathComponent(Self.manifestName), options: .atomic)
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
