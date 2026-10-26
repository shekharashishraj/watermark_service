import Foundation
import OSCDCore

/// One Detect run's SAM2 embeddings on disk (spec §4.2), so the object post-pass can reuse them without
/// re-encoding: `Caches/detect/<runID>/<frameID>.emb` holds the live then the render embedding as IEEE
/// binary16 (2 × 2 MB for 256·64·64). Written on the fusion thread, read on the detect thread once fusion
/// has finished; `removeAll()` deletes the folder, and writes after it are ignored, so a late fusion job
/// cannot leave a file behind.
final class EmbeddingCache {
    enum CacheError: LocalizedError {
        case sizeMismatch(Int, Int), corrupt(Int)
        var errorDescription: String? {
            switch self {
            case let .sizeMismatch(a, b): return "The live and render features differ in size (\(a) and \(b))."
            case let .corrupt(id): return "Frame \(id)'s cached features are unreadable."
            }
        }
    }

    let directory: URL
    private let lock = NSLock()
    private var removed = false          // guarded by lock

    /// `Caches/detect/<runID>/`, created empty. Folders left by an earlier run that never reached its cleanup
    /// (the app was killed) are removed first: only one Detect run exists at a time.
    init(runID: String) throws {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("detect")
        try? FileManager.default.removeItem(at: root)
        directory = root.appendingPathComponent(runID, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func write(frameID: Int, embLive: [Float], embRender: [Float]) throws {
        guard embLive.count == embRender.count else { throw CacheError.sizeMismatch(embLive.count, embRender.count) }
        var data = Data(capacity: 4 * embLive.count)
        for v in [embLive, embRender] { Float16Pack.pack(v).withUnsafeBufferPointer { data.append($0) } }
        lock.lock(); defer { lock.unlock() }
        guard !removed else { return }
        try data.write(to: url(frameID))
    }

    func read(frameID: Int) throws -> (embLive: [Float], embRender: [Float]) {
        let data = try Data(contentsOf: url(frameID))
        guard data.count % 4 == 0, !data.isEmpty else { throw CacheError.corrupt(frameID) }
        let n = data.count / 4, half = 2 * n
        func halves(from start: Int) -> [UInt16] {
            [UInt16](unsafeUninitializedCapacity: n) { buf, count in
                count = data.copyBytes(to: buf, from: start..<start + half) / 2
            }
        }
        return (Float16Pack.unpack(halves(from: 0)), Float16Pack.unpack(halves(from: half)))
    }

    /// Deletes the run's folder; idempotent.
    func removeAll() {
        lock.lock(); defer { lock.unlock() }
        removed = true
        try? FileManager.default.removeItem(at: directory)
    }

    private func url(_ frameID: Int) -> URL { directory.appendingPathComponent("\(frameID).emb") }
}
