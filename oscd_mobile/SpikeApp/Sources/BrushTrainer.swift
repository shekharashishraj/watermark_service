import BrushC
import Foundation

/// Brush (Apache-2.0, github.com/ArthurBrussee/brush) as the model trainer: adaptive density control, pruning
/// and mip-splatting, which is what gives scanner apps their clean walls. Linked as a static library built
/// for iOS from the remus checkout (see project.yml). `train` blocks; call it on a worker thread.
enum BrushTrainer {
    struct Options {
        var steps: UInt32 = 7000
        var refineEvery: UInt32 = 200
        var maxResolution: UInt32 = 1280
        var maxSplats: UInt32 = 1_000_000
        var shDegree: UInt32 = 2
    }

    enum Failure: Error, LocalizedError {
        case failed, noExport
        var errorDescription: String? {
            switch self {
            case .failed: "Brush training failed"
            case .noExport: "Brush wrote no model"
            }
        }
    }

    private final class Progress {
        let onStep: (Int) -> Void
        init(_ onStep: @escaping (Int) -> Void) { self.onStep = onStep }
    }

    /// Trains from a COLMAP folder (sparse/0 + images) and returns the exported PLY in `outputDir`.
    static func train(dataset: URL, outputDir: URL, options: Options, onStep: @escaping (Int) -> Void) throws -> URL {
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let progress = Progress(onStep)
        let user = Unmanaged.passRetained(progress)
        defer { user.release() }
        let rc: BrushTrainExitCode = dataset.path.withCString { data in
            outputDir.path.withCString { out in
                var o = BrushTrainOptions(total_train_steps: options.steps, refine_every: options.refineEvery,
                                          max_resolution: options.maxResolution, export_every: options.steps, output_path: out,
                                          max_splats: options.maxSplats, sh_degree: options.shDegree)
                return train_and_save(data, &o, { message, userData in
                    guard let userData, message.tag == BrushProgressTraining else { return }
                    Unmanaged<Progress>.fromOpaque(userData).takeUnretainedValue().onStep(Int(message.training.iter))
                }, user.toOpaque())
            }
        }
        guard rc == BrushTrainSuccess else { throw Failure.failed }
        let plys = (try? FileManager.default.contentsOfDirectory(at: outputDir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "ply" }.sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
        guard let last = plys.last else { throw Failure.noExport }
        return last
    }
}
