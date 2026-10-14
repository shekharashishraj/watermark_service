import CoreML

/// SAM2.1-hiera-tiny image encoder as O-SCD uses it: both images nearest-resized to 1024×1024,
/// raw [0, 1] values, one batch [render, live] → embeddings [2, 256, 64, 64].
public final class SAM2Encoder {
    public static let side = 1024
    public let model: MLModel

    public init(compiledModelURL url: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    public func encode(render: FloatImage, live: FloatImage) throws -> (render: [Float], live: [Float]) {
        let a = render.resizedNearest(width: Self.side, height: Self.side)
        let b = live.resizedNearest(width: Self.side, height: Self.side)
        let arr = try MLMultiArray(shape: [2, 3, NSNumber(value: Self.side), NSNumber(value: Self.side)], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        a.data.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: $0.count) }
        b.data.withUnsafeBufferPointer { (dst + a.data.count).update(from: $0.baseAddress!, count: $0.count) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["images": MLFeatureValue(multiArray: arr)]))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else { throw ModelError.missingOutput("embedding") }
        let all = e.denseFloats()
        let half = all.count / 2
        return (Array(all[..<half]), Array(all[half...]))
    }
}
