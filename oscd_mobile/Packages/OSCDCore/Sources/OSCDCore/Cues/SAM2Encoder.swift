import CoreML

/// SAM2.1-hiera-tiny image encoder as O-SCD uses it: each image nearest-resized to 1024×1024,
/// raw [0, 1] values → embedding [256, 64, 64]. O-SCD encodes [render, live] as one batch; the
/// encoder has no cross-batch ops, so one image at a time gives the same embeddings and lets the
/// live image be encoded while its pose is still being estimated.
public final class SAM2Encoder {
    public static let side = 1024
    public let model: MLModel

    /// The Neural Engine runs the whole graph; Core ML's GPU path for it is about 20× slower.
    public init(compiledModelURL url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    public func encode(_ image: FloatImage) throws -> [Float] {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: image.channels), NSNumber(value: Self.side), NSNumber(value: Self.side)],
                                   dataType: .float32)
        image.writeResizedNearest(into: arr.dataPointer.assumingMemoryBound(to: Float.self), width: Self.side, height: Self.side)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["images": MLFeatureValue(multiArray: arr)]))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else { throw ModelError.missingOutput("embedding") }
        return e.denseFloats()
    }

    public func encode(render: FloatImage, live: FloatImage) throws -> (render: [Float], live: [Float]) {
        (try encode(render), try encode(live))
    }
}
