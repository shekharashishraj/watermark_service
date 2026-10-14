import CoreML

/// XFeat network in Core ML (tools/convert_xfeat_coreml.py) plus O-SCD's post-processing.
public final class XFeatModel {
    public let model: MLModel
    public let inputWidth: Int
    public let inputHeight: Int

    public init(compiledModelURL url: URL, inputWidth: Int = 480, inputHeight: Int = 352,
                computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
    }

    /// Compiles an .mlpackage (tests and Mac tools; the app bundle ships compiled models).
    public static func compile(_ packageURL: URL) throws -> URL { try MLModel.compileModel(at: packageURL) }

    /// O-SCD Detector: RGB image in [0, 1] at working resolution → keypoints in that image.
    public func detect(_ image: FloatImage, topK: Int = 6144) throws -> DescribedKeypoints {
        let x = image.resizedBilinear(width: inputWidth, height: inputHeight)
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        x.data.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: $0.count) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)]))
        guard let f = out.featureValue(for: "feats")?.multiArrayValue,
              let hm = out.featureValue(for: "heatmap")?.multiArrayValue,
              let rel = out.featureValue(for: "reliability")?.multiArrayValue else {
            throw ModelError.missingOutput("feats/heatmap/reliability")
        }
        return XFeatPostprocess.run(feats: f.denseFloats(), heatmap: hm.denseFloats(), reliability: rel.denseFloats(),
                                    h: inputHeight / 8, w: inputWidth / 8, topK: topK,
                                    imageWidth: image.width, imageHeight: image.height)
    }
}
