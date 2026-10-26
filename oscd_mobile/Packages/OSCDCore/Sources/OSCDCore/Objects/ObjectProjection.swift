import simd

/// World points to the pixels of the image a view shows, for drawing object events over a camera frame.
public enum ObjectProjection {
    public enum Orientation: Sendable { case none, sensorRight }   // sensorRight: landscape sensor image shown rotated .right

    /// Points nearer than this to the camera plane are not drawn.
    static let minDepth: Float = 0.01
    /// Points farther than this fraction of the image size outside it are not drawn.
    static let outsideFraction: Float = 0.25

    /// Image-space point for a world point, nil when behind the camera (z ≤ 0.01) or more than 25 % outside the image.
    /// The result is in the displayed image's pixel frame (see `displayed`).
    public static func project(_ p: SIMD3<Float>, pose: Pose, intrinsics: Intrinsics, orientation: Orientation) -> SIMD2<Float>? {
        let c = pose.apply(p)
        guard c.z > minDepth else { return nil }
        let uv = intrinsics.project(c)
        let w = Float(intrinsics.width), h = Float(intrinsics.height)
        guard uv.x >= -outsideFraction * w, uv.x <= (1 + outsideFraction) * w,
              uv.y >= -outsideFraction * h, uv.y <= (1 + outsideFraction) * h else { return nil }
        return displayed(uv, width: intrinsics.width, height: intrinsics.height, orientation: orientation)
    }

    /// sensorRight mapping: (u, v) → (H − 1 − v, u); none: identity. Output is in the displayed image's pixel frame.
    public static func displayed(_ uv: SIMD2<Float>, width: Int, height: Int, orientation: Orientation) -> SIMD2<Float> {
        switch orientation {
        case .none: return uv
        case .sensorRight: return SIMD2(Float(height - 1) - uv.y, uv.x)
        }
    }

    public static func displayedSize(width: Int, height: Int, orientation: Orientation) -> (width: Int, height: Int) {
        switch orientation {
        case .none: return (width, height)
        case .sensorRight: return (height, width)
        }
    }
}
