import simd

/// Pinhole intrinsics with one focal length, as O-SCD uses (`f = mean(fx, fy)`).
public struct Intrinsics: Codable, Equatable, Sendable {
    public var f: Float
    public var cx: Float
    public var cy: Float
    public var width: Int
    public var height: Int

    public init(f: Float, cx: Float, cy: Float, width: Int, height: Int) {
        self.f = f; self.cx = cx; self.cy = cy; self.width = width; self.height = height
    }

    /// O-SCD's convention: principal point at ((W-1)/2, (H-1)/2).
    public static func oscdCentered(f: Float, width: Int, height: Int) -> Intrinsics {
        Intrinsics(f: f, cx: Float(width - 1) / 2, cy: Float(height - 1) / 2, width: width, height: height)
    }

    /// ARKit intrinsics (for the capture size) scaled to a working size. Uses O-SCD's single focal
    /// mean(fx, fy) and keeps ARKit's calibrated principal point (spec deviation D10).
    public static func fromARKit(fx: Float, fy: Float, cx: Float, cy: Float,
                                 captureWidth: Int, captureHeight: Int,
                                 workingWidth: Int, workingHeight: Int) -> Intrinsics {
        let sx = Float(workingWidth) / Float(captureWidth)
        let sy = Float(workingHeight) / Float(captureHeight)
        return Intrinsics(f: 0.5 * (fx * sx + fy * sy),
                          cx: (cx + 0.5) * sx - 0.5, cy: (cy + 0.5) * sy - 0.5,
                          width: workingWidth, height: workingHeight)
    }

    public var centre: SIMD2<Float> { SIMD2(cx, cy) }

    /// O-SCD `pts2px`: f * xy / z + centre.
    @inline(__always) public func project(_ p: SIMD3<Float>) -> SIMD2<Float> {
        f * SIMD2(p.x, p.y) / p.z + centre
    }

    /// O-SCD `depth2points`: ((uv - centre) / f, 1) * depth.
    @inline(__always) public func unproject(_ uv: SIMD2<Float>, depth: Float) -> SIMD3<Float> {
        let xy = (uv - centre) / f
        return SIMD3(xy.x, xy.y, 1) * depth
    }
}

/// World-to-camera rigid transform in OpenCV axes (x right, y down, z forward): p_cam = R p_world + t.
public struct Pose: Equatable, Sendable {
    public var R: simd_float3x3
    public var t: SIMD3<Float>

    public init(R: simd_float3x3, t: SIMD3<Float>) { self.R = R; self.t = t }

    public static let identity = Pose(R: matrix_identity_float3x3, t: .zero)

    @inline(__always) public func apply(_ p: SIMD3<Float>) -> SIMD3<Float> { R * p + t }

    public var inverse: Pose {
        let Rt = R.transpose
        return Pose(R: Rt, t: -(Rt * t))
    }

    /// Camera centre in world coordinates.
    public var centre: SIMD3<Float> { -(R.transpose * t) }

    /// self ∘ other: apply `other` first, then `self`.
    public func compose(_ other: Pose) -> Pose { Pose(R: R * other.R, t: R * other.t + t) }

    public var matrix: simd_float4x4 {
        simd_float4x4(columns: (SIMD4(R.columns.0, 0), SIMD4(R.columns.1, 0), SIMD4(R.columns.2, 0), SIMD4(t, 1)))
    }

    public init(matrix m: simd_float4x4) {
        R = simd_float3x3(columns: (SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                    SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                    SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
        t = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    /// ARKit `ARCamera.transform` is camera-to-world with camera axes x right, y up, z backward.
    /// Flip y and z to OpenCV axes, then invert to world-to-camera.
    public static func fromARKitCameraToWorld(_ m: simd_float4x4) -> Pose {
        let flip = simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))
        return Pose(matrix: m * flip).inverse
    }

    /// Angle of the relative rotation between two poses, in degrees.
    public func rotationDegrees(to other: Pose) -> Float {
        let Rrel = R.transpose * other.R
        let c = (Rrel.columns.0.x + Rrel.columns.1.y + Rrel.columns.2.z - 1) / 2
        return acos(min(1, max(-1, c))) * 180 / .pi
    }

    /// O-SCD 6D rotation of R (see SixD) in Double precision for the solvers.
    public var r6: [Double] { SixD.fromMatrix(simd_double3x3(converting: R)) }
    public var tDouble: SIMD3<Double> { SIMD3<Double>(t) }

    public init(r6: [Double], t: SIMD3<Double>) {
        R = simd_float3x3(converting: SixD.toMatrix(r6))
        self.t = SIMD3<Float>(t)
    }
}

/// O-SCD's 6D rotation: the first two columns of R, stored row-major as a 3×2 matrix
/// [r00, r01, r10, r11, r20, r21] (torch `R[..., :2]`), turned back into a rotation by
/// Gram–Schmidt (`sixD2mtx`).
public enum SixD {
    public static func toMatrix(_ p: ArraySlice<Double>) -> simd_double3x3 {
        let i = p.startIndex
        let a = SIMD3(p[i], p[i + 2], p[i + 4])
        let b = SIMD3(p[i + 1], p[i + 3], p[i + 5])
        let b1 = a / simd_length(a)
        var b2 = b - simd_dot(b1, b) * b1
        b2 /= simd_length(b2)
        return simd_double3x3(columns: (b1, b2, simd_cross(b1, b2)))
    }

    public static func toMatrix(_ p: [Double]) -> simd_double3x3 { toMatrix(p[0..<6]) }

    public static func fromMatrix(_ R: simd_double3x3) -> [Double] {
        [R.columns.0.x, R.columns.1.x, R.columns.0.y, R.columns.1.y, R.columns.0.z, R.columns.1.z]
    }
}

public extension simd_float3x3 {
    init(converting m: simd_double3x3) {
        self.init(columns: (SIMD3<Float>(m.columns.0), SIMD3<Float>(m.columns.1), SIMD3<Float>(m.columns.2)))
    }
}

public extension simd_double3x3 {
    init(converting m: simd_float3x3) {
        self.init(columns: (SIMD3<Double>(m.columns.0), SIMD3<Double>(m.columns.1), SIMD3<Double>(m.columns.2)))
    }
}

public extension simd_float4x4 {
    /// 16 values, column by column (the layout ARKit and simd use).
    var columnMajor: [Float] {
        [columns.0, columns.1, columns.2, columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }

    init(columnMajor v: [Float]) {
        precondition(v.count == 16)
        self.init(columns: (SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                            SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15])))
    }
}
