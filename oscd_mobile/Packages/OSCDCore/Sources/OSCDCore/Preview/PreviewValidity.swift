import simd

/// D3 validity for the live preview: a pixel is judged only when the reference render covers it and
/// its LiDAR point, placed in the reference world with the PnP pose, lies inside the ROI box.
public enum PreviewValidity {
    /// - depth/confidence: ARKit sceneDepth at depthWidth × depthHeight (metres; confidence 2 = high)
    /// - intrinsics: the working-resolution camera (also the size of `render` and of the result)
    /// - pose: world-to-camera pose of this frame in the reference world (O-SCD PnP)
    /// - roi: the reference session's ROI box, in the same world
    public static func mask(depth: [Float], confidence: [UInt8], depthWidth: Int, depthHeight: Int,
                            intrinsics K: Intrinsics, pose: Pose, roi: AABB, render: FloatImage,
                            minCoverage: Float = 0.02) -> [Bool] {
        let W = K.width, H = K.height, n = W * H
        let toWorld = pose.inverse
        var out = [Bool](repeating: false, count: n)
        for v in 0..<H {
            let sv = min(v * depthHeight / H, depthHeight - 1)
            for u in 0..<W {
                let di = sv * depthWidth + min(u * depthWidth / W, depthWidth - 1)
                let z = depth[di]
                guard confidence[di] >= 2, z.isFinite, z > 0 else { continue }
                let i = v * W + u
                guard render.data[i] + render.data[n + i] + render.data[2 * n + i] > minCoverage else { continue }
                out[i] = roi.contains(toWorld.apply(K.unproject(SIMD2(Float(u), Float(v)), depth: z)))
            }
        }
        return out
    }
}

/// The preview picture: changed pixels tinted red, pixels outside the judged area dimmed.
public enum PreviewComposite {
    /// rgb8: interleaved RGB; returns interleaved RGBA (alpha 255).
    public static func overlay(rgb8: [UInt8], changed: [Bool], valid: [Bool]) -> [UInt8] {
        var out = [UInt8](repeating: 255, count: changed.count * 4)
        for i in 0..<changed.count {
            for c in 0..<3 {
                let v = Float(rgb8[i * 3 + c])
                let tint: Float = c == 0 ? 255 : 30
                let shown = !valid[i] ? v * 0.35 : (changed[i] ? 0.4 * v + 0.6 * tint : v)
                out[i * 4 + c] = UInt8(min(255, max(0, shown.rounded())))
            }
        }
        return out
    }
}
