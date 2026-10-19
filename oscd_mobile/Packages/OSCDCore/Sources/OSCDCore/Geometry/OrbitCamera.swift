import Foundation
import simd

/// A camera orbiting a target in the ARKit world (y up): yaw about +y, pitch above the horizontal, and a
/// distance. `pose` is world-to-camera in OpenCV axes (x right, y down, z forward), as splatx renders.
public struct OrbitCamera: Equatable, Sendable {
    public static let maxPitch: Float = 1.45          // just short of straight down
    public static let minPitch: Float = -0.3          // a little below the table plane
    public static let minDistance: Float = 0.08
    public static let maxDistance: Float = 3

    public var target: SIMD3<Float>
    public var yaw: Float
    public var pitch: Float
    public var distance: Float

    public init(target: SIMD3<Float>, yaw: Float, pitch: Float, distance: Float) {
        self.target = target; self.yaw = yaw
        self.pitch = min(max(pitch, Self.minPitch), Self.maxPitch)
        self.distance = min(max(distance, Self.minDistance), Self.maxDistance)
    }

    /// The orbit whose eye is `pose`'s camera centre, looking at `target`.
    public init(lookingFrom pose: Pose, at target: SIMD3<Float>) {
        let d = pose.centre - target
        let r = max(simd_length(d), Self.minDistance)
        self.init(target: target, yaw: atan2(d.x, d.z), pitch: asin(min(max(d.y / r, -1), 1)), distance: r)
    }

    public var eye: SIMD3<Float> {
        target + distance * SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    }

    public var pose: Pose {
        let f = simd_normalize(target - eye)                                  // camera z (forward)
        let r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))          // camera x (right)
        let d = simd_cross(f, r)                                              // camera y (down)
        let R = simd_float3x3(rows: [r, d, f])
        return Pose(R: R, t: -(R * eye))
    }

    public mutating func orbit(dYaw: Float, dPitch: Float) {
        yaw += dYaw
        pitch = min(max(pitch + dPitch, Self.minPitch), Self.maxPitch)
    }

    /// factor > 1 moves closer.
    public mutating func zoom(_ factor: Float) {
        distance = min(max(distance / factor, Self.minDistance), Self.maxDistance)
    }
}
