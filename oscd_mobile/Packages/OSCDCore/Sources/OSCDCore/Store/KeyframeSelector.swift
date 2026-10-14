import simd

/// Spec §7.1: a keyframe every ≥ 5 cm or ≥ 10°, only while tracking is normal and the camera is
/// moving slower than 0.3 m/s and 30°/s (estimated from consecutive frames).
public struct KeyframeSelector: Sendable {
    public var minTranslation: Float = 0.05
    public var minRotationDeg: Float = 10
    public var maxSpeed: Float = 0.3
    public var maxAngularSpeedDeg: Float = 30
    private var lastKeyframe: Pose?
    private var previous: (pose: Pose, time: Double)?

    public init() {}

    /// `pose` is world-to-camera. Returns true when this frame should become a keyframe.
    public mutating func consider(pose: Pose, time: Double, trackingNormal: Bool) -> Bool {
        defer { previous = (pose, time) }
        guard trackingNormal else { return false }
        if let prev = previous, time > prev.time {
            let dt = Float(time - prev.time)
            if simd_distance(pose.centre, prev.pose.centre) / dt > maxSpeed { return false }
            if pose.rotationDegrees(to: prev.pose) / dt > maxAngularSpeedDeg { return false }
        }
        guard let last = lastKeyframe else {
            lastKeyframe = pose
            return true
        }
        if simd_distance(pose.centre, last.centre) >= minTranslation || pose.rotationDegrees(to: last) >= minRotationDeg {
            lastKeyframe = pose
            return true
        }
        return false
    }
}
