import simd

/// Spec §7.1 plus PocketGS's information gate. A frame is eligible ≥ 2 cm or ≥ 2° from the last keyframe
/// (video rate: a room walk gives ~1,000 keyframes, the count the scanner apps train from; it was 5 cm / 10°),
/// only while tracking is normal and the camera is moving slower than 0.6 m/s and 30°/s, both averaged over
/// the last `speedWindowSeconds` (0.25 s) of frames, so hand jitter and single 60 fps intervals do not
/// decide. The first eligible frame opens a 0.25 s window; the sharpest of up to 8 eligible frames in it
/// becomes the keyframe. While the camera turns too fast nothing is eligible, so the keyframe waits until
/// the turn slows: on the 2026-10-25 room scan the keyframes taken at 35–40°/s were motion-blurred and the
/// model's worst views. (A 15°/s limit over 0.1 s was tried: ordinary handling sits at 15°/s, so it
/// refused half the keyframes and left 7 s holes.) The speed limit was 0.3 m/s (spec §7.1); a room scan at
/// walking pace broke it for metres at a time.
///
/// `Candidate` is whatever the caller keeps for a frame (its image, depth, pose). `sharpness` and `candidate`
/// are evaluated only for eligible frames, and `candidate` only when the frame is the window's sharpest so far.
public struct KeyframeSelector<Candidate> {
    public var minTranslation: Float = 0.02
    public var minRotationDeg: Float = 2
    public var maxSpeed: Float = 0.6
    public var maxAngularSpeedDeg: Float = 30
    public var speedWindowSeconds: Double = 0.25
    /// True after `consider` when the frame was turned away for moving or turning too fast.
    public private(set) var movingTooFast = false
    /// One keyframe per window, so the window sets the keyframe rate: 0.1 s / 4 frames ≈ 10 per second at
    /// 60 fps (0.25 s / 8 frames gave 3.4 per second on the 2026-10-26 room walk, a third of video rate).
    public var windowSeconds: Double = 0.1
    public var maxCandidates = 4

    public struct Chosen {
        public var candidate: Candidate
        public var sharpness: Double
        /// Eligible frames the keyframe was chosen from.
        public var candidates: Int
        /// The window's first frame's sharpness: the frame a first-past-the-gate selector would have kept.
        public var firstSharpness: Double
    }

    private struct Window {
        var start: Double
        var count = 0
        var firstSharpness = 0.0
        var best: (pose: Pose, sharpness: Double, candidate: Candidate)?
    }

    private var lastKeyframe: Pose?
    /// Recent frames, oldest first, spanning about `speedWindowSeconds`.
    private var recent: [(pose: Pose, time: Double)] = []
    private var window: Window?

    public init() {}

    /// `pose` is world-to-camera. Returns a keyframe when this frame closes a window (at most one per call).
    public mutating func consider(pose: Pose, time: Double, trackingNormal: Bool,
                                  sharpness: () -> Double, candidate: () -> Candidate) -> Chosen? {
        defer {
            recent.append((pose, time))
            while recent.count > 2, time - recent[1].time >= speedWindowSeconds { recent.removeFirst() }
        }
        var chosen: Chosen?
        if let w = window, w.count >= maxCandidates || time - w.start > windowSeconds { chosen = closeWindow() }
        movingTooFast = false
        guard trackingNormal else { return chosen }
        if let oldest = recent.first, time > oldest.time {
            let dt = Float(time - oldest.time)
            movingTooFast = simd_distance(pose.centre, oldest.pose.centre) / dt > maxSpeed
                || pose.rotationDegrees(to: oldest.pose) / dt > maxAngularSpeedDeg
            if movingTooFast { return chosen }
        }
        if let last = lastKeyframe,
           simd_distance(pose.centre, last.centre) < minTranslation, pose.rotationDegrees(to: last) < minRotationDeg {
            return chosen
        }
        if window == nil { window = Window(start: time) }
        let s = sharpness()
        if window!.count == 0 { window!.firstSharpness = s }
        window!.count += 1
        if window!.best.map({ s > $0.sharpness }) ?? true { window!.best = (pose, s, candidate()) }
        return chosen
    }

    /// Ends the recording: the open window's sharpest frame, if any.
    public mutating func finish() -> Chosen? {
        defer { recent = [] }
        return closeWindow()
    }

    private mutating func closeWindow() -> Chosen? {
        defer { window = nil }
        guard let w = window, let best = w.best else { return nil }
        lastKeyframe = best.pose
        return Chosen(candidate: best.candidate, sharpness: best.sharpness, candidates: w.count, firstSharpness: w.firstSharpness)
    }
}
