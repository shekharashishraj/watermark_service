/// Everything the object layer needs from one processed live frame.
public struct FrameContext: Sendable {
    public var frameID: Int
    public var pose: Pose
    public var intrinsics: Intrinsics
    public var changed: [Bool]        // fused mask ∧ valid, W·H
    public var valid: [Bool]
    public var dLive: [Float]         // W·H metres, NaN where untrusted
    public var dRen: [Float]          // W·H metres, NaN where the reference is uncovered
    public var live: FloatImage
    public var render: FloatImage     // 3 planes, W×H
    public var embLive: [Float]
    public var embRender: [Float]     // 256·64·64
    /// W·H metres: the before-scan's LiDAR splatted into this view (`BeforeDepth`), NaN where no point lands. When set,
    /// the object layer compares dLive with it instead of dRen (spec §4.4); dRen stays for the render's colour gain.
    public var dBefore: [Float]?

    public init(frameID: Int, pose: Pose, intrinsics: Intrinsics, changed: [Bool], valid: [Bool],
                dLive: [Float], dRen: [Float], live: FloatImage, render: FloatImage,
                embLive: [Float], embRender: [Float], dBefore: [Float]? = nil) {
        self.frameID = frameID; self.pose = pose; self.intrinsics = intrinsics
        self.changed = changed; self.valid = valid
        self.dLive = dLive; self.dRen = dRen
        self.live = live; self.render = render
        self.embLive = embLive; self.embRender = embRender
        self.dBefore = dBefore
    }
}
