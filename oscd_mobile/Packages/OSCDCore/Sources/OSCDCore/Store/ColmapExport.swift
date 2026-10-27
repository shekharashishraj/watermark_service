import CoreGraphics
import Foundation
import ImageIO
import simd

/// A recorded scan as a COLMAP text model (PINHOLE camera, ARKit poses, LiDAR points), the input splatx
/// trains the reference from. A line-for-line port of tools/session_to_colmap.py so the phone and the
/// Mac build their models from the same data.
public enum ColmapExport {
    public struct Point: Equatable {
        public var xyz: SIMD3<Double>
        public var rgb: SIMD3<UInt8>
    }

    public struct Result: Equatable {
        public let images: Int
        public let points: Int
    }

    /// Writes `dir/images/<id>.png` and `dir/sparse/0/{cameras,images,points3D}.txt`. With `fullResolution`
    /// and a scan that has full-size images, the images are those JPEGs (`<id>.jpg`) and the camera is scaled
    /// to the capture size; the LiDAR points are the same either way.
    @discardableResult
    /// `maxPoints` keeps every k-th point after the voxel dedupe (the trainers subsample to 150–600k anyway; a room
    /// scan's 4M-point text file cost the phone 1.5 GB of memory before training even started).
    public static func write(session: SessionStore, to dir: URL, voxel: Double = 0.004, fullResolution: Bool = false,
                             maxPoints: Int = 200_000) throws -> Result {
        let m = session.manifest
        let K = m.intrinsics
        let full = fullResolution && session.hasFullImages
        let fm = FileManager.default
        let images = dir.appendingPathComponent("images"), sparse = dir.appendingPathComponent("sparse/0")
        try fm.createDirectory(at: images, withIntermediateDirectories: true)
        try fm.createDirectory(at: sparse, withIntermediateDirectories: true)

        var imageLines = ""
        var voxels: [SIMD3<Int64>: Point] = [:]        // first point per voxel, filled frame by frame
        let lo = m.roi.map { SIMD3<Double>($0.min) }, hi = m.roi.map { SIMD3<Double>($0.max) }
        // At video rate a room has 1,000 keyframes × 49k depth pixels; every 2nd pixel in u and v keeps the
        // export under ~1.5M samples and the table small.
        let pixelStride = m.keyframes.count > 300 ? 2 : 1
        for (n, kf) in m.keyframes.enumerated() {
            let rgb = try session.array(.keyframes, id: kf.id, name: "rgb").uint8()
            let (depth, conf) = try session.depth(.keyframes, id: kf.id)
            let name = String(format: full ? "%06d.jpg" : "%06d.png", kf.id)
            if full {
                try? fm.removeItem(at: images.appendingPathComponent(name))
                try fm.copyItem(at: session.fullImageURL(.keyframes, id: kf.id), to: images.appendingPathComponent(name))
            } else {
                try writePNG(rgb: rgb, width: K.width, height: K.height, to: images.appendingPathComponent(name))
            }
            let (R, t) = worldToCamera(arkitCameraToWorld: kf.arkitCameraToWorld)
            let q = qvec(R)
            imageLines += "\(n + 1) \(q.x) \(q.y) \(q.z) \(q.w) \(t.x) \(t.y) \(t.z) 1 \(name)\n\n"
            for p in lidarPoints(depth: depth, confidence: conf, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                                 intrinsics: K, arkitCameraToWorld: kf.arkitCameraToWorld, rgb: rgb, pixelStride: pixelStride) {
                if let lo, let hi, !(all(p.xyz .>= lo) && all(p.xyz .<= hi)) { continue }
                let key = SIMD3<Int64>(Int64((p.xyz.x / voxel).rounded(.down)), Int64((p.xyz.y / voxel).rounded(.down)),
                                       Int64((p.xyz.z / voxel).rounded(.down)))
                if voxels[key] == nil { voxels[key] = p }
            }
        }
        var points = voxels.sorted { a, b in
            a.key.x != b.key.x ? a.key.x < b.key.x : a.key.y != b.key.y ? a.key.y < b.key.y : a.key.z < b.key.z
        }.map(\.value)
        voxels = [:]
        if points.count > maxPoints {
            let stride = Double(points.count) / Double(maxPoints)
            points = (0..<maxPoints).map { points[Int(Double($0) * stride)] }
        }

        // Pixel centres scale as (c + 0.5)·s − 0.5.
        let s = full ? Float(m.captureWidth) / Float(K.width) : 1
        let (cw, ch) = full ? (m.captureWidth, m.captureHeight) : (K.width, K.height)
        try "1 PINHOLE \(cw) \(ch) \(K.f * s) \(K.f * s) \((K.cx + 0.5) * s - 0.5) \((K.cy + 0.5) * s - 0.5)\n"
            .write(to: sparse.appendingPathComponent("cameras.txt"), atomically: true, encoding: .utf8)
        try imageLines.write(to: sparse.appendingPathComponent("images.txt"), atomically: true, encoding: .utf8)
        var pointLines = ""
        pointLines.reserveCapacity(points.count * 64)
        for (i, p) in points.enumerated() {
            pointLines += "\(i + 1) \(p.xyz.x) \(p.xyz.y) \(p.xyz.z) \(p.rgb.x) \(p.rgb.y) \(p.rgb.z) 0\n"
        }
        try pointLines.write(to: sparse.appendingPathComponent("points3D.txt"), atomically: true, encoding: .utf8)
        return Result(images: m.keyframes.count, points: points.count)
    }

    /// ARKit camera-to-world (column-major) → OpenCV-axes camera-to-world → its inverse (R, t).
    static func cameraToWorld(_ v: [Float]) -> simd_double4x4 {
        let m = simd_double4x4(columns: (SIMD4(Double(v[0]), Double(v[1]), Double(v[2]), Double(v[3])),
                                         SIMD4(Double(v[4]), Double(v[5]), Double(v[6]), Double(v[7])),
                                         SIMD4(Double(v[8]), Double(v[9]), Double(v[10]), Double(v[11])),
                                         SIMD4(Double(v[12]), Double(v[13]), Double(v[14]), Double(v[15]))))
        return m * simd_double4x4(diagonal: SIMD4(1, -1, -1, 1))
    }

    static func worldToCamera(arkitCameraToWorld v: [Float]) -> (R: simd_double3x3, t: SIMD3<Double>) {
        let c2w = cameraToWorld(v)
        let Rc = simd_double3x3(SIMD3(c2w[0].x, c2w[0].y, c2w[0].z), SIMD3(c2w[1].x, c2w[1].y, c2w[1].z),
                                SIMD3(c2w[2].x, c2w[2].y, c2w[2].z))
        let R = Rc.transpose
        return (R, -(R * SIMD3(c2w[3].x, c2w[3].y, c2w[3].z)))
    }

    /// Rotation matrix → COLMAP (qw, qx, qy, qz) as (x, y, z, w) of the SIMD4. session_to_colmap.rot_to_qvec.
    static func qvec(_ R: simd_double3x3) -> SIMD4<Double> {
        func m(_ r: Int, _ c: Int) -> Double { R[c][r] }
        let tr = m(0, 0) + m(1, 1) + m(2, 2)
        if tr > 0 {
            let s = 0.5 / (tr + 1).squareRoot()
            return SIMD4(0.25 / s, (m(2, 1) - m(1, 2)) * s, (m(0, 2) - m(2, 0)) * s, (m(1, 0) - m(0, 1)) * s)
        }
        let diag = [m(0, 0), m(1, 1), m(2, 2)]
        let i = diag.firstIndex(of: diag.max()!)!, j = (i + 1) % 3, k = (i + 2) % 3
        let s = 2 * (1 + m(i, i) - m(j, j) - m(k, k)).squareRoot()
        var q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m(k, j) - m(j, k)) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m(j, i) + m(i, j)) / s
        q[1 + k] = (m(k, i) + m(i, k)) / s
        return SIMD4(q[0], q[1], q[2], q[3])
    }

    /// High-confidence LiDAR depth → world points, coloured from the nearest working-resolution pixel.
    static func lidarPoints(depth: [Float], confidence: [UInt8], depthWidth dw: Int, depthHeight dh: Int,
                            intrinsics K: Intrinsics, arkitCameraToWorld v: [Float], rgb: [UInt8], pixelStride: Int = 1) -> [Point] {
        let c2w = cameraToWorld(v)
        let s = Double(dw) / Double(K.width)
        let fd = Double(K.f) * s, cxd = (Double(K.cx) + 0.5) * s - 0.5, cyd = (Double(K.cy) + 0.5) * s - 0.5
        var out: [Point] = []
        out.reserveCapacity(dw * dh)
        for v in stride(from: 0, to: dh, by: pixelStride) {
            for u in stride(from: 0, to: dw, by: pixelStride) {
                let i = v * dw + u
                let z = Double(depth[i])
                guard confidence[i] >= 2, z.isFinite, z > 0 else { continue }
                let cam = SIMD4((Double(u) - cxd) / fd * z, (Double(v) - cyd) / fd * z, z, 1)
                let w = c2w * cam
                // np.round rounds half to even.
                let uw = min(max(Int(((Double(u) + 0.5) / s - 0.5).rounded(.toNearestOrEven)), 0), K.width - 1)
                let vw = min(max(Int(((Double(v) + 0.5) / s - 0.5).rounded(.toNearestOrEven)), 0), K.height - 1)
                let p = (vw * K.width + uw) * 3
                out.append(Point(xyz: SIMD3(w.x, w.y, w.z), rgb: SIMD3(rgb[p], rgb[p + 1], rgb[p + 2])))
            }
        }
        return out
    }

    /// Keep points inside the region, then one point per voxel: the first, in sorted voxel order
    /// (np.unique(..., axis=0, return_index=True)).
    static func cropAndDedupe(_ pts: [Point], roi: AABB, voxel: Double) -> [Point] {
        let lo = SIMD3<Double>(roi.min), hi = SIMD3<Double>(roi.max)
        return dedupe(pts.filter { all($0.xyz .>= lo) && all($0.xyz .<= hi) }, voxel: voxel)
    }

    static func dedupe(_ pts: [Point], voxel: Double) -> [Point] {
        var first: [SIMD3<Int64>: Int] = [:]
        for (i, p) in pts.enumerated() {
            let key = SIMD3<Int64>(Int64((p.xyz.x / voxel).rounded(.down)), Int64((p.xyz.y / voxel).rounded(.down)),
                                   Int64((p.xyz.z / voxel).rounded(.down)))
            if first[key] == nil { first[key] = i }
        }
        return first.sorted { a, b in
            a.key.x != b.key.x ? a.key.x < b.key.x : a.key.y != b.key.y ? a.key.y < b.key.y : a.key.z < b.key.z
        }.map { pts[$0.value] }
    }

    static func writePNG(rgb: [UInt8], width: Int, height: Int, to url: URL) throws {
        try writeImage(rgb: rgb, width: width, height: height, to: url, type: "public.png", properties: nil)
    }

    public static func writeJPEG(rgb: [UInt8], width: Int, height: Int, to url: URL, quality: Double = 0.92) throws {
        try writeImage(rgb: rgb, width: width, height: height, to: url, type: "public.jpeg",
                       properties: [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    }

    static func writeImage(rgb: [UInt8], width: Int, height: Int, to url: URL, type: String, properties: CFDictionary?) throws {
        let data = Data(rgb) as CFData
        guard let provider = CGDataProvider(data: data),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, type as CFString, 1, nil) else {
            throw SessionStore.StoreError.missingFile("could not encode \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(dest, image, properties)
        guard CGImageDestinationFinalize(dest) else { throw SessionStore.StoreError.missingFile("could not write \(url.lastPathComponent)") }
    }
}
