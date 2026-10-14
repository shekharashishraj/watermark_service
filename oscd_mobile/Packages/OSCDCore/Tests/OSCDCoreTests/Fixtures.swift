import XCTest
import simd
@testable import OSCDCore

/// Locates golden tensors dumped from the real O-SCD (tools/dump_oscd_fixtures.py) and converted models.
enum Fixtures {
    /// oscd_mobile/ (this file lives in oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/).
    static let oscdMobile = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    static var root: URL {
        if let p = ProcessInfo.processInfo.environment["OSCD_FIXTURES"] { return URL(fileURLWithPath: p) }
        return oscdMobile.appendingPathComponent("fixtures")
    }
    static var models: URL { oscdMobile.appendingPathComponent("Models") }

    static func dir(_ name: String) throws -> URL {
        let u = root.appendingPathComponent(name)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: u.path),
                          "fixtures missing at \(u.path); run tools/fetch_fixtures.sh (Task 4)")
        return u
    }

    static func modelPackage(_ name: String) throws -> URL {
        let u = models.appendingPathComponent(name)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: u.path), "model missing at \(u.path)")
        return u
    }

    static func npy(_ dir: URL, _ name: String) throws -> NPYArray {
        try NPYArray.load(dir.appendingPathComponent(name + ".npy"))
    }

    static func json(_ url: URL) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }

    static func vec2s(_ a: NPYArray) -> [SIMD2<Float>] {
        let f = a.floats()
        return stride(from: 0, to: f.count, by: 2).map { SIMD2(f[$0], f[$0 + 1]) }
    }

    static func vec3s(_ a: NPYArray) -> [SIMD3<Float>] {
        let f = a.floats()
        return stride(from: 0, to: f.count, by: 3).map { SIMD3(f[$0], f[$0 + 1], f[$0 + 2]) }
    }

    /// numpy row-major 4×4 world-to-camera matrix → Pose.
    static func pose(rowMajor4x4 m: [Float]) -> Pose {
        let R = simd_float3x3(rows: [SIMD3(m[0], m[1], m[2]), SIMD3(m[4], m[5], m[6]), SIMD3(m[8], m[9], m[10])])
        return Pose(R: R, t: SIMD3(m[3], m[7], m[11]))
    }
}
