# O-SCD Mobile, Phase 1 (Foundations, Pose Pipeline, Cues, Spike) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify every engine-independent part of the on-device O-SCD port: XFeat, matching, F-RANSAC, triangulated anchors, P4P-RANSAC + mini-BA pose, the Method A and B change cues, and the SAM2 encoder. Also record tabletop sessions and run the day-3 spike that picks the 3DGS engine.

**Architecture:** Shared code lives in one Swift package, `OSCDCore` (iOS 17 / macOS 14). Each O-SCD component is ported directly from the released code at `3abfeaa` and checked against golden tensors dumped from the real O-SCD on Sol. Core ML runs XFeat and SAM2.1-tiny. Everything else is Swift + Accelerate on the CPU in this phase; work moves to Metal only where the spike shows it's needed. A small iOS spike app records sessions and benchmarks every stage on the iPhone.

**Tech Stack:** Swift 6.4 toolchain in Swift 5 language mode, XCTest, Accelerate (BLAS/vDSP), Core ML, ARKit + RealityKit, SwiftUI, XcodeGen, Python 3.10 (torch, coremltools, transformers) for model conversion, Sol (Slurm) for fixture generation.

**Spec:** `docs/superpowers/specs/2026-10-14-oscd-mobile-design.md`, plus its code reference `oscd_mobile/docs/oscd_code_reference.md`. Read both before starting.

**Scope note:** The spec spans two phases. This plan is **Phase 1** (spec schedule days 1–13 minus fusion). **Phase 2** gets its own plan, written after Task 17's spike report picks the 3DGS engine. Phase 2 covers the SplatEngine adapter, reference build, `R_change` + SSF fusion + refine, the live demo app, Method B replay, RefServer, and rehearsal. The fusion code can't be planned honestly until the engine API is known.

## Global Constraints

- Branch: implement on `oscd-mobile/phase1`, created from `oscd-mobile/design`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Package platforms: `.iOS(.v17), .macOS(.v14)`; `swiftLanguageModes: [.v5]`. No third-party Swift dependencies in `OSCDCore`.
- Working resolution 480×360. XFeat input 480×352 (floor to a multiple of 32, bilinear, `align_corners=False`).
- XFeat: `top_k = 6144`, detection threshold 0.05, 5×5 NMS, `1.5·top_k` candidates.
- Matching: mutual nearest neighbour, cosine similarity > 0.82, no ratio test.
- `max_error = max(2e-3·W, 1.5)` px, which is 1.5 px at W = 480. F-RANSAC uses 2000 hypotheses with Sampson error < `max_error²`.
- Triangulation: up to 8 other keyframes, `min_dis = 30·max_error`, disambiguation ray depths (1, 10) world units (parameterised; see Review Focus and Task 17).
- Pose:
  - retrieve the top 4 keyframes;
  - if there are more than 4000 correspondences, subsample to 2000;
  - P4P-RANSAC: 2000 hypotheses × 20 LM iterations, Huber δ = 1, no MAD;
  - mini-BA: 2000 points, 20 iterations, MAD scale 4;
  - LM settings λ₀ = 1e-5, diagonal floor 1e-2, factor 2;
  - accept when the mask sum is > 100 residual **components** (more than 50 points).
- SSIM: 11×11 Gaussian window (σ = 1.5), zero padding, C1 = 0.01², C2 = 0.03². Pixel cue `0.8·L1 + 0.2·(1−SSIM)`, then min-max with ε = 1e-8.
- SAM2.1-hiera-tiny: nearest resize to 1024×1024, raw [0,1] input (no ImageNet normalisation), batch of 2 (render, live), output = last FPN level + no-memory embedding, shape [2,256,64,64].
- Method B constants:
  - depth tolerance τ = max(1 cm, 2%·z), ramp 2 cm;
  - color cue: gain alignment where |Δdepth| < 1 cm (at least 500 pixels), blur σ = 1 px, ΔE = √((0.25·ΔL*)² + Δa*² + Δb*²), ramp 10 → 25, masked where L* < 15 or L* > 95;
  - `C_B = clamp(C_A + λd·C_depth + λc·C_color, 0, 2)` with λd = λc = 1.
- The method's pose never comes from ARKit. ARKit supplies only reference keyframe poses, LiDAR depth, intrinsics, and display smoothing.
- Seeds default to 0. `fixtures/`, `Models/`, `third_party/` and `tools/.venv/` are gitignored.
- Tests that need fixtures call `XCTSkip` when `oscd_mobile/fixtures/` is missing. Tests that need models skip when `oscd_mobile/Models/` lacks them. Heavy parity tests have `Parity` in their name and run with `swift test -c release --filter Parity`.

## Review Focus

These input classes and failure modes are implied by the spec but not exercised by the parity tests. Each has a test added to its owning task.

1. **A frame with too few matches** (pointing at the wall, motion blur): fewer than 8 matches for F-RANSAC or fewer than 4 correspondences for P4P must produce a clean rejection with a reason, never a crash. This is the case O-SCD's released code crashes on. Tests: Task 7 and Task 11.
2. **Degenerate geometry in the LM solver**: padded dummy points whose camera-frame depth is 0, or a singular `JᵀJ`, must not return a NaN pose. Test: Task 8.
3. **A cue frame with no valid pixels** (camera aimed off the table, so the validity mask is all false): the cues return all zeros with no division blow-ups. Tests: Task 12 and Task 14.
4. **Missing or garbage LiDAR depth** (0, NaN, low confidence): those pixels are excluded from the depth and color cues, not treated as 0 m surfaces. Test: Task 14.
5. **An app killed mid-write**: a session folder whose last frame's files were written but whose manifest wasn't updated still opens cleanly and lists only complete frames. Test: Task 15.

---

## File Structure

```
oscd_mobile/
  .gitignore                                  Task 1
  README.md                                   Task 1 (extended in Tasks 5, 13, 16)
  Packages/OSCDCore/
    Package.swift                             Task 1
    Sources/OSCDCore/
      Util/NPY.swift                          Task 1   .npy read/write
      Util/Random.swift                       Task 7   IndexSampler, SeededSampler, InjectedSampler
      Util/LinearSolve.swift                  Task 8   small dense solver
      Geometry/Camera.swift                   Task 2   Intrinsics, Pose, SixD, matrix helpers
      Image/FloatImage.swift                  Task 3   planar float image + torch-exact resizes
      Image/CameraImage.swift                 Task 15  ARKit YCbCr → RGB area downsample
      Features/DescribedKeypoints.swift       Task 6
      Features/XFeatPostprocess.swift         Task 6   O-SCD Detector post-network steps
      Features/MLMultiArray+Dense.swift       Task 6
      Features/XFeatModel.swift               Task 6   Core ML wrapper
      Matching/Matcher.swift                  Task 7   mutual NN (+ F-RANSAC filtering)
      Matching/FundamentalRANSAC.swift        Task 7
      Pose/MiniBA.swift                       Task 8
      Pose/P4PRANSAC.swift                    Task 9
      Pose/Triangulator.swift                 Task 10
      Pose/AnchorBuilder.swift                Task 10
      Pose/PoseEstimator.swift                Task 11
      Cues/SSIM.swift                         Task 12
      Cues/CuesA.swift                        Task 12
      Cues/SAM2Encoder.swift                  Task 13
      Cues/CuesB.swift                        Task 14
      Store/SessionStore.swift                Task 15
      Store/KeyframeSelector.swift            Task 15
    Sources/oscd-harness/main.swift           Task 17  Mac CLI: anchors yield on a session
    Tests/OSCDCoreTests/
      Fixtures.swift                          Task 1
      NPYTests.swift                          Task 1
      CameraTests.swift                       Task 2
      FloatImageTests.swift                   Task 3
      XFeatTests.swift                        Task 6
      MatcherTests.swift                      Task 7
      MiniBATests.swift                       Task 8
      P4PTests.swift                          Task 9
      TriangulatorTests.swift                 Task 10
      PoseEstimatorTests.swift                Task 11
      CuesATests.swift                        Task 12
      SAM2Tests.swift                         Task 13
      CuesBTests.swift                        Task 14
      StoreTests.swift                        Task 15
  tools/
    requirements-mac.txt                      Task 5
    dump_oscd_fixtures.py                     Task 4   runs on Sol in the scd_robustness `oscd` env
    dump_fixtures.sbatch                      Task 4
    fetch_fixtures.sh                         Task 4
    convert_xfeat_coreml.py                   Task 5
    convert_sam2_coreml.py                    Task 13
    session_to_colmap.py                      Task 17
  SpikeApp/
    project.yml                               Task 16
    Sources/OSCDSpikeApp.swift                Task 16
    Sources/CaptureController.swift           Task 16
    Sources/CaptureView.swift                 Task 16
    Sources/BenchmarkRunner.swift             Task 16
    Sources/BenchmarkView.swift               Task 16
  docs/
    oscd_code_reference.md                    (exists)
    spike_report.md                           Task 17
```

Conventions:
- **Images.** `FloatImage` is planar CHW `Float` in [0, 1].
- **Keypoint and descriptor sets.** These are `DescribedKeypoints`: keypoints as `[SIMD2<Float>]`, plus 64-float descriptor rows, where an all-zero row marks an invalid keypoint.
- **Poses.** `Pose` is world-to-camera in OpenCV axes.
- **Solver precision.** Solvers run in `Double`. Everything that mirrors O-SCD's fp32 tensors runs in `Float`.

---

### Task 1: Package scaffold, `.npy` I/O, fixture helpers

**Files:**
- Create: `oscd_mobile/.gitignore`
- Create: `oscd_mobile/README.md`
- Create: `oscd_mobile/Packages/OSCDCore/Package.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Util/NPY.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/Fixtures.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/NPYTests.swift`

**Interfaces:**
- Produces:
  - `NPYArray` (`shape: [Int]`, `dtype: NPYArray.DType`, `count`).
  - Initializers `init(floats:shape:)`, `init(uint8:shape:)`, `init(uint16:shape:)`, `init(int64:shape:)`, `init(bools:shape:)`.
  - `static func load(_ url: URL) throws -> NPYArray`, `static func parse(_ bytes: Data) throws -> NPYArray`.
  - Conversions `floats() -> [Float]`, `ints() -> [Int]`, `bools() -> [Bool]`, `uint8() -> [UInt8]`.
  - Writing: `encoded() -> Data`, `write(to:) throws`.
  - Test-only `Fixtures`: `root`, `models`, `dir(_:)`, `npy(_:_:)`, `json(_:)`, `vec2s(_:)`, `vec3s(_:)`, `pose(rowMajor4x4:)`, `modelPackage(_:)`.

- [ ] **Step 1: Create the branch and folders**

```bash
cd "/Users/ashishrajshekhar/Desktop/2025/ASU 2/Fall 2025/watermark_service"
git switch oscd-mobile/design && git switch -c oscd-mobile/phase1
mkdir -p oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Util oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests oscd_mobile/tools oscd_mobile/Models oscd_mobile/fixtures
```

- [ ] **Step 2: Write `.gitignore`, `README.md`, `Package.swift`**

`oscd_mobile/.gitignore`:
```gitignore
fixtures/
Models/
third_party/
tools/.venv/
.build/
.swiftpm/
SpikeApp/*.xcodeproj
xcuserdata/
DerivedData/
```

`oscd_mobile/README.md`:
```markdown
# oscd_mobile

On-device port of O-SCD for a live tabletop change-detection demo on an iPhone Pro.
Design: `../docs/superpowers/specs/2026-10-14-oscd-mobile-design.md`.
O-SCD code reference: `docs/oscd_code_reference.md`.

## Layout
- `Packages/OSCDCore`: Swift package with all algorithms (iOS 17 / macOS 14)
- `tools/`: Python model conversion, Sol fixture dump, COLMAP export
- `SpikeApp/`: iOS app for recording sessions and on-device benchmarks
- `fixtures/`, `Models/`, `third_party/`: generated or downloaded, not in git

## Tests
    cd Packages/OSCDCore
    swift test                                  # fast tests; fixture tests skip if fixtures are absent
    swift test -c release --filter Parity       # O-SCD parity tests (need fixtures/ and Models/)
```

`oscd_mobile/Packages/OSCDCore/Package.swift`:
```swift
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "OSCDCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "OSCDCore", targets: ["OSCDCore"])],
    targets: [
        .target(name: "OSCDCore"),
        .testTarget(name: "OSCDCoreTests", dependencies: ["OSCDCore"]),
    ],
    swiftLanguageModes: [.v5]
)
```

- [ ] **Step 3: Write the failing tests**

`Tests/OSCDCoreTests/NPYTests.swift`:
```swift
import XCTest
@testable import OSCDCore

final class NPYTests: XCTestCase {
    func testFloatRoundTrip() throws {
        let a = NPYArray(floats: [1, 2.5, -3, 4, 5, 6], shape: [2, 3])
        let b = try NPYArray.parse(a.encoded())
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(b.dtype, .f4)
        XCTAssertEqual(b.floats(), [1, 2.5, -3, 4, 5, 6])
    }

    func testHeaderPaddedTo64Bytes() {
        let a = NPYArray(floats: [1, 2, 3], shape: [3])
        XCTAssertEqual((a.encoded().count - 3 * 4) % 64, 0)
    }

    /// Bytes laid out the way np.save writes them: magic, v1.0, header padded with spaces so the
    /// preamble is a multiple of 64 bytes and ends in a newline, then the payload.
    func npyBytes(_ header: String, _ payload: Data) -> Data {
        let pad = (64 - (10 + header.utf8.count + 1) % 64) % 64
        let h = header + String(repeating: " ", count: pad) + "\n"
        return Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0, UInt8(h.utf8.count & 0xff), UInt8(h.utf8.count >> 8)]) + Data(h.utf8) + payload
    }

    func testParsesNumpyStyleBytes() throws {
        // Exactly what np.save writes for np.array([True, False, True]).
        let bytes = npyBytes("{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }", Data([1, 0, 1]))
        let a = try NPYArray.parse(bytes)
        XCTAssertEqual(a.shape, [3])
        XCTAssertEqual(a.bools(), [true, false, true])
    }

    func testIntsAndScalars() throws {
        let a = try NPYArray.parse(NPYArray(int64: [7], shape: []).encoded())
        XCTAssertEqual(a.shape, [])
        XCTAssertEqual(a.count, 1)
        XCTAssertEqual(a.ints(), [7])
    }

    func testRejectsBadMagicAndFortranOrder() {
        XCTAssertThrowsError(try NPYArray.parse(Data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))
        let bytes = npyBytes("{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }", Data(count: 4))
        XCTAssertThrowsError(try NPYArray.parse(bytes)) { XCTAssertEqual($0 as? NPYError, .fortranOrder) }
    }

    func testFileRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".npy")
        try NPYArray(uint8: [1, 2, 3, 4], shape: [2, 2]).write(to: url)
        let b = try NPYArray.load(url)
        XCTAssertEqual(b.uint8(), [1, 2, 3, 4])
    }
}
```

`Tests/OSCDCoreTests/Fixtures.swift`:
```swift
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
```
(`Pose` is defined in Task 2. Until then the file won't compile, so in this task write `Fixtures.swift` **without** the `pose(rowMajor4x4:)` function and add that function in Task 2, Step 3.)

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd oscd_mobile/Packages/OSCDCore && swift test --filter NPYTests`
Expected: FAIL to compile with `cannot find 'NPYArray' in scope`.

- [ ] **Step 5: Implement `NPY.swift`**

`Sources/OSCDCore/Util/NPY.swift`:
```swift
import Foundation

public enum NPYError: Error, Equatable {
    case badMagic
    case badHeader(String)
    case unsupportedDType(String)
    case fortranOrder
    case truncated
}

/// Minimal NumPy .npy (v1–v3) reader/writer for little-endian C-order arrays.
public struct NPYArray: Sendable {
    public enum DType: String, Sendable {
        case f4 = "<f4", f8 = "<f8", i4 = "<i4", i8 = "<i8", u1 = "|u1", u2 = "<u2", b1 = "|b1"
        var size: Int {
            switch self {
            case .f4, .i4: return 4
            case .f8, .i8: return 8
            case .u2: return 2
            case .u1, .b1: return 1
            }
        }
    }

    public let shape: [Int]
    public let dtype: DType
    public let payload: Data

    public var count: Int { shape.reduce(1, *) }

    public init(shape: [Int], dtype: DType, payload: Data) {
        precondition(payload.count == shape.reduce(1, *) * dtype.size, "payload size does not match shape")
        self.shape = shape
        self.dtype = dtype
        self.payload = payload
    }

    public init(floats: [Float], shape: [Int]) {
        self.init(shape: shape, dtype: .f4, payload: floats.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(uint8: [UInt8], shape: [Int]) { self.init(shape: shape, dtype: .u1, payload: Data(uint8)) }
    public init(uint16: [UInt16], shape: [Int]) {
        self.init(shape: shape, dtype: .u2, payload: uint16.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(int64: [Int64], shape: [Int]) {
        self.init(shape: shape, dtype: .i8, payload: int64.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(bools: [Bool], shape: [Int]) {
        self.init(shape: shape, dtype: .b1, payload: Data(bools.map { $0 ? 1 : 0 }))
    }

    public static func load(_ url: URL) throws -> NPYArray { try parse(Data(contentsOf: url)) }

    public static func parse(_ bytes: Data) throws -> NPYArray {
        let b = [UInt8](bytes.prefix(12))
        guard b.count >= 10, b[0] == 0x93, Array(b[1...5]) == Array("NUMPY".utf8) else { throw NPYError.badMagic }
        let headerLen: Int, headerStart: Int
        if b[6] == 1 {
            headerLen = Int(b[8]) | Int(b[9]) << 8
            headerStart = 10
        } else {
            guard b.count >= 12 else { throw NPYError.truncated }
            headerLen = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
            headerStart = 12
        }
        guard bytes.count >= headerStart + headerLen else { throw NPYError.truncated }
        let base = bytes.startIndex
        let headerData = bytes.subdata(in: base + headerStart ..< base + headerStart + headerLen)
        guard let header = String(data: headerData, encoding: .ascii) else { throw NPYError.badHeader("not ASCII") }
        guard let descr = capture(header, #"'descr':\s*'([^']+)'"#) else { throw NPYError.badHeader(header) }
        guard let dtype = DType(rawValue: descr) else { throw NPYError.unsupportedDType(descr) }
        if capture(header, #"'fortran_order':\s*(True|False)"#) == "True" { throw NPYError.fortranOrder }
        guard let shapeString = capture(header, #"'shape':\s*\(([^)]*)\)"#) else { throw NPYError.badHeader(header) }
        let shape = shapeString.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let start = base + headerStart + headerLen
        let n = shape.reduce(1, *) * dtype.size
        guard bytes.endIndex - start >= n else { throw NPYError.truncated }
        return NPYArray(shape: shape, dtype: dtype, payload: bytes.subdata(in: start ..< start + n))
    }

    private static func capture(_ s: String, _ pattern: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
              let r = Range(m.range(at: 1), in: s) else { return nil }
        return String(s[r])
    }

    private func raw<T>(_: T.Type) -> [T] {
        [T](unsafeUninitializedCapacity: count) { buffer, initialized in
            _ = payload.copyBytes(to: buffer)
            initialized = count
        }
    }

    public func floats() -> [Float] {
        switch dtype {
        case .f4: return raw(Float.self)
        case .f8: return raw(Double.self).map(Float.init)
        case .i4: return raw(Int32.self).map(Float.init)
        case .i8: return raw(Int64.self).map(Float.init)
        case .u2: return raw(UInt16.self).map(Float.init)
        case .u1, .b1: return raw(UInt8.self).map(Float.init)
        }
    }

    public func ints() -> [Int] {
        switch dtype {
        case .i8: return raw(Int64.self).map(Int.init)
        case .i4: return raw(Int32.self).map(Int.init)
        case .u2: return raw(UInt16.self).map(Int.init)
        case .u1, .b1: return raw(UInt8.self).map(Int.init)
        case .f4, .f8: return floats().map { Int($0) }
        }
    }

    public func bools() -> [Bool] {
        switch dtype {
        case .b1, .u1: return raw(UInt8.self).map { $0 != 0 }
        default: return floats().map { $0 != 0 }
        }
    }

    public func uint8() -> [UInt8] {
        precondition(dtype == .u1 || dtype == .b1, "uint8() needs a u1/b1 array")
        return raw(UInt8.self)
    }

    public func encoded() -> Data {
        let shapeString = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(dtype.rawValue)', 'fortran_order': False, 'shape': \(shapeString), }"
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var out = Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0])
        let len = UInt16(header.utf8.count)
        out.append(contentsOf: [UInt8(len & 0xff), UInt8(len >> 8)])
        out.append(Data(header.utf8))
        out.append(payload)
        return out
    }

    public func write(to url: URL) throws { try encoded().write(to: url, options: .atomic) }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd oscd_mobile/Packages/OSCDCore && swift test --filter NPYTests`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add oscd_mobile/.gitignore oscd_mobile/README.md oscd_mobile/Packages/OSCDCore
git commit -m "Scaffold the OSCDCore package with .npy I/O and fixture helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Camera geometry: intrinsics, poses, 6D rotation, ARKit axes

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Geometry/Camera.swift`
- Modify: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/Fixtures.swift` (add `pose(rowMajor4x4:)` from Task 1)
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/CameraTests.swift`

**Interfaces:**
- Produces:
  - `Intrinsics(f:cx:cy:width:height:)`, `Intrinsics.oscdCentered(f:width:height:)`, `Intrinsics.fromARKit(fx:fy:cx:cy:captureWidth:captureHeight:workingWidth:workingHeight:)`. Members `centre`, `project(_:) -> SIMD2<Float>`, `unproject(_:depth:) -> SIMD3<Float>`. Codable.
  - `Pose(R: simd_float3x3, t: SIMD3<Float>)`. Members `.identity`, `apply`, `inverse`, `centre`, `compose(_:)`, `matrix`, `init(matrix:)`, `fromARKitCameraToWorld(_:)`, `rotationDegrees(to:)`, `r6: [Double]`, `tDouble`, `init(r6:t:)`.
  - `SixD.toMatrix(_ p: ArraySlice<Double>)` and `toMatrix(_ p: [Double]) -> simd_double3x3`; `SixD.fromMatrix(_:) -> [Double]`.
  - `simd_float3x3(converting: simd_double3x3)`, `simd_double3x3(converting: simd_float3x3)`.
  - `simd_float4x4.columnMajor: [Float]`, `simd_float4x4(columnMajor:)`.

- [ ] **Step 1: Write the failing test**

`Tests/OSCDCoreTests/CameraTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class CameraTests: XCTestCase {
    func testProjectAndUnproject() {
        let K = Intrinsics(f: 100, cx: 50, cy: 40, width: 100, height: 80)
        XCTAssertEqual(K.project(SIMD3(1, 2, 4)), SIMD2(75, 90))
        let p = K.unproject(SIMD2(75, 90), depth: 4)
        XCTAssertEqual(p.x, 1, accuracy: 1e-5); XCTAssertEqual(p.y, 2, accuracy: 1e-5); XCTAssertEqual(p.z, 4)
    }

    func testOSCDCentreAndARKitScaling() {
        let c = Intrinsics.oscdCentered(f: 360, width: 480, height: 360)
        XCTAssertEqual(c.cx, 239.5); XCTAssertEqual(c.cy, 179.5)
        let k = Intrinsics.fromARKit(fx: 1440, fy: 1442, cx: 959.5, cy: 719.5,
                                     captureWidth: 1920, captureHeight: 1440, workingWidth: 480, workingHeight: 360)
        XCTAssertEqual(k.f, 360.25, accuracy: 1e-4)
        XCTAssertEqual(k.cx, 239.5, accuracy: 1e-4)
        XCTAssertEqual(k.cy, 179.5, accuracy: 1e-4)
    }

    func testSixDRoundTripAndOrthonormalisation() {
        let R = simd_double3x3(simd_quatd(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 3))))
        let back = SixD.toMatrix(SixD.fromMatrix(R))
        for c in 0..<3 { for r in 0..<3 { XCTAssertEqual(back[c][r], R[c][r], accuracy: 1e-12) } }
        // Non-orthonormal input: columns (2,0,0) and (0,3,0) in O-SCD's row-major (3,2) layout.
        let I = SixD.toMatrix([2, 0, 0, 3, 0, 0])
        XCTAssertEqual(I, matrix_identity_double3x3)
    }

    func testPoseInverseAndCentre() {
        let p = Pose(R: simd_float3x3(simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0))), t: SIMD3(0.1, -0.2, 0.5))
        let q = p.compose(p.inverse)
        XCTAssertLessThan(simd_length(q.t), 1e-6)
        XCTAssertLessThan(p.rotationDegrees(to: p), 1e-3)
        XCTAssertLessThan(simd_length(p.apply(p.centre)), 1e-6)
    }

    func testARKitAxesBecomeOpenCV() {
        // ARKit camera at the origin looking down -z. A point 2 m in front of it must land at z = +2.
        let pose = Pose.fromARKitCameraToWorld(matrix_identity_float4x4)
        let pc = pose.apply(SIMD3(0, 0, -2))
        XCTAssertEqual(pc, SIMD3(0, 0, 2))
        // A point above the camera (ARKit +y) is at negative image y in OpenCV axes.
        XCTAssertLessThan(pose.apply(SIMD3(0, 1, -2)).y, 0)
    }

    func testR6RoundTripThroughPose() {
        let p = Pose(R: simd_float3x3(simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3(1, 1, 0)))), t: SIMD3(1, 2, 3))
        let q = Pose(r6: p.r6, t: p.tDouble)
        XCTAssertLessThan(p.rotationDegrees(to: q), 1e-3)
        XCTAssertEqual(q.t, p.t)
    }

    func testColumnMajorRoundTrip() {
        let m = Pose(R: simd_float3x3(simd_quatf(angle: 0.2, axis: SIMD3(1, 0, 0))), t: SIMD3(4, 5, 6)).matrix
        XCTAssertEqual(simd_float4x4(columnMajor: m.columnMajor), m)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter CameraTests`
Expected: FAIL to compile (`cannot find 'Intrinsics' in scope`).

- [ ] **Step 3: Implement `Camera.swift` and finish `Fixtures.swift`**

`Sources/OSCDCore/Geometry/Camera.swift`:
```swift
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
```

Add the deferred helper to `Fixtures.swift` (inside `enum Fixtures`):
```swift
    /// numpy row-major 4×4 world-to-camera matrix → Pose.
    static func pose(rowMajor4x4 m: [Float]) -> Pose {
        let R = simd_float3x3(rows: [SIMD3(m[0], m[1], m[2]), SIMD3(m[4], m[5], m[6]), SIMD3(m[8], m[9], m[10])])
        return Pose(R: R, t: SIMD3(m[3], m[7], m[11]))
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter CameraTests`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Add camera geometry: intrinsics, poses, O-SCD 6D rotation, ARKit axes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `FloatImage` with torch-exact resizing

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Image/FloatImage.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/FloatImageTests.swift`

**Interfaces:**
- Produces:
  - `FloatImage(width:height:channels:data:)` and `FloatImage(width:height:channels:fill:)`, laid out planar CHW.
  - `init(hwc:width:height:channels:)`, `init(rgb8:width:height:)`, `subscript(c, y, x)`, `plane(_:) -> ArraySlice<Float>`.
  - Resizes: `resizedBilinear(width:height:)` (torch `align_corners=False`), `resizedNearest(width:height:)` (torch legacy nearest), `downsampledArea(factor:)`.

- [ ] **Step 1: Write the failing test**

`Tests/OSCDCoreTests/FloatImageTests.swift`:
```swift
import XCTest
@testable import OSCDCore

final class FloatImageTests: XCTestCase {
    func row(_ v: [Float]) -> FloatImage { FloatImage(width: v.count, height: 1, channels: 1, data: v) }

    func testBilinearMatchesTorchDownAndUp() {
        // torch.nn.functional.interpolate(x, size=..., mode='bilinear', align_corners=False)
        XCTAssertEqual(row([0, 1, 2, 3]).resizedBilinear(width: 2, height: 1).data, [0.5, 2.5])
        XCTAssertEqual(row([0, 4]).resizedBilinear(width: 4, height: 1).data, [0, 1, 3, 4])
        let same = row([3, 1, 4, 1, 5]).resizedBilinear(width: 5, height: 1)
        XCTAssertEqual(same.data, [3, 1, 4, 1, 5])
    }

    func testNearestMatchesTorchLegacyNearest() {
        XCTAssertEqual(row([7, 9]).resizedNearest(width: 4, height: 1).data, [7, 7, 9, 9])
        XCTAssertEqual(row([1, 2, 3]).resizedNearest(width: 4, height: 1).data, [1, 1, 2, 3])
    }

    func testAreaDownsample() {
        let img = FloatImage(width: 4, height: 2, channels: 1, data: [1, 3, 5, 7, 1, 3, 5, 7])
        XCTAssertEqual(img.downsampledArea(factor: 2).data, [2, 6])
    }

    func testHWCAndRGB8Conversion() {
        let img = FloatImage(hwc: [1, 2, 3, 4, 5, 6], width: 2, height: 1, channels: 3)
        XCTAssertEqual(img.data, [1, 4, 2, 5, 3, 6])
        let rgb = FloatImage(rgb8: [255, 0, 51], width: 1, height: 1)
        XCTAssertEqual(rgb.data[0], 1); XCTAssertEqual(rgb.data[1], 0); XCTAssertEqual(rgb.data[2], 0.2, accuracy: 1e-6)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter FloatImageTests`
Expected: FAIL to compile (`cannot find 'FloatImage' in scope`).

- [ ] **Step 3: Implement `FloatImage.swift`**

```swift
import Foundation

/// Planar (CHW) float image; RGB values are in [0, 1].
public struct FloatImage: Sendable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public var data: [Float]   // data[(c * height + y) * width + x]

    public init(width: Int, height: Int, channels: Int, data: [Float]) {
        precondition(data.count == width * height * channels, "data size does not match dimensions")
        self.width = width; self.height = height; self.channels = channels; self.data = data
    }

    public init(width: Int, height: Int, channels: Int, fill: Float = 0) {
        self.init(width: width, height: height, channels: channels,
                  data: [Float](repeating: fill, count: width * height * channels))
    }

    /// From interleaved HWC values (e.g. a numpy (H, W, C) array).
    public init(hwc: [Float], width: Int, height: Int, channels: Int) {
        var d = [Float](repeating: 0, count: hwc.count)
        for y in 0..<height { for x in 0..<width { for c in 0..<channels {
            d[(c * height + y) * width + x] = hwc[(y * width + x) * channels + c]
        } } }
        self.init(width: width, height: height, channels: channels, data: d)
    }

    /// From interleaved 8-bit RGB.
    public init(rgb8: [UInt8], width: Int, height: Int) {
        self.init(hwc: rgb8.map { Float($0) / 255 }, width: width, height: height, channels: 3)
    }

    @inline(__always) public subscript(c: Int, y: Int, x: Int) -> Float {
        get { data[(c * height + y) * width + x] }
        set { data[(c * height + y) * width + x] = newValue }
    }

    public func plane(_ c: Int) -> ArraySlice<Float> { data[(c * width * height)..<((c + 1) * width * height)] }

    /// torch F.interpolate(mode='bilinear', align_corners=False).
    public func resizedBilinear(width ow: Int, height oh: Int) -> FloatImage {
        let xs = Self.linearTaps(inSize: width, outSize: ow)
        let ys = Self.linearTaps(inSize: height, outSize: oh)
        var out = FloatImage(width: ow, height: oh, channels: channels)
        for c in 0..<channels {
            for (oy, ty) in ys.enumerated() {
                for (ox, tx) in xs.enumerated() {
                    let v00 = self[c, ty.i0, tx.i0], v01 = self[c, ty.i0, tx.i1]
                    let v10 = self[c, ty.i1, tx.i0], v11 = self[c, ty.i1, tx.i1]
                    out[c, oy, ox] = (1 - ty.w) * ((1 - tx.w) * v00 + tx.w * v01)
                                   + ty.w * ((1 - tx.w) * v10 + tx.w * v11)
                }
            }
        }
        return out
    }

    /// Source taps for one axis: src = max(0, (dst + 0.5) * in / out - 0.5).
    static func linearTaps(inSize: Int, outSize: Int) -> [(i0: Int, i1: Int, w: Float)] {
        let scale = Float(inSize) / Float(outSize)
        return (0..<outSize).map { o in
            let src = max(0, (Float(o) + 0.5) * scale - 0.5)
            let i0 = min(Int(src), inSize - 1)
            return (i0, min(i0 + 1, inSize - 1), src - Float(i0))
        }
    }

    /// torch F.interpolate(mode='nearest'): src = min(floor(dst * in / out), in - 1).
    public func resizedNearest(width ow: Int, height oh: Int) -> FloatImage {
        let sx = Float(width) / Float(ow), sy = Float(height) / Float(oh)
        let xi = (0..<ow).map { min(Int(Float($0) * sx), width - 1) }
        let yi = (0..<oh).map { min(Int(Float($0) * sy), height - 1) }
        var out = FloatImage(width: ow, height: oh, channels: channels)
        for c in 0..<channels { for y in 0..<oh { for x in 0..<ow { out[c, y, x] = self[c, yi[y], xi[x]] } } }
        return out
    }

    /// Mean over factor×factor blocks (OpenCV INTER_AREA for an integer factor).
    public func downsampledArea(factor k: Int) -> FloatImage {
        let ow = width / k, oh = height / k
        var out = FloatImage(width: ow, height: oh, channels: channels)
        let inv = 1 / Float(k * k)
        for c in 0..<channels { for y in 0..<oh { for x in 0..<ow {
            var s: Float = 0
            for dy in 0..<k { for dx in 0..<k { s += self[c, y * k + dy, x * k + dx] } }
            out[c, y, x] = s * inv
        } } }
        return out
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter FloatImageTests`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Add FloatImage with torch-exact bilinear and nearest resizing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Golden fixtures from the real O-SCD (Sol)

This task produces the reference tensors that every parity test in Tasks 6–13 reads. The script calls O-SCD's own functions on one PASLCD instance and records the random index choices, so the Swift tests can replay them exactly.

**Files:**
- Create: `oscd_mobile/tools/dump_oscd_fixtures.py`
- Create: `oscd_mobile/tools/dump_fixtures.sbatch`
- Create: `oscd_mobile/tools/fetch_fixtures.sh`

**Interfaces:**
- Produces fixture folders under `oscd_mobile/fixtures/`. Array shapes are in parentheses; `H, W` = working size, `h = H32/8`, `w = W32/8`, `N` = hypotheses, `n` = points, `C` = cameras.

| File or folder | Contents |
|---|---|
| `meta.json` | width, height, f, cx, cy, max_error, n_refs, xfeat_commit, oscd_commit, versions |
| `xfeat/` | `image (H,W,3)`, `net_feats (64,h,w)`, `net_heatmap (H32,W32)`, `net_reliability (h,w)`, `kpts (6144,2)`, `feats (6144,64)` |
| `match/` | `kpts_a`, `feats_a`, `kpts_b`, `feats_b`, `match12 (6144)`, `mask (6144)` |
| `fransac/` | `kpts1 (n,2)`, `kpts2 (n,2)`, `idxs (2000,8)`, `models (2000,3,3)`, `inliers (2000,n)`, `best_id ()`, `max_error ()` |
| `triangulate/` | `uv (6144,2)`, `uvs_others (C,6144,2)`, `Rt (4,4)`, `Rts_others (C,4,4)`, `f ()`, `centre (2)`, `max_error ()`, `min_dis ()`, `pts3d (6144,3)`, `depth (6144)`, `best_dis (6144)`, `valid (6144)` |
| `anchors/` | `info.json`; per keyframe `kf_0000_kpts`, `kf_0000_feats`, `kf_0000_has_pt3d`, `kf_0000_pts3d`, `kf_0000_Rt`, … |

`pose/frame_000/` … `frame_002/` contain:

| Group | Arrays |
|---|---|
| Frame features | `kpts`, `feats` |
| Retrieval | `retrieval_counts (n_refs)`, `retrieved (4)` |
| Correspondences | `corr_xyz (n,3)`, `corr_uv (n,2)`, `corr_conf (n)` |
| P4P-RANSAC | `p4p_idxs (2000,4)`, `p4p_init_r6 (3,2)`, `p4p_init_t (3)`, `p4p_models (2000,3,4)`, `p4p_inliers (2000,n)`, `p4p_best ()` |
| Mini-BA | `ba_in_xyz (2000,3)`, `ba_in_uv (2000,2)`, `ba_in_r6 (3,2)`, `ba_in_t (3)`, `ba_out_r6 (3,2)`, `ba_out_t (3)`, `ba_r (4000)`, `ba_r_init (4000)`, `ba_mask (4000)`, `accepted ()` |
| O-SCD end to end | `oscd_Rt (4,4)`, `oscd_accepted ()` |

`cues/frame_000/` … contain `live (3,H,W)`, `render (3,H,W)`, `ssim_map (H,W)`, `l1 (H,W)`, `delta_raw (H,W)`, `vmin ()`, `vmax ()`, `emb (2,256,64,64)` (order: render, live), `C (H,W)`, and `sam_input (2,3,1024,1024)` (float16, Python only).

- [ ] **Step 1: Write `dump_oscd_fixtures.py`**

```python
#!/usr/bin/env python
"""Dump O-SCD intermediate tensors as golden fixtures for the Swift port (oscd_mobile).

Run on a GPU node inside the scd_robustness `oscd` env (see dump_fixtures.sbatch). The script
calls O-SCD's own functions (commit 3abfeaa, patched by scd_robustness) on one PASLCD instance
and saves each component's inputs and outputs as .npy files. Random index choices (RANSAC
samples, subsets) are generated here and saved, so the Swift tests can replay them.
"""
import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np


def save(d, name, x):
    import torch
    os.makedirs(d, exist_ok=True)
    if isinstance(x, torch.Tensor):
        x = x.detach()
        x = x.float().cpu().numpy() if x.dtype.is_floating_point else x.cpu().numpy()
    x = np.asarray(x)
    if x.dtype in (np.float64, np.float16):
        x = x.astype(np.float32)
    if x.dtype == np.int32:
        x = x.astype(np.int64)
    np.save(os.path.join(d, name + ".npy"), x)


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd-root", required=True, help="O-SCD checkout (scd_robustness code/O-SCD)")
    ap.add_argument("--instance", required=True, help=".../PASLCD/Instance_1/<Scene>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    instance = os.path.abspath(a.instance)

    # O-SCD opens poses/ransac.cu with a relative path, so run from its root.
    os.chdir(a.oscd_root)
    sys.path.insert(0, a.oscd_root)
    sys.argv = ["oscd.py", "-s", instance, "-m", os.path.join(out, "_oscd_model"), "--resolution", "4"]

    import torch
    import torch.nn.functional as F
    from arguments.config_args import parse_args
    from dataloaders.image_dataset import ImageDataset
    from gaussian_renderer import render
    from oscd import generate_candidate_map
    from poses.feature_detector import Detector
    from poses.matcher import Matcher, match
    from poses.pose_initializer import PoseInitializer
    from poses.triangulator import Triangulator
    from scene import GaussianModel
    from scene.cameras import Camera
    from scene.dense_extractor import DenseExtractor
    from scene.keyframe import Keyframe
    from transformers import Sam2Model
    from utils.loss_utils import ssim

    args, lp, op, pp = parse_args()
    dataset_args, pipe = lp.extract(args), pp.extract(args)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); np.random.seed(0); random.seed(0)

    ref = ImageDataset(args, instance="ref")
    inf = ImageDataset(args, instance="inf")
    height, width = inf.get_image_size()
    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(args.num_kpts, args.num_keyframes_for_triangulation, max_error)
    pose_init = PoseInitializer(width, height, triangulator, matcher, max_error, args)
    dense = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)

    refs = []
    for i in range(len(ref)):
        image, info = ref.getnext()
        kp = detector(image)
        refs.append(Keyframe(image, info, kp, info["Rt"], i, info["focal"], dense, triangulator, args))
        f_ref, fovx, fovy = info["focal"], info["FovX"], info["FovY"]
    pose_init.init_focal(f_ref)
    f = float(torch.as_tensor(f_ref).reshape(-1)[0])
    centre = pose_init.centre

    # ---- xfeat: raw network outputs and the Detector's final keypoints for reference image 0
    xf = torch.hub.load(os.environ["SCD_XFEAT_REPO"], "XFeat", source="local", pretrained=True,
                        top_k=args.num_kpts).cuda().eval().half()
    img0 = refs[0].image_pyr[0]
    H32, W32 = (height // 32) * 32, (width // 32) * 32
    with torch.no_grad():
        x = F.interpolate(img0[None].half(), (H32, W32), mode="bilinear", align_corners=False)
        M1, K1, H1 = xf.net(x)
        M1 = F.normalize(M1, dim=1)
        K1h = xf.get_kpts_heatmap(K1)
    d = os.path.join(out, "xfeat")
    save(d, "image", img0.permute(1, 2, 0))
    save(d, "net_feats", M1[0]); save(d, "net_heatmap", K1h[0, 0]); save(d, "net_reliability", H1[0, 0])
    save(d, "kpts", refs[0].desc_kpts.kpts); save(d, "feats", refs[0].desc_kpts.feats)

    # ---- mutual-NN matching between reference images 0 and 1
    ka, kb = refs[0].desc_kpts, refs[1].desc_kpts
    idx0, m12, mask = match(ka.feats.cuda(), kb.feats.cuda())
    d = os.path.join(out, "match")
    save(d, "kpts_a", ka.kpts); save(d, "feats_a", ka.feats); save(d, "kpts_b", kb.kpts); save(d, "feats_b", kb.feats)
    save(d, "match12", m12); save(d, "mask", mask)

    # ---- 8-point F-RANSAC with recorded samples
    k1, k2 = ka.kpts[idx0[mask]].contiguous(), kb.kpts[m12[mask]].contiguous()
    est = matcher.fundmat_estimator
    idxs = torch.topk(torch.rand(est.N, k1.shape[0], device="cuda"), est.m, dim=1)[1]
    est.estimate(k1, k2, idxs, None, None, None, None)
    inl = est.get_inlier_mask(k1, k2, None, None)
    d = os.path.join(out, "fransac")
    save(d, "kpts1", k1); save(d, "kpts2", k2); save(d, "idxs", idxs); save(d, "models", est.models)
    save(d, "inliers", inl); save(d, "best_id", torch.argmax(inl.sum(dim=1))); save(d, "max_error", np.float32(max_error))

    # ---- reference anchors exactly as oscd.py builds them
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            matcher(refs[i].desc_kpts, refs[j].desc_kpts, remove_outliers=True,
                    update_kpts_flag="inliers", kID=i, kID_other=j)
    kf = refs[0]
    uv, uvs_others, chosen = triangulator.prepare_matches(kf.desc_kpts)
    Rts = torch.stack([refs[c].get_Rt() for c in chosen])
    n_cams = triangulator.n_cams
    Rts_pad = torch.cat([Rts, torch.eye(4, device="cuda")[None].repeat(n_cams - len(Rts), 1, 1)]) if len(Rts) < n_cams else Rts
    pts, depth, best, valid = triangulator(uv, uvs_others, kf.get_Rt(), Rts_pad, kf.f, kf.centre)
    d = os.path.join(out, "triangulate")
    save(d, "uv", uv); save(d, "uvs_others", uvs_others[:len(chosen)]); save(d, "Rt", kf.get_Rt())
    save(d, "Rts_others", Rts); save(d, "f", np.float32(f)); save(d, "centre", kf.centre)
    save(d, "max_error", np.float32(max_error)); save(d, "min_dis", np.float32(30 * max_error))
    save(d, "pts3d", pts); save(d, "depth", depth); save(d, "best_dis", best); save(d, "valid", valid)
    for k in refs:
        k.update_3dpts(refs)
    d = os.path.join(out, "anchors")
    for i, k in enumerate(refs):
        dk = k.desc_kpts
        save(d, f"kf_{i:04d}_kpts", dk.kpts); save(d, f"kf_{i:04d}_feats", dk.feats)
        save(d, f"kf_{i:04d}_has_pt3d", dk.has_pt3d); save(d, f"kf_{i:04d}_pts3d", dk.pts3d)
        save(d, f"kf_{i:04d}_Rt", k.get_Rt())
    with open(os.path.join(d, "info.json"), "w") as fh:
        json.dump({"count": len(refs), "f": f, "cx": float(centre[0]), "cy": float(centre[1]),
                   "width": width, "height": height, "max_error": max_error}, fh, indent=2)

    # ---- per-frame pose pipeline with recorded randomness, then O-SCD's own end-to-end pose
    gaussians = GaussianModel(dataset_args.sh_degree, dataset_args.sh_degree)
    gaussians.load_ply(os.path.join(instance, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().to("cuda")
    bg = torch.zeros(3, device="cuda")
    for k in range(a.frames):
        image, info = inf.getnext()
        kp = detector(image)
        d = os.path.join(out, "pose", f"frame_{k:03d}")
        save(d, "kpts", kp.kpts); save(d, "feats", kp.feats)
        counts = torch.tensor([float(matcher.evaluate_match(r.desc_kpts, kp)) for r in refs])
        top = torch.topk(counts, args.num_reference_keyframes).indices
        save(d, "retrieval_counts", counts); save(d, "retrieved", top)
        kfs = [refs[i] for i in top.tolist()]
        xyz, uvs, confs = [], [], []
        for kfr in kfs:
            m = matcher(kp, kfr.desc_kpts, remove_outliers=True)
            msk = kfr.desc_kpts.has_pt3d[m.idx_other]
            xyz.append(kfr.desc_kpts.pts3d[m.idx_other[msk]]); uvs.append(m.kpts[msk])
            confs.append(kfr.desc_kpts.pts_conf[m.idx_other[msk]])
        xyz, uvs, confs = torch.cat(xyz), torch.cat(uvs), torch.cat(confs)
        if len(xyz) > pose_init.num_pts_pnpransac:
            sel = torch.multinomial(confs, pose_init.num_pts_miniba_incr, replacement=False)
            xyz, uvs, confs = xyz[sel], uvs[sel], confs[sel]
        save(d, "corr_xyz", xyz); save(d, "corr_uv", uvs); save(d, "corr_conf", confs)
        est = pose_init.PnPRANSAC
        p_idxs = torch.topk(torch.rand(est.N, uvs.shape[0], device="cuda"), est.m, dim=1)[1]
        r6_init, t_init = kfs[0].rW2C, kfs[0].tW2C
        est.estimate(uvs, xyz, p_idxs, pose_init.f, centre, r6_init, t_init)
        p_inl = est.get_inlier_mask(uvs, xyz, pose_init.f, centre)
        best_h = torch.argmax((p_inl * confs[None]).sum(dim=1))
        save(d, "p4p_idxs", p_idxs); save(d, "p4p_init_r6", r6_init); save(d, "p4p_init_t", t_init)
        save(d, "p4p_models", est.models); save(d, "p4p_inliers", p_inl); save(d, "p4p_best", best_h)
        Rt = est.models[best_h].clone()
        sel_in = p_inl[best_h] > 0
        xyz_i, uv_i = xyz[sel_in], uvs[sel_in]
        N = pose_init.num_pts_miniba_incr
        if len(xyz_i) >= N:
            sel = torch.topk(torch.rand_like(xyz_i[..., 0]), N, dim=0, largest=False)[1]
            xyz_ba, uv_ba = xyz_i[sel], uv_i[sel]
        else:
            xyz_ba = torch.cat([xyz_i, torch.zeros(N - len(xyz_i), 3, device="cuda")])
            uv_ba = torch.cat([uv_i, -torch.ones(N - len(uv_i), 2, device="cuda")])
        R6, t = Rt[:3, :2][None], Rt[:3, 3][None]
        R6o, to, _, _, r, r_init, ba_mask = pose_init.miniBA(R6, t, pose_init.f, xyz_ba, centre, uv_ba.view(-1))
        save(d, "ba_in_xyz", xyz_ba); save(d, "ba_in_uv", uv_ba); save(d, "ba_in_r6", R6[0]); save(d, "ba_in_t", t[0])
        save(d, "ba_out_r6", R6o.reshape(-1, 3, 2)[0]); save(d, "ba_out_t", to.reshape(-1, 3)[0])
        save(d, "ba_r", r); save(d, "ba_r_init", r_init); save(d, "ba_mask", ba_mask)
        save(d, "accepted", np.bool_(bool(ba_mask.sum() > args.min_num_inliers)))
        res = pose_init.init_inference_pose(kfs, kp, k, False, image)
        save(d, "oscd_Rt", res[0] if res is not None else torch.zeros(4, 4))
        save(d, "oscd_accepted", np.bool_(res is not None))

        # ---- change cues at O-SCD's pose, against the real reference render
        if res is None:
            continue
        Rw = res[0]
        view = Camera(colmap_id=f"{k:04d}", R=np.transpose(Rw[:3, :3].cpu().numpy()), T=Rw[:3, 3].cpu().numpy(),
                      FoVx=fovx, FoVy=fovy, image=image[:3], gt_alpha_mask=None,
                      image_name=info["name"].split(".")[0], uid=f"{k:04d}")
        with torch.no_grad():
            ren = render(view, gaussians, pipe, bg)["render"]
            live = view.original_image[:3]
            C = generate_candidate_map(live, ren, sam, 14, height, width)
            ssim_map = ssim(ren, live, window_size=11, map=True).mean(dim=0)
            l1 = torch.abs(ren - live).mean(dim=0)
            delta = 0.8 * l1 + 0.2 * (1 - ssim_map)
            r1024 = F.interpolate(ren[None], size=(1024, 1024), mode="nearest")[0]
            l1024 = F.interpolate(live[None], size=(1024, 1024), mode="nearest")[0]
            stack = torch.stack([r1024, l1024]).half()
            emb = sam.get_image_embeddings(stack)[-1]
        d = os.path.join(out, "cues", f"frame_{k:03d}")
        save(d, "live", live); save(d, "render", ren); save(d, "ssim_map", ssim_map); save(d, "l1", l1)
        save(d, "delta_raw", delta); save(d, "vmin", delta.min()); save(d, "vmax", delta.max())
        save(d, "emb", emb); save(d, "C", C[0])
        np.save(os.path.join(d, "sam_input.npy"), stack.cpu().numpy().astype(np.float16))

    import transformers
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"width": width, "height": height, "f": f, "cx": float(centre[0]), "cy": float(centre[1]),
                   "max_error": max_error, "n_refs": len(refs), "instance": instance,
                   "xfeat_commit": git_head(os.environ["SCD_XFEAT_REPO"]), "oscd_commit": git_head(a.oscd_root),
                   "torch": torch.__version__, "transformers": transformers.__version__}, fh, indent=2)
    print("fixtures written to", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `dump_fixtures.sbatch` and `fetch_fixtures.sh`**

`oscd_mobile/tools/dump_fixtures.sbatch`:
```bash
#!/bin/bash
# Golden fixtures for oscd_mobile. Submit from oscd_mobile/tools on Sol:
#   SCD_REPO=/path/to/scd_robustness sbatch -p <partition> -q <qos> dump_fixtures.sbatch
#SBATCH -J oscd-fixtures
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm-oscd-fixtures-%j.out
set -eo pipefail
: "${SCD_REPO:?set SCD_REPO to the scd_robustness checkout on Sol}"
source "$SCD_REPO/setup/sol.env"
scd_offline
scd_activate oscd
SCENE="${SCENE:-Garden}"
OUT="${OUT:-$SCD_ROOT/fixtures/oscd_mobile}"
python "$SLURM_SUBMIT_DIR/dump_oscd_fixtures.py" \
  --oscd-root "$SCD_ROOT/code/O-SCD" \
  --instance "$SCD_ROOT/data/raw/online/PASLCD/Instance_1/$SCENE" \
  --out "$OUT" --frames 3
du -sh "$OUT"
```

`oscd_mobile/tools/fetch_fixtures.sh`:
```bash
#!/bin/bash
# Copy fixtures from Sol into oscd_mobile/fixtures/.
# Usage: tools/fetch_fixtures.sh <user>@<sol-login-host> [remote_dir]
set -euo pipefail
REMOTE="${1:?usage: fetch_fixtures.sh user@host [remote_dir]}"
DIR="${2:-/scratch/\$USER/scd/fixtures/oscd_mobile}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/fixtures"
rsync -av --delete --exclude "_oscd_model" "$REMOTE:$DIR/" "$HERE/fixtures/"
```

```bash
chmod +x oscd_mobile/tools/fetch_fixtures.sh
python3 -m py_compile oscd_mobile/tools/dump_oscd_fixtures.py
```
Expected: no output (the script compiles).

- [ ] **Step 3: Human step. Run on Sol and fetch**

```bash
# on Sol, from the repo checkout's oscd_mobile/tools:
SCD_REPO=$HOME/watermark_service/scd_robustness sbatch -p <partition> -q <qos> dump_fixtures.sbatch
# on the Mac, after the job finishes:
oscd_mobile/tools/fetch_fixtures.sh <asurite>@login.sol.rc.asu.edu
ls oscd_mobile/fixtures
```
Expected: `anchors cues fransac match meta.json pose triangulate xfeat`, plus three frame folders under `pose/` and `cues/`. A frame with `oscd_accepted = False` has no `cues/` folder. If all three are rejected, rerun with `SCENE=Lounge`.

- [ ] **Step 4: Commit**

```bash
git add oscd_mobile/tools/dump_oscd_fixtures.py oscd_mobile/tools/dump_fixtures.sbatch oscd_mobile/tools/fetch_fixtures.sh
git commit -m "Add the Sol job that dumps O-SCD golden fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Convert XFeat to Core ML

**Files:**
- Create: `oscd_mobile/tools/requirements-mac.txt`
- Create: `oscd_mobile/tools/convert_xfeat_coreml.py`
- Modify: `oscd_mobile/README.md` (append a "Models" section)

**Interfaces:**
- Produces `oscd_mobile/Models/XFeat_480x352.mlpackage`, with:
  - input `image`: Float32 [1,3,352,480], RGB in [0,1];
  - output `feats`: Float32 [1,64,44,60], L2-normalised over channels;
  - output `heatmap`: Float32 [1,1,352,480];
  - output `reliability`: Float32 [1,1,44,60].

- [ ] **Step 1: Python environment and XFeat checkout**

`oscd_mobile/tools/requirements-mac.txt`:
```text
numpy==1.26.4
torch==2.5.0
coremltools==8.3.0
transformers==4.56.1
pillow==10.4.0
opencv-python-headless==4.10.0.84
```

```bash
cd oscd_mobile
python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements-mac.txt
git clone https://github.com/verlab/accelerated_features third_party/accelerated_features
# Use the same XFeat commit as the Sol fixtures (falls back to upstream HEAD without fixtures).
XCOMMIT=$(python3 -c "import json;print(json.load(open('fixtures/meta.json'))['xfeat_commit'])" 2>/dev/null || echo HEAD)
git -C third_party/accelerated_features checkout "$XCOMMIT"
```
Expected: pip finishes without errors, and the checkout reports `HEAD is now at <commit>`.

- [ ] **Step 2: Write `convert_xfeat_coreml.py`**

```python
#!/usr/bin/env python
"""Convert XFeat's network (+ O-SCD's descriptor normalisation and heatmap) to Core ML.

Output Models/XFeat_<W>x<H>.mlpackage. The Swift side (XFeatPostprocess) does NMS, scoring,
top-k and descriptor sampling exactly as O-SCD's Detector does.
"""
import argparse
import os

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


class XFeatCore(torch.nn.Module):
    def __init__(self, xf):
        super().__init__()
        self.net = xf.net

    def forward(self, x):
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        scores = F.softmax(K1, 1)[:, :64]                      # XFeat.get_kpts_heatmap, softmax_temp = 1
        B, _, H, W = scores.shape
        heat = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heat = heat.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return M1, heat, H1


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--xfeat-repo", default=os.path.join(here, "..", "third_party", "accelerated_features"))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--out", default=os.path.join(here, "..", "Models"))
    ap.add_argument("--fixtures", default=os.path.join(here, "..", "fixtures"))
    a = ap.parse_args()

    xf = torch.hub.load(a.xfeat_repo, "XFeat", source="local", pretrained=True, top_k=6144).eval()
    core = XFeatCore(xf).eval()
    example = torch.rand(1, 3, a.height, a.width)
    with torch.no_grad():
        ref_heat = xf.get_kpts_heatmap(xf.net(example)[1])
        assert torch.allclose(core(example)[1], ref_heat, atol=1e-6), "heatmap reshape differs from XFeat"
        traced = torch.jit.trace(core, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, a.height, a.width), dtype=np.float32)],
        outputs=[ct.TensorType(name="feats", dtype=np.float32),
                 ct.TensorType(name="heatmap", dtype=np.float32),
                 ct.TensorType(name="reliability", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"XFeat_{a.width}x{a.height}.mlpackage")
    mlmodel.save(path)
    print("saved", path)

    # Parity against the Sol fixture (fp16 CUDA network outputs), when present.
    img_path = os.path.join(a.fixtures, "xfeat", "image.npy")
    if not os.path.exists(img_path):
        print("no fixtures; skipped parity check")
        return
    img = torch.from_numpy(np.load(img_path)).permute(2, 0, 1)[None]
    x = F.interpolate(img, (a.height, a.width), mode="bilinear", align_corners=False)
    pred = mlmodel.predict({"image": x.numpy().astype(np.float32)})
    for name, fx in [("feats", "net_feats"), ("heatmap", "net_heatmap"), ("reliability", "net_reliability")]:
        got = pred[name].reshape(-1)
        want = np.load(os.path.join(a.fixtures, "xfeat", fx + ".npy")).reshape(-1)
        err = np.abs(got - want).max()
        print(f"{name}: max |coreml - oscd| = {err:.4g}")
        assert err < {"feats": 3e-2, "heatmap": 5e-3, "reliability": 2e-2}[name], name
    print("parity OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the conversion**

Run: `oscd_mobile/tools/.venv/bin/python oscd_mobile/tools/convert_xfeat_coreml.py`
Expected: `saved .../Models/XFeat_480x352.mlpackage`, then three `max |coreml - oscd|` lines and `parity OK` when fixtures exist; otherwise `no fixtures; skipped parity check`.

- [ ] **Step 4: Document and commit**

Append to `oscd_mobile/README.md`:
```markdown
## Models (not in git)
    python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements-mac.txt
    git clone https://github.com/verlab/accelerated_features third_party/accelerated_features
    tools/.venv/bin/python tools/convert_xfeat_coreml.py     # → Models/XFeat_480x352.mlpackage
```

```bash
git add oscd_mobile/tools/requirements-mac.txt oscd_mobile/tools/convert_xfeat_coreml.py oscd_mobile/README.md
git commit -m "Convert XFeat to Core ML with a parity check against the O-SCD fixture

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 6: XFeat post-processing and the Core ML detector

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Features/DescribedKeypoints.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Features/XFeatPostprocess.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Features/MLMultiArray+Dense.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Features/XFeatModel.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/XFeatTests.swift`

**Interfaces:**
- Consumes: `FloatImage.resizedBilinear` (Task 3), `NPYArray` (Task 1).
- Produces:
  - `DescribedKeypoints(kpts: [SIMD2<Float>], feats: [Float])` with `dim = 64`, `count`, `isValid(_:)`, and `static func load(kpts: NPYArray, feats: NPYArray)`.
  - `XFeatPostprocess.run(feats:heatmap:reliability:h:w:topK:imageWidth:imageHeight:) -> DescribedKeypoints`. It returns only keypoints with a positive score, best first.
  - `XFeatModel(compiledModelURL:inputWidth:inputHeight:computeUnits:)`, `XFeatModel.compile(_ packageURL: URL) throws -> URL`, `detect(_ image: FloatImage, topK: Int = 6144) throws -> DescribedKeypoints`.
  - `ModelError`, `MLMultiArray.denseFloats() -> [Float]`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/XFeatTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class XFeatTests: XCTestCase {
    /// A single strong peak must come back as one keypoint at the right place with a unit descriptor.
    func testSyntheticPeak() {
        let h = 4, w = 4, H = 32, W = 32
        var heat = [Float](repeating: 0.01, count: H * W)
        heat[10 * W + 13] = 0.9
        let rel = [Float](repeating: 0.5, count: h * w)
        var feats = [Float](repeating: 0, count: 64 * h * w)
        for i in 0..<(h * w) { feats[3 * h * w + i] = 1 }       // channel 3 everywhere
        let kp = XFeatPostprocess.run(feats: feats, heatmap: heat, reliability: rel, h: h, w: w,
                                      topK: 10, imageWidth: 32, imageHeight: 32)
        XCTAssertEqual(kp.count, 1)
        XCTAssertEqual(kp.kpts[0], SIMD2(13, 10))
        XCTAssertEqual(kp.feats[3], 1, accuracy: 1e-5)
    }

    /// Exact O-SCD Detector post-processing on the Sol network outputs.
    func testParityPostprocessMatchesOSCDDetector() throws {
        let d = try Fixtures.dir("xfeat")
        let nf = try Fixtures.npy(d, "net_feats"), hm = try Fixtures.npy(d, "net_heatmap"), rl = try Fixtures.npy(d, "net_reliability")
        let img = try Fixtures.npy(d, "image")
        let got = XFeatPostprocess.run(feats: nf.floats(), heatmap: hm.floats(), reliability: rl.floats(),
                                       h: nf.shape[1], w: nf.shape[2], topK: 6144,
                                       imageWidth: img.shape[1], imageHeight: img.shape[0])
        let want = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
        let (posAgree, cosMean) = Self.compare(want: want, got: got, top: 1000)
        XCTAssertGreaterThanOrEqual(posAgree, 0.99)
        XCTAssertGreaterThanOrEqual(cosMean, 0.999)
    }

    /// Core ML network + post-processing end to end, against the O-SCD Detector output.
    func testParityCoreMLDetector() throws {
        let d = try Fixtures.dir("xfeat")
        let pkg = try Fixtures.modelPackage("XFeat_480x352.mlpackage")
        let model = try XFeatModel(compiledModelURL: try XFeatModel.compile(pkg))
        let img = try Fixtures.npy(d, "image")
        let image = FloatImage(hwc: img.floats(), width: img.shape[1], height: img.shape[0], channels: 3)
        let got = try model.detect(image)
        let want = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
        let (posAgree, cosMean) = Self.compare(want: want, got: got, top: 1000)
        XCTAssertGreaterThanOrEqual(posAgree, 0.95)
        XCTAssertGreaterThanOrEqual(cosMean, 0.99)
    }

    /// Fraction of O-SCD's first `top` valid keypoints we also found within 1 px, and the mean
    /// descriptor cosine over those pairs.
    static func compare(want: DescribedKeypoints, got: DescribedKeypoints, top: Int) -> (Double, Double) {
        var index: [SIMD2<Int32>: Int] = [:]
        for i in 0..<got.count { index[SIMD2(Int32(got.kpts[i].x.rounded()), Int32(got.kpts[i].y.rounded()))] = i }
        var hit = 0, total = 0, cosSum = 0.0
        for i in 0..<want.count where want.isValid(i) {
            if total == top { break }
            total += 1
            let k = SIMD2(Int32(want.kpts[i].x.rounded()), Int32(want.kpts[i].y.rounded()))
            var found: Int?
            outer: for dy in -1...1 { for dx in -1...1 {
                if let j = index[k &+ SIMD2(Int32(dx), Int32(dy))], simd_distance(got.kpts[j], want.kpts[i]) <= 1 { found = j; break outer }
            } }
            guard let j = found else { continue }
            hit += 1
            var c: Float = 0
            for t in 0..<64 { c += want.feats[i * 64 + t] * got.feats[j * 64 + t] }
            cosSum += Double(c)
        }
        return (Double(hit) / Double(max(total, 1)), hit > 0 ? cosSum / Double(hit) : 0)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter XFeatTests`
Expected: FAIL to compile (`cannot find 'XFeatPostprocess' in scope`).

- [ ] **Step 3: Implement the four source files**

`Sources/OSCDCore/Features/DescribedKeypoints.swift`:
```swift
import Foundation

/// XFeat keypoints with 64-d L2-normalised descriptors (O-SCD `DescribedKeypoints`).
/// An all-zero descriptor row marks an invalid keypoint (O-SCD zeroes features with score ≤ 0).
public struct DescribedKeypoints: Sendable {
    public static let dim = 64
    public var kpts: [SIMD2<Float>]
    public var feats: [Float]          // count × 64, row-major

    public init(kpts: [SIMD2<Float>], feats: [Float]) {
        precondition(feats.count == kpts.count * Self.dim, "feats must hold 64 values per keypoint")
        self.kpts = kpts
        self.feats = feats
    }

    public var count: Int { kpts.count }

    public func isValid(_ i: Int) -> Bool {
        for c in 0..<Self.dim where feats[i * Self.dim + c] != 0 { return true }
        return false
    }

    public static func load(kpts: NPYArray, feats: NPYArray) -> DescribedKeypoints {
        let k = kpts.floats()
        return DescribedKeypoints(kpts: stride(from: 0, to: k.count, by: 2).map { SIMD2(k[$0], k[$0 + 1]) },
                                  feats: feats.floats())
    }
}
```

`Sources/OSCDCore/Features/XFeatPostprocess.swift`:
```swift
import Foundation

/// O-SCD `Detector.detectAndCompute` after the network (poses/feature_detector.py).
public enum XFeatPostprocess {
    public static let detectionThreshold: Float = 0.05
    public static let nmsKernel = 5

    /// - feats: 64×h×w descriptors, already L2-normalised over channels
    /// - heatmap: H×W keypoint heatmap (H = 8h, W = 8w), XFeat `get_kpts_heatmap`
    /// - reliability: h×w
    /// Returns keypoints with a positive score, best first, in the original image's pixel grid.
    public static func run(feats: [Float], heatmap: [Float], reliability: [Float],
                           h: Int, w: Int, topK: Int, imageWidth: Int, imageHeight: Int) -> DescribedKeypoints {
        let H = h * 8, W = w * 8
        let nvalid = Int(1.5 * Double(topK))

        // 1. NMS: thresholded local maxima of a 5×5 window (MaxPool2d pads with -inf; ties count as maxima).
        var cand: [(v: Float, i: Int)] = []
        let r = nmsKernel / 2
        for y in 0..<H {
            for x in 0..<W {
                let v = heatmap[y * W + x]
                guard v > detectionThreshold else { continue }
                var isMax = true
                outer: for yy in max(0, y - r)...min(H - 1, y + r) {
                    for xx in max(0, x - r)...min(W - 1, x + r) where heatmap[yy * W + xx] > v {
                        isMax = false
                        break outer
                    }
                }
                if isMax { cand.append((v, y * W + x)) }
            }
        }
        cand.sort { $0.v != $1.v ? $0.v > $1.v : $0.i < $1.i }
        if cand.count > nvalid { cand.removeLast(cand.count - nvalid) }

        // 2. score = nearest(heatmap) × bilinear(reliability); a keypoint at (0, 0) scores -1.
        var scored: [(s: Float, x: Int, y: Int)] = cand.map { c in
            let x = c.i % W, y = c.i / W
            if x == 0 && y == 0 { return (-1, x, y) }
            let gx = gridCoord(x, W), gy = gridCoord(y, H)
            return (sampleNearest(heatmap, W, H, gx, gy) * sampleBilinear(reliability, w, h, gx, gy), x, y)
        }
        scored.sort { $0.s != $1.s ? $0.s > $1.s : ($0.y * W + $0.x) < ($1.y * W + $1.x) }
        if scored.count > topK { scored.removeLast(scored.count - topK) }
        scored.removeAll { $0.s <= 0 }

        // 3. Bicubic descriptors, L2-normalised. 4. Rescale positions to the original image.
        let rw = Float(imageWidth) / Float(W), rh = Float(imageHeight) / Float(H)
        var kpts: [SIMD2<Float>] = []
        kpts.reserveCapacity(scored.count)
        var out = [Float](repeating: 0, count: scored.count * 64)
        for (n, k) in scored.enumerated() {
            sampleBicubic(feats, channels: 64, w, h, gridCoord(k.x, W), gridCoord(k.y, H), into: &out, offset: n * 64)
            var norm: Float = 0
            for c in 0..<64 { norm += out[n * 64 + c] * out[n * 64 + c] }
            norm = max(norm.squareRoot(), 1e-12)
            for c in 0..<64 { out[n * 64 + c] /= norm }
            kpts.append(SIMD2(Float(k.x) * rw, Float(k.y) * rh))
        }
        return DescribedKeypoints(kpts: kpts, feats: out)
    }

    /// InterpolateSparse2d.normgrid, then the cast to the feature dtype (fp16 in O-SCD).
    @inline(__always) static func gridCoord(_ p: Int, _ size: Int) -> Float {
        let g = 2 * (Float(p) / Float(size - 1)) - 1
        #if arch(arm64)
        return Float(Float16(g))
        #else
        return g
        #endif
    }

    /// grid_sample un-normalisation with align_corners = False.
    @inline(__always) static func unnormalize(_ g: Float, _ size: Int) -> Float { ((g + 1) * Float(size) - 1) / 2 }

    static func sampleNearest(_ m: [Float], _ w: Int, _ h: Int, _ gx: Float, _ gy: Float) -> Float {
        let ix = Int(unnormalize(gx, w).rounded(.toNearestOrEven))
        let iy = Int(unnormalize(gy, h).rounded(.toNearestOrEven))
        guard ix >= 0, ix < w, iy >= 0, iy < h else { return 0 }
        return m[iy * w + ix]
    }

    static func sampleBilinear(_ m: [Float], _ w: Int, _ h: Int, _ gx: Float, _ gy: Float) -> Float {
        let px = unnormalize(gx, w), py = unnormalize(gy, h)
        let x0 = Int(floor(px)), y0 = Int(floor(py))
        let tx = px - Float(x0), ty = py - Float(y0)
        func at(_ x: Int, _ y: Int) -> Float { (x >= 0 && x < w && y >= 0 && y < h) ? m[y * w + x] : 0 }
        return at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty)
             + at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty
    }

    /// PyTorch cubic convolution weights (A = -0.75) for fractional offset t.
    static func cubicWeights(_ t: Float) -> [Float] {
        let A: Float = -0.75
        func c1(_ x: Float) -> Float { ((A + 2) * x - (A + 3)) * x * x + 1 }
        func c2(_ x: Float) -> Float { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
        return [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
    }

    /// grid_sample(mode='bicubic', padding_mode='zeros', align_corners=False) at one point, all channels.
    static func sampleBicubic(_ m: [Float], channels: Int, _ w: Int, _ h: Int, _ gx: Float, _ gy: Float,
                              into out: inout [Float], offset: Int) {
        let px = unnormalize(gx, w), py = unnormalize(gy, h)
        let x0 = Int(floor(px)), y0 = Int(floor(py))
        let wx = cubicWeights(px - Float(x0)), wy = cubicWeights(py - Float(y0))
        for c in 0..<channels {
            var acc: Float = 0
            for j in 0..<4 {
                let yy = y0 - 1 + j
                guard yy >= 0, yy < h else { continue }
                var row: Float = 0
                for i in 0..<4 {
                    let xx = x0 - 1 + i
                    guard xx >= 0, xx < w else { continue }
                    row += wx[i] * m[(c * h + yy) * w + xx]
                }
                acc += wy[j] * row
            }
            out[offset + c] = acc
        }
    }
}
```

`Sources/OSCDCore/Features/MLMultiArray+Dense.swift`:
```swift
import CoreML

public enum ModelError: Error, Equatable {
    case missingOutput(String)
    case unsupportedDataType
}

public extension MLMultiArray {
    /// Dense row-major copy of a float32 or float16 array, honouring the array's strides.
    func denseFloats() -> [Float] {
        let shape = self.shape.map(\.intValue), strides = self.strides.map(\.intValue)
        let n = shape.reduce(1, *)
        var contiguous = true
        var expected = 1
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[d] != expected { contiguous = false }
            expected *= shape[d]
        }
        var out = [Float](repeating: 0, count: n)
        let isF32 = dataType == .float32
        withUnsafeBytes { raw in
            if contiguous && isF32 {
                out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[0..<(n * 4)])) }
                return
            }
            var idx = [Int](repeating: 0, count: shape.count)
            for lin in 0..<n {
                var off = 0
                for d in 0..<shape.count { off += idx[d] * strides[d] }
                if isF32 {
                    out[lin] = raw.load(fromByteOffset: off * 4, as: Float.self)
                } else {
                    #if arch(arm64)
                    out[lin] = Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
                    #endif
                }
                var d = shape.count - 1
                while d >= 0 {
                    idx[d] += 1
                    if idx[d] < shape[d] { break }
                    idx[d] = 0
                    d -= 1
                }
            }
        }
        return out
    }
}
```

`Sources/OSCDCore/Features/XFeatModel.swift`:
```swift
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter XFeatTests` (the synthetic test runs; the parity tests skip without fixtures)
Then: `swift test -c release --filter XFeatTests.testParity`
Expected: PASS. If `testParityPostprocessMatchesOSCDDetector` fails on agreement, check the fp16 grid cast and `.toNearestOrEven` rounding first. Those two steps are where the port is most likely to drift from O-SCD.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD's XFeat post-processing and wrap the Core ML detector

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Mutual-NN matching and 8-point F-RANSAC

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Util/Random.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Matching/FundamentalRANSAC.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Matching/Matcher.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/MatcherTests.swift`

**Interfaces:**
- Consumes: `DescribedKeypoints` (Task 6).
- Produces:
  - `protocol IndexSampler { mutating func sample(_ m: Int, from n: Int) -> [Int] }`
  - `SeededSampler(seed:)` with `next()`, `uniform()`, `sample(_:from:)`, `subset(_:of:)`
  - `InjectedSampler(rows: [[Int]])`
  - `MatchSet(idx: [Int], idxOther: [Int])` with `count`
  - `Matcher.minCosSim`, `Matcher.mutualNearest(_:_:minCosSim:) -> (match12: [Int], mask: [Bool])`, `Matcher.match(_:_:) -> MatchSet`, `Matcher.matchFiltered(_:_:maxError:hypotheses:sampler:) -> MatchSet`, `Matcher.countMutual(_:_:) -> Int`
  - `FundamentalRANSAC.estimate(_:_:_:) -> [Float]` (row-major, `F[8] = 1`), `sampsonError(_:_:_:) -> Float`, `run(_:_:hypotheses:maxError:sampler:) -> FundamentalRANSAC.Result?` with fields `F`, `inliers`, `bestIndex`, `models`, `inlierCounts`

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/MatcherTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class MatcherTests: XCTestCase {
    func unit(_ v: [Float]) -> [Float] { let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n } }

    func testMutualNearestAndThreshold() {
        var a = [Float](repeating: 0, count: 3 * 64), b = [Float](repeating: 0, count: 3 * 64)
        a[0 * 64 + 0] = 1; a[1 * 64 + 1] = 1                 // a2 stays all-zero (invalid)
        b[0 * 64 + 1] = 1                                     // b0 ↔ a1
        b[1 * 64 + 0] = 1                                     // b1 ↔ a0
        let w = unit([0.5, 0.5] + [Float](repeating: 0, count: 62))   // cos 0.707 with a0/a1: below 0.82
        for c in 0..<64 { b[2 * 64 + c] = w[c] }
        let ka = DescribedKeypoints(kpts: [.zero, .zero, .zero], feats: a)
        let kb = DescribedKeypoints(kpts: [.zero, .zero, .zero], feats: b)
        let m = Matcher.match(ka, kb)
        XCTAssertEqual(m.idx, [0, 1]); XCTAssertEqual(m.idxOther, [1, 0])
        XCTAssertEqual(Matcher.countMutual(ka, kb), 2)
    }

    func testEightPointRecoversEpipolarGeometry() {
        // Two cameras 0.3 m apart; F-RANSAC must keep every true correspondence and drop the outliers.
        let K = Intrinsics(f: 300, cx: 240, cy: 180, width: 480, height: 360)
        let cam2 = Pose(R: simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 1, 0))), t: SIMD3(-0.3, 0, 0))
        var sampler = SeededSampler(seed: 1)
        var p1: [SIMD2<Float>] = [], p2: [SIMD2<Float>] = []
        for _ in 0..<200 {
            let X = SIMD3(Float(sampler.uniform()) - 0.5, Float(sampler.uniform()) - 0.5, 2 + Float(sampler.uniform()))
            p1.append(K.project(X)); p2.append(K.project(cam2.apply(X)))
        }
        for _ in 0..<40 { p1.append(SIMD2(Float(sampler.uniform()) * 480, Float(sampler.uniform()) * 360)); p2.append(SIMD2(Float(sampler.uniform()) * 480, Float(sampler.uniform()) * 360)) }
        let res = FundamentalRANSAC.run(p1, p2, hypotheses: 2000, maxError: 1.5, sampler: &sampler)!
        XCTAssertGreaterThanOrEqual(res.inliers[0..<200].filter { $0 }.count, 195)
        XCTAssertLessThanOrEqual(res.inliers[200...].filter { $0 }.count, 4)
    }

    /// Review Focus 1: fewer than 8 matches is a clean "no model", not a crash.
    func testTooFewMatchesReturnsNil() {
        var s = SeededSampler(seed: 0)
        let pts = [SIMD2<Float>](repeating: SIMD2(1, 2), count: 5)
        XCTAssertNil(FundamentalRANSAC.run(pts, pts, hypotheses: 10, maxError: 1.5, sampler: &s))
        let kp = DescribedKeypoints(kpts: [], feats: [])
        XCTAssertEqual(Matcher.matchFiltered(kp, kp, maxError: 1.5, sampler: &s).count, 0)
    }

    func testParityMutualNN() throws {
        let d = try Fixtures.dir("match")
        let a = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts_a"), feats: try Fixtures.npy(d, "feats_a"))
        let b = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts_b"), feats: try Fixtures.npy(d, "feats_b"))
        let (m12, mask) = Matcher.mutualNearest(a, b)
        let wantMask = try Fixtures.npy(d, "mask").bools(), want12 = try Fixtures.npy(d, "match12").ints()
        var agree = 0, total = 0
        for i in 0..<a.count where mask[i] || wantMask[i] {
            total += 1
            if mask[i] == wantMask[i] && m12[i] == want12[i] { agree += 1 }
        }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(max(total, 1)), 0.999)
    }

    func testParityFundamentalRANSAC() throws {
        let d = try Fixtures.dir("fransac")
        let p1 = Fixtures.vec2s(try Fixtures.npy(d, "kpts1")), p2 = Fixtures.vec2s(try Fixtures.npy(d, "kpts2"))
        let idxs = try Fixtures.npy(d, "idxs"), rows = idxs.ints()
        var sampler = InjectedSampler(rows: (0..<idxs.shape[0]).map { Array(rows[($0 * 8)..<($0 * 8 + 8)]) })
        let res = FundamentalRANSAC.run(p1, p2, hypotheses: idxs.shape[0], maxError: try Fixtures.npy(d, "max_error").floats()[0], sampler: &sampler)!
        let wantInl = try Fixtures.npy(d, "inliers").bools(), n = p1.count
        let wantBest = try Fixtures.npy(d, "best_id").ints()[0]
        let wantCount = (0..<n).filter { wantInl[wantBest * n + $0] }.count
        XCTAssertLessThanOrEqual(abs(res.inlierCounts[res.bestIndex] - wantCount), max(2, wantCount / 100))
        var agree = 0
        for k in 0..<n where res.inliers[k] == wantInl[wantBest * n + k] { agree += 1 }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(n), 0.995)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter MatcherTests`
Expected: FAIL to compile (`cannot find 'Matcher' in scope`).

- [ ] **Step 3: Implement the three source files**

`Sources/OSCDCore/Util/Random.swift`:
```swift
import Foundation

/// Draws m distinct indices from 0..<n (one RANSAC hypothesis or one subset).
public protocol IndexSampler {
    mutating func sample(_ m: Int, from n: Int) -> [Int]
}

/// SplitMix64 generator; deterministic for a seed.
public struct SeededSampler: IndexSampler, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func uniform() -> Double { Double(next() >> 11) * 0x1.0p-53 }

    public mutating func sample(_ m: Int, from n: Int) -> [Int] {
        precondition(m <= n, "cannot sample \(m) of \(n)")
        if m > 16 { return subset(m, of: n) }
        var picked: [Int] = []
        picked.reserveCapacity(m)
        while picked.count < m {
            let k = Int(next() % UInt64(n))
            if !picked.contains(k) { picked.append(k) }
        }
        return picked
    }

    /// m distinct indices from 0..<n by a partial Fisher–Yates shuffle.
    public mutating func subset(_ m: Int, of n: Int) -> [Int] {
        var a = Array(0..<n)
        for i in 0..<m {
            let j = i + Int(next() % UInt64(n - i))
            a.swapAt(i, j)
        }
        return Array(a.prefix(m))
    }
}

/// Replays index sets recorded from O-SCD (parity tests).
public struct InjectedSampler: IndexSampler {
    private let rows: [[Int]]
    private var position = 0

    public init(rows: [[Int]]) { self.rows = rows }

    public mutating func sample(_ m: Int, from n: Int) -> [Int] {
        defer { position += 1 }
        return Array(rows[position].prefix(m))
    }
}
```

`Sources/OSCDCore/Matching/FundamentalRANSAC.swift`:
```swift
import Foundation

/// 8-point fundamental-matrix RANSAC as in O-SCD poses/ransac.cu (OpenCV usac solver, no point
/// normalisation) with Sampson-error inliers.
public enum FundamentalRANSAC {
    public struct Result: Sendable {
        public var F: [Float]
        public var inliers: [Bool]
        public var bestIndex: Int
        public var models: [[Float]]
        public var inlierCounts: [Int]
    }

    /// Row-major F with F[8] = 1, from 8 correspondences selected by `idx`.
    public static func estimate(_ p1: [SIMD2<Float>], _ p2: [SIMD2<Float>], _ idx: [Int]) -> [Float] {
        let m = 8, n = 9
        var a = [Float](repeating: 0, count: m * n)
        for row in 0..<m {
            let u1 = p1[idx[row]], u2 = p2[idx[row]]
            let x1 = u1.x, y1 = u1.y, x2 = u2.x, y2 = u2.y
            a[row * 9 + 0] = x2 * x1; a[row * 9 + 1] = x2 * y1; a[row * 9 + 2] = x2
            a[row * 9 + 3] = y2 * x1; a[row * 9 + 4] = y2 * y1; a[row * 9 + 5] = y2
            a[row * 9 + 6] = x1;      a[row * 9 + 7] = y1;      a[row * 9 + 8] = 1
        }
        // eliminateUpperTriangular
        for r in 0..<m {
            var pivot = a[r * n + r]
            var rowWithPivot = r
            for k in (r + 1)..<m where abs(pivot) < abs(a[k * n + r]) {
                pivot = a[k * n + r]
                rowWithPivot = k
            }
            if abs(pivot) < 1e-8 { continue }
            for c in r..<n { a.swapAt(rowWithPivot * n + c, r * n + c) }
            for j in (r + 1)..<m {
                let fac = a[j * n + r] / pivot
                a[j * n + r] = 0
                for c in (r + 1)..<n { a[j * n + c] -= fac * a[r * n + c] }
            }
        }
        // back-substitution with f[8] = 1
        var f = [Float](repeating: 0, count: 9)
        f[8] = 1
        for i in stride(from: m - 1, through: 0, by: -1) {
            var acc: Float = 0
            for j in (i + 1)..<n { acc -= a[i * n + j] * f[j] }
            f[i] = acc / a[i * n + i]
        }
        return f
    }

    @inline(__always) public static func sampsonError(_ F: [Float], _ u1: SIMD2<Float>, _ u2: SIMD2<Float>) -> Float {
        let x1 = u1.x, y1 = u1.y, x2 = u2.x, y2 = u2.y
        let fx = F[0] * x1 + F[1] * y1 + F[2], fy = F[3] * x1 + F[4] * y1 + F[5]
        let gx = x2 * F[0] + y2 * F[3] + F[6], gy = x2 * F[1] + y2 * F[4] + F[7]
        let e = x2 * fx + y2 * fy + F[6] * x1 + F[7] * y1 + F[8]
        return e * e / (fx * fx + fy * fy + gx * gx + gy * gy)
    }

    /// Nil when fewer than 8 correspondences (O-SCD would raise here; P1 skips the frame).
    public static func run<S: IndexSampler>(_ p1: [SIMD2<Float>], _ p2: [SIMD2<Float>], hypotheses: Int = 2000,
                                            maxError: Float, sampler: inout S) -> Result? {
        let n = p1.count
        guard n >= 8, hypotheses > 0 else { return nil }
        let idx = (0..<hypotheses).map { _ in sampler.sample(8, from: n) }
        let threshold = maxError * maxError
        var models = [[Float]](repeating: [], count: hypotheses)
        var counts = [Int](repeating: 0, count: hypotheses)
        models.withUnsafeMutableBufferPointer { M in
            counts.withUnsafeMutableBufferPointer { C in
                DispatchQueue.concurrentPerform(iterations: hypotheses) { h in
                    let F = estimate(p1, p2, idx[h])
                    var c = 0
                    for k in 0..<n where sampsonError(F, p1[k], p2[k]) < threshold { c += 1 }
                    M[h] = F
                    C[h] = c
                }
            }
        }
        var best = 0
        for h in 1..<hypotheses where counts[h] > counts[best] { best = h }
        let F = models[best]
        return Result(F: F, inliers: (0..<n).map { sampsonError(F, p1[$0], p2[$0]) < threshold },
                      bestIndex: best, models: models, inlierCounts: counts)
    }
}
```

`Sources/OSCDCore/Matching/Matcher.swift`:
```swift
import Accelerate
import Foundation

/// Corresponding keypoint indices between two keypoint sets.
public struct MatchSet: Sendable, Equatable {
    public var idx: [Int]
    public var idxOther: [Int]
    public init(idx: [Int], idxOther: [Int]) { self.idx = idx; self.idxOther = idxOther }
    public var count: Int { idx.count }
}

/// O-SCD poses/matcher.py.
public enum Matcher {
    public static let minCosSim: Float = 0.82

    /// For every row of `x`: best cosine similarity against `y` and its index (first on ties).
    static func rowArgmax(_ x: DescribedKeypoints, _ y: DescribedKeypoints) -> (best: [Float], arg: [Int]) {
        let nx = x.count, ny = y.count, d = DescribedKeypoints.dim
        var best = [Float](repeating: -.infinity, count: nx)
        var arg = [Int](repeating: 0, count: nx)
        guard nx > 0, ny > 0 else { return (best, arg) }
        let block = 512
        var sim = [Float](repeating: 0, count: block * ny)
        x.feats.withUnsafeBufferPointer { X in
            y.feats.withUnsafeBufferPointer { Y in
                sim.withUnsafeMutableBufferPointer { S in
                    var start = 0
                    while start < nx {
                        let rows = min(block, nx - start)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(ny), Int32(d),
                                    1, X.baseAddress! + start * d, Int32(d), Y.baseAddress!, Int32(d),
                                    0, S.baseAddress!, Int32(ny))
                        for r in 0..<rows {
                            var m: Float = 0
                            var mi: vDSP_Length = 0
                            vDSP_maxvi(S.baseAddress! + r * ny, 1, &m, &mi, vDSP_Length(ny))
                            best[start + r] = m
                            arg[start + r] = Int(mi)
                        }
                        start += rows
                    }
                }
            }
        }
        return (best, arg)
    }

    /// O-SCD `match`: match12 for every row of `a` and the mutual-NN + cos > minCosSim mask.
    public static func mutualNearest(_ a: DescribedKeypoints, _ b: DescribedKeypoints,
                                     minCosSim: Float = minCosSim) -> (match12: [Int], mask: [Bool]) {
        let (best12, m12) = rowArgmax(a, b)
        let (_, m21) = rowArgmax(b, a)
        var mask = [Bool](repeating: false, count: a.count)
        for i in 0..<a.count where !m21.isEmpty { mask[i] = m21[m12[i]] == i && best12[i] > minCosSim }
        return (m12, mask)
    }

    public static func match(_ a: DescribedKeypoints, _ b: DescribedKeypoints) -> MatchSet {
        let (m12, mask) = mutualNearest(a, b)
        var s = MatchSet(idx: [], idxOther: [])
        for i in 0..<a.count where mask[i] {
            s.idx.append(i)
            s.idxOther.append(m12[i])
        }
        return s
    }

    /// O-SCD Matcher(..., remove_outliers=True): mutual NN, then keep the 8-point F-RANSAC inliers.
    public static func matchFiltered<S: IndexSampler>(_ a: DescribedKeypoints, _ b: DescribedKeypoints,
                                                      maxError: Float, hypotheses: Int = 2000,
                                                      sampler: inout S) -> MatchSet {
        let m = match(a, b)
        let p1 = m.idx.map { a.kpts[$0] }, p2 = m.idxOther.map { b.kpts[$0] }
        guard let res = FundamentalRANSAC.run(p1, p2, hypotheses: hypotheses, maxError: maxError, sampler: &sampler) else {
            return MatchSet(idx: [], idxOther: [])
        }
        var s = MatchSet(idx: [], idxOther: [])
        for k in 0..<m.count where res.inliers[k] {
            s.idx.append(m.idx[k])
            s.idxOther.append(m.idxOther[k])
        }
        return s
    }

    /// O-SCD `Matcher.evaluate_match`: number of mutual-NN matches (used for retrieval).
    public static func countMutual(_ a: DescribedKeypoints, _ b: DescribedKeypoints) -> Int {
        mutualNearest(a, b).mask.reduce(0) { $0 + ($1 ? 1 : 0) }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter MatcherTests` then `swift test -c release --filter MatcherTests.testParity`
Expected: PASS (the parity tests skip without fixtures). `cblas_sgemm` may print a deprecation warning; that is fine.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD mutual-NN matching and 8-point F-RANSAC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Mini bundle adjustment (pose-only LM)

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Util/LinearSolve.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Pose/MiniBA.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/MiniBATests.swift`

**Interfaces:**
- Consumes: `SixD` (Task 2).
- Produces:
  - `LinearSolve.solve(_ A: [Double], _ b: [Double], n: Int) -> [Double]`. It returns NaNs when A is singular.
  - `MiniBAConfig(iters:madScale:)` with fields `huberDelta`, `lm`, `ep`, `k`, and statics `.poseRefine` (20 iterations, MAD 4) and `.p4p` (20 iterations, MAD 0).
  - `MiniBAResult` with `r6`, `t`, `residuals`, `initialResiduals`, `mask`, `inlierComponents`.
  - `MiniBA.solve(r6:t:f:centre:xyz:uv:config:) -> MiniBAResult`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/MiniBATests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class MiniBATests: XCTestCase {
    func synthetic(n: Int, outliers: Int, seed: UInt64) -> (truth: Pose, xyz: [SIMD3<Double>], uv: [SIMD2<Double>]) {
        var s = SeededSampler(seed: seed)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.4, axis: simd_normalize(SIMD3(0.2, 1, 0.1)))), t: SIMD3(0.05, -0.1, 0.8))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var xyz: [SIMD3<Double>] = [], uv: [SIMD2<Double>] = []
        for i in 0..<n {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i < n - outliers { p += SIMD2(Float(s.uniform() - 0.5) * 0.6, Float(s.uniform() - 0.5) * 0.6) }
            else { p += SIMD2(40, -30) }
            xyz.append(SIMD3<Double>(X)); uv.append(SIMD2<Double>(p))
        }
        return (truth, xyz, uv)
    }

    func testRecoversPoseAndMasksOutliers() {
        let (truth, xyz, uv) = synthetic(n: 300, outliers: 30, seed: 3)
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.03, axis: SIMD3(1, 0, 0))), t: truth.t + SIMD3(0.02, 0, -0.02))
        let res = MiniBA.solve(r6: start.r6, t: start.tDouble, f: 360, centre: SIMD2(239.5, 179.5), xyz: xyz, uv: uv, config: .poseRefine)
        let got = Pose(r6: res.r6, t: res.t)
        XCTAssertLessThan(got.rotationDegrees(to: truth), 0.05)
        XCTAssertLessThan(simd_distance(got.t, truth.t), 0.002)
        XCTAssertEqual(res.mask[(2 * 270)...].reduce(0, +), 0)            // the 30 outliers are masked
        XCTAssertGreaterThan(res.inlierComponents, 2 * 260)
    }

    /// Review Focus 2: padded dummies at depth 0 and a fully padded problem never produce NaN.
    func testDegenerateInputsStayFinite() {
        let r6 = Pose.identity.r6
        let allPadded = MiniBA.solve(r6: r6, t: SIMD3(0, 0, 0), f: 360, centre: SIMD2(239.5, 179.5),
                                     xyz: [SIMD3<Double>](repeating: .zero, count: 50),
                                     uv: [SIMD2<Double>](repeating: SIMD2(-1, -1), count: 50), config: .poseRefine)
        XCTAssertTrue(allPadded.r6.allSatisfy(\.isFinite))
        XCTAssertTrue([allPadded.t.x, allPadded.t.y, allPadded.t.z].allSatisfy(\.isFinite))
        XCTAssertEqual(allPadded.inlierComponents, 0)

        var (_, xyz, uv) = synthetic(n: 60, outliers: 0, seed: 9)
        xyz += [SIMD3<Double>](repeating: .zero, count: 40); uv += [SIMD2<Double>](repeating: SIMD2(-1, -1), count: 40)
        let res = MiniBA.solve(r6: r6, t: SIMD3(0, 0, 0), f: 360, centre: SIMD2(239.5, 179.5), xyz: xyz, uv: uv, config: .poseRefine)
        XCTAssertTrue(res.r6.allSatisfy(\.isFinite))
    }

    func testParityMiniBA() throws {
        _ = try Fixtures.dir("pose")
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let meta = try Fixtures.json(Fixtures.root.appendingPathComponent("meta.json"))
            let f = meta["f"] as! Double, c = SIMD2(meta["cx"] as! Double, meta["cy"] as! Double)
            let xyz = Fixtures.vec3s(try Fixtures.npy(d, "ba_in_xyz")).map { SIMD3<Double>($0) }
            let uv = Fixtures.vec2s(try Fixtures.npy(d, "ba_in_uv")).map { SIMD2<Double>($0) }
            let r6 = try Fixtures.npy(d, "ba_in_r6").floats().map(Double.init)
            let t = try Fixtures.npy(d, "ba_in_t").floats().map(Double.init)
            let res = MiniBA.solve(r6: r6, t: SIMD3(t[0], t[1], t[2]), f: f, centre: c, xyz: xyz, uv: uv, config: .poseRefine)
            let want = Pose(r6: try Fixtures.npy(d, "ba_out_r6").floats().map(Double.init),
                            t: SIMD3<Double>(Fixtures.vec3s(try Fixtures.npy(d, "ba_out_t"))[0]))
            let got = Pose(r6: res.r6, t: res.t)
            XCTAssertLessThan(got.rotationDegrees(to: want), 0.1, "frame \(k)")
            XCTAssertLessThan(simd_distance(got.t, want.t), 1e-3 * max(1, simd_length(want.t)), "frame \(k)")
            let wantMask = try Fixtures.npy(d, "ba_mask").floats()
            let agree = zip(res.mask, wantMask).filter { Float($0) == $1 }.count
            XCTAssertGreaterThanOrEqual(Double(agree) / Double(wantMask.count), 0.99, "frame \(k)")
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter MiniBATests`
Expected: FAIL to compile (`cannot find 'MiniBA' in scope`).

- [ ] **Step 3: Implement `LinearSolve.swift` and `MiniBA.swift`**

`Sources/OSCDCore/Util/LinearSolve.swift`:
```swift
/// Dense Gaussian elimination with partial pivoting for small systems. Returns NaNs when singular.
public enum LinearSolve {
    public static func solve(_ A0: [Double], _ b0: [Double], n: Int) -> [Double] {
        var A = A0, b = b0
        for col in 0..<n {
            var piv = col
            for r in (col + 1)..<n where abs(A[r * n + col]) > abs(A[piv * n + col]) { piv = r }
            if !(abs(A[piv * n + col]) > 1e-300) { return [Double](repeating: .nan, count: n) }
            if piv != col {
                for c in 0..<n { A.swapAt(piv * n + c, col * n + c) }
                b.swapAt(piv, col)
            }
            for r in (col + 1)..<n {
                let f = A[r * n + col] / A[col * n + col]
                if f == 0 { continue }
                for c in col..<n { A[r * n + c] -= f * A[col * n + c] }
                b[r] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<n { s -= A[r * n + c] * x[c] }
            x[r] = s / A[r * n + r]
        }
        return x
    }
}
```

`Sources/OSCDCore/Pose/MiniBA.swift`:
```swift
import simd

public struct MiniBAConfig: Sendable {
    public var iters: Int
    public var huberDelta: Double = 1
    public var madScale: Double            // 0 disables MAD outlier rejection
    public var lm: Double = 1e-5
    public var ep: Double = 1e-2
    public var k: Double = 2

    public init(iters: Int, madScale: Double) {
        self.iters = iters
        self.madScale = madScale
    }

    /// O-SCD PoseInitializer.miniBA: MiniBA defaults with iters_miniba_incr = 20.
    public static let poseRefine = MiniBAConfig(iters: 20, madScale: 4)
    /// O-SCD RANSACEstimator P4P hypothesis solver.
    public static let p4p = MiniBAConfig(iters: 20, madScale: 0)
}

public struct MiniBAResult: Sendable {
    public var r6: [Double]
    public var t: SIMD3<Double>
    /// Residuals at the last *tentative* parameters, as O-SCD returns them (code quirk).
    public var residuals: [Double]
    public var initialResiduals: [Double]
    /// Per residual component (2 per point): 1 = inlier, 0 = masked.
    public var mask: [Double]
    public var inlierComponents: Int { Int(mask.reduce(0, +)) }
}

/// Pose-only Levenberg–Marquardt over O-SCD's 9 raw parameters (6D rotation + translation),
/// ported from poses/mini_ba.py with optimize_focal = optimize_3Dpts = False. Points whose uv
/// has a negative coordinate are padding and never contribute.
public enum MiniBA {
    public static func solve(r6 r6Init: [Double], t tInit: SIMD3<Double>, f: Double, centre: SIMD2<Double>,
                             xyz: [SIMD3<Double>], uv: [SIMD2<Double>], config: MiniBAConfig) -> MiniBAResult {
        let n = xyz.count, m = 2 * n
        let valid = uv.map { $0.x >= 0 && $0.y >= 0 }
        var p = r6Init + [tInit.x, tInit.y, tInit.z]
        var pTmp = p
        var lm = config.lm
        var initial: [Double] = []
        for it in 0..<config.iters {
            let rIn = residuals(p, f, centre, xyz, uv)
            if it == 0 { initial = rIn }
            let J = jacobian(p, f, centre, xyz, uv)
            let mask = componentMask(rIn, valid: valid, madScale: config.madScale)
            var w = [Double](repeating: 0, count: m), r = [Double](repeating: 0, count: m)
            for i in 0..<m where mask[i] != 0 {
                w[i] = huberWeight(rIn[i], config.huberDelta)
                r[i] = rIn[i] * w[i]
            }
            var JtJ = [Double](repeating: 0, count: 81), Jtr = [Double](repeating: 0, count: 9)
            for i in 0..<m where w[i] != 0 {
                for a in 0..<9 {
                    let ja = J[i * 9 + a] * w[i]
                    Jtr[a] += ja * r[i]
                    for b in 0..<9 { JtJ[a * 9 + b] += ja * J[i * 9 + b] * w[i] }
                }
            }
            for a in 0..<9 { JtJ[a * 9 + a] = max(JtJ[a * 9 + a] * (1 + lm), config.ep) }
            let d = LinearSolve.solve(JtJ, Jtr, n: 9).map { $0.isFinite ? $0 : 0 }
            for k in 0..<9 { pTmp[k] = p[k] - d[k] }
            let rNew = residuals(pTmp, f, centre, xyz, uv)
            var newSq = 0.0, oldSq = 0.0
            for i in 0..<m where mask[i] != 0 {
                let v = rNew[i] * huberWeight(rNew[i], config.huberDelta)
                newSq += v * v
                oldSq += r[i] * r[i]
            }
            let success = newSq < oldSq
            if success { p = pTmp }
            lm *= success ? 1 / config.k : config.k
            let r6n = SixD.fromMatrix(SixD.toMatrix(p[0..<6]))   // re-orthonormalise every iteration
            for k in 0..<6 { p[k] = r6n[k] }
        }
        let rFinal = residuals(pTmp, f, centre, xyz, uv)
        return MiniBAResult(r6: Array(p[0..<6]), t: SIMD3(p[6], p[7], p[8]), residuals: rFinal,
                            initialResiduals: initial,
                            mask: componentMask(rFinal, valid: valid, madScale: config.madScale))
    }

    static func residuals(_ p: [Double], _ f: Double, _ c: SIMD2<Double>,
                          _ xyz: [SIMD3<Double>], _ uv: [SIMD2<Double>]) -> [Double] {
        let R = SixD.toMatrix(p[0..<6]), t = SIMD3(p[6], p[7], p[8])
        var r = [Double](repeating: 0, count: 2 * xyz.count)
        for i in 0..<xyz.count {
            let q = R * xyz[i] + t
            r[2 * i] = f * q.x / q.z + c.x - uv[i].x
            r[2 * i + 1] = f * q.y / q.z + c.y - uv[i].y
        }
        return r
    }

    /// Central differences in Double (O-SCD uses forward-mode autodiff in fp32).
    static func jacobian(_ p: [Double], _ f: Double, _ c: SIMD2<Double>,
                         _ xyz: [SIMD3<Double>], _ uv: [SIMD2<Double>]) -> [Double] {
        let m = 2 * xyz.count
        var J = [Double](repeating: 0, count: m * 9)
        var q = p
        for k in 0..<9 {
            let h = 1e-6 * max(1, abs(p[k]))
            q[k] = p[k] + h
            let rp = residuals(q, f, c, xyz, uv)
            q[k] = p[k] - h
            let rm = residuals(q, f, c, xyz, uv)
            q[k] = p[k]
            for i in 0..<m { J[i * 9 + k] = (rp[i] - rm[i]) / (2 * h) }
        }
        return J
    }

    @inline(__always) static func huberWeight(_ r: Double, _ delta: Double) -> Double {
        abs(r) <= delta ? 1 : delta / abs(r).squareRoot()
    }

    /// O-SCD get_mask: optional MAD rejection on per-point error, expanded to both components.
    static func componentMask(_ r: [Double], valid: [Bool], madScale: Double) -> [Double] {
        let n = valid.count
        var mask = [Double](repeating: 0, count: 2 * n)
        guard n > 0 else { return mask }
        if madScale > 0 {
            var err = [Double](repeating: 0, count: n)
            var nValid = 0
            for i in 0..<n where valid[i] {
                err[i] = (r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]).squareRoot()
                nValid += 1
            }
            let q = 1 - 0.5 * Double(nValid) / Double(n)
            let med = quantile(err, q)
            let mad = quantile(err.map { abs($0 - med) }, q)
            let cutoff = max(med + madScale * mad, 5)
            for i in 0..<n where valid[i] && err[i] < cutoff { mask[2 * i] = 1; mask[2 * i + 1] = 1 }
        } else {
            for i in 0..<n where valid[i] { mask[2 * i] = 1; mask[2 * i + 1] = 1 }
        }
        return mask
    }

    /// torch.quantile with linear interpolation.
    static func quantile(_ v: [Double], _ q: Double) -> Double {
        let s = v.sorted()
        let pos = q * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }
}
```

Note on degenerate input: O-SCD multiplies a masked residual by a zero weight, so a padded point at depth 0 turns into NaN there. The `where mask[i] != 0` guards skip those components entirely, which keeps every result finite (Review Focus 2).

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter MiniBATests` then `swift test -c release --filter MiniBATests.testParity`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD's pose-only mini bundle adjustment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: P4P-RANSAC

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Pose/P4PRANSAC.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/P4PTests.swift`

**Interfaces:**
- Consumes: `MiniBA.solve`, `MiniBAConfig.p4p` (Task 8); `IndexSampler` (Task 7); `SixD` (Task 2).
- Produces:
  - `P4PRANSAC.run(uv:xyz:confs:f:centre:initR6:initT:hypotheses:maxError:sampler:) -> P4PRANSAC.Result?`. It returns nil when there are fewer than 4 points.
  - `Result` has `r6`, `t`, `inliers`, `bestIndex`, `inlierScores`, `models: [(R: simd_double3x3, t: SIMD3<Double>)]`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/P4PTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class P4PTests: XCTestCase {
    func testRecoversPoseWithHalfOutliers() {
        var s = SeededSampler(seed: 5)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(0, 1, 0.3)))), t: SIMD3(0.1, 0.05, 0.9))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = []
        for i in 0..<400 {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i % 2 == 1 { p = SIMD2(Float(s.uniform()) * 480, Float(s.uniform()) * 360) }
            uv.append(SIMD2<Double>(p)); xyz.append(SIMD3<Double>(X))
        }
        // Start 6° / 5 cm away, as from the top-retrieved reference keyframe.
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 0, 1))), t: truth.t + SIMD3(0.05, 0, 0))
        let res = P4PRANSAC.run(uv: uv, xyz: xyz, confs: [Double](repeating: 1, count: 400), f: 360,
                                centre: SIMD2(239.5, 179.5), initR6: start.r6, initT: start.tDouble,
                                hypotheses: 500, maxError: 1.5, sampler: &s)!
        XCTAssertLessThan(Pose(r6: res.r6, t: res.t).rotationDegrees(to: truth), 0.5)
        XCTAssertGreaterThanOrEqual((0..<400).filter { $0 % 2 == 0 && res.inliers[$0] }.count, 190)
    }

    func testFewerThanFourPointsIsNil() {
        var s = SeededSampler(seed: 0)
        XCTAssertNil(P4PRANSAC.run(uv: [SIMD2(1, 1)], xyz: [SIMD3(0, 0, 1)], confs: [1], f: 360, centre: .zero,
                                   initR6: Pose.identity.r6, initT: .zero, maxError: 1.5, sampler: &s))
    }

    func testParityP4P() throws {
        _ = try Fixtures.dir("pose")
        let meta = try Fixtures.json(Fixtures.root.appendingPathComponent("meta.json"))
        let f = meta["f"] as! Double, c = SIMD2(meta["cx"] as! Double, meta["cy"] as! Double)
        let maxErr = meta["max_error"] as! Double
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let uv = Fixtures.vec2s(try Fixtures.npy(d, "corr_uv")).map { SIMD2<Double>($0) }
            let xyz = Fixtures.vec3s(try Fixtures.npy(d, "corr_xyz")).map { SIMD3<Double>($0) }
            let conf = try Fixtures.npy(d, "corr_conf").floats().map(Double.init)
            let idxs = try Fixtures.npy(d, "p4p_idxs"), rows = idxs.ints()
            var sampler = InjectedSampler(rows: (0..<idxs.shape[0]).map { Array(rows[($0 * 4)..<($0 * 4 + 4)]) })
            let initT = try Fixtures.npy(d, "p4p_init_t").floats().map(Double.init)
            let res = P4PRANSAC.run(uv: uv, xyz: xyz, confs: conf, f: f, centre: c,
                                    initR6: try Fixtures.npy(d, "p4p_init_r6").floats().map(Double.init),
                                    initT: SIMD3(initT[0], initT[1], initT[2]),
                                    hypotheses: idxs.shape[0], maxError: maxErr, sampler: &sampler)!
            // Every hypothesis model against O-SCD's (row-major 3×4 [R | t]).
            let models = try Fixtures.npy(d, "p4p_models").floats()
            var close = 0
            for h in 0..<idxs.shape[0] {
                let m = Array(models[(h * 12)..<(h * 12 + 12)])
                let want = Fixtures.pose(rowMajor4x4: m + [0, 0, 0, 1])
                let got = Pose(R: simd_float3x3(converting: res.models[h].R), t: SIMD3<Float>(res.models[h].t))
                if got.rotationDegrees(to: want) < 0.05 { close += 1 }
            }
            XCTAssertGreaterThanOrEqual(Double(close) / Double(idxs.shape[0]), 0.98, "frame \(k)")
            let wantInl = try Fixtures.npy(d, "p4p_inliers").floats(), n = uv.count
            let best = try Fixtures.npy(d, "p4p_best").ints()[0]
            let agree = (0..<n).filter { res.inliers[$0] == (wantInl[best * n + $0] > 0) }.count
            XCTAssertGreaterThanOrEqual(Double(agree) / Double(n), 0.99, "frame \(k)")
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter P4PTests`
Expected: FAIL to compile (`cannot find 'P4PRANSAC' in scope`).

- [ ] **Step 3: Implement `P4PRANSAC.swift`**

```swift
import Foundation
import simd

/// O-SCD RANSACEstimator(EstimatorType.P4P): every hypothesis is a 20-iteration pose-only MiniBA
/// on 4 random correspondences, started from the same initial pose (the top-retrieved keyframe).
public enum P4PRANSAC {
    public struct Result {
        public var r6: [Double]
        public var t: SIMD3<Double>
        public var inliers: [Bool]
        public var bestIndex: Int
        public var inlierScores: [Double]
        public var models: [(R: simd_double3x3, t: SIMD3<Double>)]
    }

    @inline(__always) static func isInlier(_ R: simd_double3x3, _ t: SIMD3<Double>, _ X: SIMD3<Double>,
                                           _ uv: SIMD2<Double>, _ f: Double, _ c: SIMD2<Double>, _ maxError: Double) -> Bool {
        let q = R * X + t
        let px = SIMD2(f * q.x / q.z, f * q.y / q.z) + c
        return simd_length(uv - px) < maxError
    }

    public static func run<S: IndexSampler>(uv: [SIMD2<Double>], xyz: [SIMD3<Double>], confs: [Double],
                                            f: Double, centre: SIMD2<Double>, initR6: [Double], initT: SIMD3<Double>,
                                            hypotheses: Int = 2000, maxError: Double, sampler: inout S) -> Result? {
        let n = uv.count
        guard n >= 4, hypotheses > 0 else { return nil }
        let idx = (0..<hypotheses).map { _ in sampler.sample(4, from: n) }
        var models = [(R: simd_double3x3, t: SIMD3<Double>)](repeating: (matrix_identity_double3x3, .zero), count: hypotheses)
        var scores = [Double](repeating: 0, count: hypotheses)
        models.withUnsafeMutableBufferPointer { M in
            scores.withUnsafeMutableBufferPointer { S in
                DispatchQueue.concurrentPerform(iterations: hypotheses) { h in
                    let sel = idx[h]
                    let res = MiniBA.solve(r6: initR6, t: initT, f: f, centre: centre,
                                           xyz: sel.map { xyz[$0] }, uv: sel.map { uv[$0] }, config: .p4p)
                    let R = SixD.toMatrix(res.r6)
                    var s = 0.0
                    for k in 0..<n where isInlier(R, res.t, xyz[k], uv[k], f, centre, maxError) { s += confs[k] }
                    M[h] = (R, res.t)
                    S[h] = s
                }
            }
        }
        var best = 0
        for h in 1..<hypotheses where scores[h] > scores[best] { best = h }
        let (R, t) = models[best]
        let inliers = (0..<n).map { isInlier(R, t, xyz[$0], uv[$0], f, centre, maxError) && confs[$0] > 0 }
        return Result(r6: SixD.fromMatrix(R), t: t, inliers: inliers, bestIndex: best, inlierScores: scores, models: models)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter P4PTests` then `swift test -c release --filter P4PTests.testParity`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD's P4P-RANSAC (LM hypotheses from the top reference pose)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Triangulation and reference anchors

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Pose/Triangulator.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Pose/AnchorBuilder.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/TriangulatorTests.swift`

**Interfaces:**
- Consumes: `Matcher.matchFiltered`, `SeededSampler` (Task 7); `Pose`, `Intrinsics` (Task 2); `DescribedKeypoints` (Task 6).
- Produces:
  - `Triangulator.run(uv:uvsOthers:pose:posesOthers:intrinsics:maxError:minDisambiguation:disambiguationDepths:) -> Triangulator.Output`. Output fields: `pts3d` (world), `depth` (camera z), `bestDisambiguation`, `valid`.
  - `KeyframeAnchors` with `index`, `pose`, `keypoints`, `matches: [Int: MatchSet]`, `hasPt3d`, `pts3d`, `conf`.
  - `AnchorSet` with `keyframes`, `intrinsics`, `maxError`, `anchorCount`, `yield`.
  - `AnchorBuilder.maxError(width:matchMaxError:) -> Float`.
  - `AnchorBuilder.build(keyframes: [(pose: Pose, keypoints: DescribedKeypoints)], intrinsics:triangulationCams:hypotheses:seed:disambiguationDepths:) -> AnchorSet`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/TriangulatorTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class TriangulatorTests: XCTestCase {
    let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)

    func testTwoViewTriangulationAndParallaxGate() {
        let a = Pose.identity
        let b = Pose(R: matrix_identity_float3x3, t: SIMD3(-0.5, 0, 0))          // 0.5 m baseline
        let X = SIMD3<Float>(0.1, -0.05, 2)
        let out = Triangulator.run(uv: [K.project(a.apply(X))], uvsOthers: [[K.project(b.apply(X))]],
                                   pose: a, posesOthers: [b], intrinsics: K, maxError: 1.5, minDisambiguation: 45)
        XCTAssertTrue(out.valid[0])
        XCTAssertLessThan(simd_distance(out.pts3d[0], X), 1e-3)
        // A 2 cm baseline gives too little parallax between ray depths 1 and 10: rejected.
        let c = Pose(R: matrix_identity_float3x3, t: SIMD3(-0.02, 0, 0))
        let weak = Triangulator.run(uv: [K.project(X)], uvsOthers: [[K.project(c.apply(X))]],
                                    pose: a, posesOthers: [c], intrinsics: K, maxError: 1.5, minDisambiguation: 45)
        XCTAssertFalse(weak.valid[0])
    }

    /// Three views of a synthetic textured scene: anchors come back at the right 3D positions.
    func testAnchorBuilderOnSyntheticScene() {
        var s = SeededSampler(seed: 11)
        let points = (0..<300).map { _ in SIMD3(Float(s.uniform()) * 2 - 1, Float(s.uniform()) * 1 - 0.5, 3 + Float(s.uniform())) }
        let descriptors: [[Float]] = points.map { _ in
            let v = (0..<64).map { _ in Float(s.uniform() - 0.5) }
            let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n }
        }
        let poses = [Pose.identity, Pose(R: matrix_identity_float3x3, t: SIMD3(-0.6, 0, 0)),
                     Pose(R: simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 1, 0))), t: SIMD3(0.6, 0.1, 0))]
        let views = poses.map { pose -> (pose: Pose, keypoints: DescribedKeypoints) in
            var kpts: [SIMD2<Float>] = [], feats: [Float] = []
            for (X, d) in zip(points, descriptors) {
                let uv = K.project(pose.apply(X))
                guard uv.x > 1, uv.y > 1, uv.x < 478, uv.y < 358 else { continue }
                kpts.append(uv); feats += d
            }
            return (pose, DescribedKeypoints(kpts: kpts, feats: feats))
        }
        let anchors = AnchorBuilder.build(keyframes: views, intrinsics: K, hypotheses: 300)
        XCTAssertGreaterThan(anchors.yield, 0.8)
        let kf = anchors.keyframes[0]
        var worst: Float = 0
        for i in 0..<kf.keypoints.count where kf.hasPt3d[i] {
            let truth = points.min { simd_distance(K.project($0), kf.keypoints.kpts[i]) < simd_distance(K.project($1), kf.keypoints.kpts[i]) }!
            worst = max(worst, simd_distance(kf.pts3d[i], truth))
        }
        XCTAssertLessThan(worst, 0.01)
    }

    func testParityTriangulator() throws {
        let d = try Fixtures.dir("triangulate")
        let uvs = try Fixtures.npy(d, "uvs_others"), flat = Fixtures.vec2s(uvs)
        let nPts = uvs.shape[1]
        let others = (0..<uvs.shape[0]).map { Array(flat[($0 * nPts)..<(($0 + 1) * nPts)]) }
        let rts = try Fixtures.npy(d, "Rts_others").floats()
        let posesOthers = (0..<uvs.shape[0]).map { Fixtures.pose(rowMajor4x4: Array(rts[($0 * 16)..<($0 * 16 + 16)])) }
        let centre = try Fixtures.npy(d, "centre").floats()
        let Kf = Intrinsics(f: try Fixtures.npy(d, "f").floats()[0], cx: centre[0], cy: centre[1], width: 0, height: 0)
        let out = Triangulator.run(uv: Fixtures.vec2s(try Fixtures.npy(d, "uv")), uvsOthers: others,
                                   pose: Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, "Rt").floats()),
                                   posesOthers: posesOthers, intrinsics: Kf,
                                   maxError: try Fixtures.npy(d, "max_error").floats()[0],
                                   minDisambiguation: try Fixtures.npy(d, "min_dis").floats()[0])
        let wantValid = try Fixtures.npy(d, "valid").bools(), wantPts = Fixtures.vec3s(try Fixtures.npy(d, "pts3d"))
        XCTAssertGreaterThanOrEqual(Double(zip(out.valid, wantValid).filter { $0 == $1 }.count) / Double(nPts), 0.995)
        var far = 0, both = 0
        for i in 0..<nPts where out.valid[i] && wantValid[i] {
            both += 1
            if simd_distance(out.pts3d[i], wantPts[i]) > 1e-3 * max(1, simd_length(wantPts[i])) { far += 1 }
        }
        XCTAssertLessThanOrEqual(Double(far) / Double(max(both, 1)), 0.005)
    }

    /// Full anchor building from O-SCD's own keypoints and poses: the same keypoints get anchors.
    func testParityAnchorBuilder() throws {
        let d = try Fixtures.dir("anchors")
        let info = try Fixtures.json(d.appendingPathComponent("info.json"))
        let count = info["count"] as! Int
        let K = Intrinsics(f: Float(info["f"] as! Double), cx: Float(info["cx"] as! Double), cy: Float(info["cy"] as! Double),
                           width: info["width"] as! Int, height: info["height"] as! Int)
        var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
        for i in 0..<count {
            let p = String(format: "kf_%04d_", i)
            views.append((Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, p + "Rt").floats()),
                          DescribedKeypoints.load(kpts: try Fixtures.npy(d, p + "kpts"), feats: try Fixtures.npy(d, p + "feats"))))
        }
        let anchors = AnchorBuilder.build(keyframes: views, intrinsics: K)
        var agree = 0, total = 0
        for i in 0..<count {
            let want = try Fixtures.npy(d, String(format: "kf_%04d_has_pt3d", i)).bools()
            for k in 0..<want.count where views[i].keypoints.isValid(k) {
                total += 1
                if anchors.keyframes[i].hasPt3d[k] == want[k] { agree += 1 }
            }
        }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(total), 0.95)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter TriangulatorTests`
Expected: FAIL to compile (`cannot find 'Triangulator' in scope`).

- [ ] **Step 3: Implement `Triangulator.swift` and `AnchorBuilder.swift`**

`Sources/OSCDCore/Pose/Triangulator.swift`:
```swift
import simd

/// O-SCD poses/triangulator.py for one keyframe. For each keypoint, intersect its ray with the
/// matched ray in each other camera, keep the candidate with the most parallax ("disambiguation":
/// pixel distance in the other view between the ray's points at two depths), and accept it if that
/// parallax exceeds `minDisambiguation`.
public enum Triangulator {
    public struct Output: Sendable {
        public var pts3d: [SIMD3<Float>]          // world coordinates
        public var depth: [Float]                 // camera z in this keyframe
        public var bestDisambiguation: [Float]
        public var valid: [Bool]
    }

    /// - uvsOthers[c][i]: keypoint i's match in other camera c, or (-1, -1).
    /// - disambiguationDepths: O-SCD uses 1 and 10 world units. Our world is metric; Task 17 measures
    ///   whether that leaves enough anchors on a tabletop (spec deviation D11 if it doesn't).
    public static func run(uv: [SIMD2<Float>], uvsOthers: [[SIMD2<Float>]], pose: Pose, posesOthers: [Pose],
                           intrinsics K: Intrinsics, maxError: Float, minDisambiguation: Float,
                           disambiguationDepths: (Float, Float) = (1, 10)) -> Output {
        let n = uv.count
        var pts = [SIMD3<Float>](repeating: .zero, count: n)
        var best = [Float](repeating: 0, count: n)
        for (c, other) in posesOthers.enumerated() {
            let rel = pose.compose(other.inverse)                 // other camera → this camera
            let R = rel.R, p2 = rel.t, Rt = R.transpose
            let others = uvsOthers[c]
            for i in 0..<n {
                let o = others[i]
                guard min(o.x, o.y) > 0 else { continue }
                let d1 = simd_normalize(K.unproject(uv[i], depth: 1))
                let d2 = simd_normalize(R * K.unproject(o, depth: 1))
                let nrm = simd_cross(d1, d2), n2 = simd_cross(d2, nrm)
                let xyz = d1 * (simd_dot(n2, p2) / simd_dot(n2, d1))
                let q1 = Rt * (d1 * disambiguationDepths.0 - p2)
                let q2 = Rt * (d1 * disambiguationDepths.1 - p2)
                let dis = simd_length(K.project(q1) - K.project(q2))
                let err = simd_length(K.project(Rt * (xyz - p2)) - o)
                if xyz.z > 1e-6 && dis > best[i] && err < maxError {
                    pts[i] = xyz
                    best[i] = dis
                }
            }
        }
        let Rkf = pose.R.transpose
        return Output(pts3d: pts.map { Rkf * ($0 - pose.t) }, depth: pts.map(\.z),
                      bestDisambiguation: best, valid: best.map { $0 > minDisambiguation })
    }
}
```

`Sources/OSCDCore/Pose/AnchorBuilder.swift`:
```swift
import Foundation
import simd

/// One reference keyframe: XFeat features, stored F-RANSAC-inlier matches, triangulated anchors.
public struct KeyframeAnchors: Sendable {
    public let index: Int
    public let pose: Pose
    public let keypoints: DescribedKeypoints
    public var matches: [Int: MatchSet] = [:]          // other keyframe → (idx here, idx there)
    public var hasPt3d: [Bool]
    public var pts3d: [SIMD3<Float>]
    public var conf: [Float]

    public init(index: Int, pose: Pose, keypoints: DescribedKeypoints) {
        self.index = index
        self.pose = pose
        self.keypoints = keypoints
        hasPt3d = [Bool](repeating: false, count: keypoints.count)
        pts3d = [SIMD3<Float>](repeating: .zero, count: keypoints.count)
        conf = [Float](repeating: 0, count: keypoints.count)
    }
}

public struct AnchorSet: Sendable {
    public var keyframes: [KeyframeAnchors]
    public let intrinsics: Intrinsics
    public let maxError: Float
    public var anchorCount: Int
    /// Fraction of valid keypoints that received a 3D anchor.
    public var yield: Double
}

/// O-SCD reference preparation (oscd.py): exhaustive pairwise matching with F-RANSAC, then
/// Keyframe.update_3dpts for every keyframe.
public enum AnchorBuilder {
    /// O-SCD `max_error = max(match_max_error * width, 1.5)`.
    public static func maxError(width: Int, matchMaxError: Float = 2e-3) -> Float {
        max(matchMaxError * Float(width), 1.5)
    }

    public static func build(keyframes input: [(pose: Pose, keypoints: DescribedKeypoints)], intrinsics: Intrinsics,
                             triangulationCams: Int = 8, hypotheses: Int = 2000, seed: UInt64 = 0,
                             disambiguationDepths: (Float, Float) = (1, 10)) -> AnchorSet {
        var kfs = input.enumerated().map { KeyframeAnchors(index: $0.offset, pose: $0.element.pose, keypoints: $0.element.keypoints) }
        let maxErr = maxError(width: intrinsics.width)
        let snapshot = kfs
        let pairs = (0..<kfs.count).flatMap { i in ((i + 1)..<kfs.count).map { (i, $0) } }
        var results = [MatchSet](repeating: MatchSet(idx: [], idxOther: []), count: pairs.count)
        results.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: pairs.count) { p in
                let (i, j) = pairs[p]
                var sampler = SeededSampler(seed: seed &+ UInt64(p) &* 0x9E37_79B9_7F4A_7C15)
                out[p] = Matcher.matchFiltered(snapshot[i].keypoints, snapshot[j].keypoints,
                                               maxError: maxErr, hypotheses: hypotheses, sampler: &sampler)
            }
        }
        for (p, (i, j)) in pairs.enumerated() {
            kfs[i].matches[j] = results[p]
            kfs[j].matches[i] = MatchSet(idx: results[p].idxOther, idxOther: results[p].idx)
        }

        var anchors = 0, validKeypoints = 0
        for k in 0..<kfs.count {
            let kf = kfs[k]
            let chosen = kf.matches
                .sorted { $0.value.count != $1.value.count ? $0.value.count > $1.value.count : $0.key < $1.key }
                .prefix(triangulationCams)
            var uvs: [[SIMD2<Float>]] = [], poses: [Pose] = []
            for (other, m) in chosen {
                var row = [SIMD2<Float>](repeating: SIMD2(-1, -1), count: kf.keypoints.count)
                for (a, b) in zip(m.idx, m.idxOther) { row[a] = kfs[other].keypoints.kpts[b] }
                uvs.append(row)
                poses.append(kfs[other].pose)
            }
            let out = Triangulator.run(uv: kf.keypoints.kpts, uvsOthers: uvs, pose: kf.pose, posesOthers: poses,
                                       intrinsics: intrinsics, maxError: maxErr, minDisambiguation: 30 * maxErr,
                                       disambiguationDepths: disambiguationDepths)
            for i in 0..<kf.keypoints.count where out.valid[i] {
                kfs[k].hasPt3d[i] = true
                kfs[k].pts3d[i] = out.pts3d[i]
                kfs[k].conf[i] = 1
                anchors += 1
            }
            validKeypoints += (0..<kf.keypoints.count).filter { kf.keypoints.isValid($0) }.count
        }
        return AnchorSet(keyframes: kfs, intrinsics: intrinsics, maxError: maxErr, anchorCount: anchors,
                         yield: validKeypoints > 0 ? Double(anchors) / Double(validKeypoints) : 0)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter TriangulatorTests` then `swift test -c release --filter TriangulatorTests.testParity`
Expected: PASS. Record how long `testParityAnchorBuilder` takes on the M4 (it builds anchors for every PASLCD reference image). This is the Mac baseline for Task 17.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD triangulation and reference anchor building

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: `PoseEstimator`: the full per-frame O-SCD pose

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Pose/PoseEstimator.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/PoseEstimatorTests.swift`

**Interfaces:**
- Consumes:
  - `AnchorSet`, `KeyframeAnchors` (Task 10)
  - `Matcher.countMutual`, `Matcher.matchFiltered`, `SeededSampler.subset` (Task 7)
  - `P4PRANSAC.run` (Task 9)
  - `MiniBA.solve`, `MiniBAConfig.poseRefine` (Task 8)
- Produces:
  - `PoseEstimate` with `pose: Pose?`, `accepted`, `retrieved`, `nCorrespondences`, `nPnPInliers`, `nBAInlierComponents`, `failReason: String?`. The fail reasons are `"no_reference"`, `"too_few_correspondences"` and `"too_few_inliers"`.
  - `PoseEstimator(anchors:seed:)`, with mutable settings `numReference` (4), `pnpHypotheses` (2000), `numPtsMiniBA` (2000) and `minNumInliers` (100).
  - `estimate(_ frame: DescribedKeypoints) -> PoseEstimate`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/PoseEstimatorTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class PoseEstimatorTests: XCTestCase {
    func loadFixtureAnchors() throws -> AnchorSet {
        let d = try Fixtures.dir("anchors")
        let info = try Fixtures.json(d.appendingPathComponent("info.json"))
        let K = Intrinsics(f: Float(info["f"] as! Double), cx: Float(info["cx"] as! Double), cy: Float(info["cy"] as! Double),
                           width: info["width"] as! Int, height: info["height"] as! Int)
        var kfs: [KeyframeAnchors] = []
        for i in 0..<(info["count"] as! Int) {
            let p = String(format: "kf_%04d_", i)
            var kf = KeyframeAnchors(index: i, pose: Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, p + "Rt").floats()),
                                     keypoints: DescribedKeypoints.load(kpts: try Fixtures.npy(d, p + "kpts"), feats: try Fixtures.npy(d, p + "feats")))
            kf.hasPt3d = try Fixtures.npy(d, p + "has_pt3d").bools()
            kf.pts3d = Fixtures.vec3s(try Fixtures.npy(d, p + "pts3d"))
            kf.conf = kf.hasPt3d.map { $0 ? 1 : 0 }
            kfs.append(kf)
        }
        return AnchorSet(keyframes: kfs, intrinsics: K, maxError: Float(info["max_error"] as! Double),
                         anchorCount: kfs.reduce(0) { $0 + $1.hasPt3d.filter { $0 }.count }, yield: 0)
    }

    /// Review Focus 1: a frame with almost no features is rejected with a reason.
    func testFrameWithTooFewFeaturesIsRejectedCleanly() {
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var kf = KeyframeAnchors(index: 0, pose: .identity, keypoints: DescribedKeypoints(kpts: [SIMD2(10, 10)], feats: [1] + [Float](repeating: 0, count: 63)))
        kf.hasPt3d = [true]; kf.pts3d = [SIMD3(0, 0, 1)]; kf.conf = [1]
        let est = PoseEstimator(anchors: AnchorSet(keyframes: [kf], intrinsics: K, maxError: 1.5, anchorCount: 1, yield: 1))
        let r = est.estimate(DescribedKeypoints(kpts: [SIMD2(10, 10)], feats: [1] + [Float](repeating: 0, count: 63)))
        XCTAssertFalse(r.accepted)
        XCTAssertNil(r.pose)
        XCTAssertEqual(r.failReason, "too_few_correspondences")
    }

    /// End to end against O-SCD's own init_inference_pose (its own randomness, so poses are
    /// compared with tolerances rather than bit-exactly).
    func testParityPoseEstimator() throws {
        let anchors = try loadFixtureAnchors()
        let centres = anchors.keyframes.map { $0.pose.centre }
        let mean = centres.reduce(.zero, +) / Float(centres.count)
        let sceneScale = centres.map { simd_distance($0, mean) }.sorted()[centres.count / 2]
        let estimator = PoseEstimator(anchors: anchors)
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let frame = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
            let r = estimator.estimate(frame)
            XCTAssertEqual(Set(r.retrieved), Set(try Fixtures.npy(d, "retrieved").ints()), "frame \(k)")
            let wantAccepted = try Fixtures.npy(d, "oscd_accepted").bools()[0]
            XCTAssertEqual(r.accepted, wantAccepted, "frame \(k)")
            guard wantAccepted, let pose = r.pose else { continue }
            let want = Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, "oscd_Rt").floats())
            XCTAssertLessThan(pose.rotationDegrees(to: want), 0.1, "frame \(k)")
            XCTAssertLessThan(simd_distance(pose.centre, want.centre), 0.005 * sceneScale, "frame \(k)")
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter PoseEstimatorTests`
Expected: FAIL to compile (`cannot find 'PoseEstimator' in scope`).

- [ ] **Step 3: Implement `PoseEstimator.swift`**

```swift
import Foundation
import simd

public struct PoseEstimate: Sendable {
    public var pose: Pose?
    public var accepted = false
    public var retrieved: [Int] = []
    public var nCorrespondences = 0
    public var nPnPInliers = 0
    /// O-SCD n_ba_inliers: mask components, 2 per point.
    public var nBAInlierComponents = 0
    public var failReason: String?
}

/// O-SCD PoseInitializer.init_inference_pose + get_reference_keyframes, with P1 semantics:
/// a rejected frame returns a reason instead of crashing.
public final class PoseEstimator {
    public let anchors: AnchorSet
    public var numReference = 4
    public var pnpHypotheses = 2000
    public var numPtsMiniBA = 2000
    public var minNumInliers = 100          // compared with the component count, as O-SCD does
    private var rng: SeededSampler

    public init(anchors: AnchorSet, seed: UInt64 = 0) {
        self.anchors = anchors
        rng = SeededSampler(seed: seed)
    }

    public func estimate(_ frame: DescribedKeypoints) -> PoseEstimate {
        var est = PoseEstimate()
        let kfs = anchors.keyframes

        // 1. Retrieval: the keyframes with the most mutual-NN matches (no geometric check).
        var counts = [Int](repeating: 0, count: kfs.count)
        counts.withUnsafeMutableBufferPointer { c in
            DispatchQueue.concurrentPerform(iterations: kfs.count) { k in c[k] = Matcher.countMutual(kfs[k].keypoints, frame) }
        }
        est.retrieved = Array(counts.indices.sorted { counts[$0] != counts[$1] ? counts[$0] > counts[$1] : $0 < $1 }.prefix(numReference))
        guard let top = est.retrieved.first else { est.failReason = "no_reference"; return est }

        // 2. 2D–3D correspondences through each retrieved keyframe's anchors.
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = [], conf: [Double] = []
        for k in est.retrieved {
            let kf = kfs[k]
            let m = Matcher.matchFiltered(frame, kf.keypoints, maxError: anchors.maxError, hypotheses: 2000, sampler: &rng)
            for (a, b) in zip(m.idx, m.idxOther) where kf.hasPt3d[b] {
                uv.append(SIMD2<Double>(frame.kpts[a]))
                xyz.append(SIMD3<Double>(kf.pts3d[b]))
                conf.append(Double(kf.conf[b]))
            }
        }
        est.nCorrespondences = uv.count
        if uv.count > 2 * numPtsMiniBA {
            let sel = rng.subset(numPtsMiniBA, of: uv.count)
            uv = sel.map { uv[$0] }; xyz = sel.map { xyz[$0] }; conf = sel.map { conf[$0] }
        }

        // 3. P4P-RANSAC started from the top-retrieved keyframe's pose.
        let K = anchors.intrinsics
        let f = Double(K.f), c = SIMD2<Double>(K.centre)
        guard let ransac = P4PRANSAC.run(uv: uv, xyz: xyz, confs: conf, f: f, centre: c,
                                         initR6: kfs[top].pose.r6, initT: kfs[top].pose.tDouble,
                                         hypotheses: pnpHypotheses, maxError: Double(anchors.maxError), sampler: &rng) else {
            est.failReason = "too_few_correspondences"
            return est
        }
        var uvIn: [SIMD2<Double>] = [], xyzIn: [SIMD3<Double>] = []
        for k in 0..<uv.count where ransac.inliers[k] { uvIn.append(uv[k]); xyzIn.append(xyz[k]) }
        est.nPnPInliers = uvIn.count

        // 4. Mini-BA on exactly numPtsMiniBA points: a random subset, or padded with masked dummies.
        if uvIn.count >= numPtsMiniBA {
            let sel = rng.subset(numPtsMiniBA, of: uvIn.count)
            uvIn = sel.map { uvIn[$0] }; xyzIn = sel.map { xyzIn[$0] }
        } else {
            let pad = numPtsMiniBA - uvIn.count
            uvIn += [SIMD2<Double>](repeating: SIMD2(-1, -1), count: pad)
            xyzIn += [SIMD3<Double>](repeating: .zero, count: pad)
        }
        let ba = MiniBA.solve(r6: ransac.r6, t: ransac.t, f: f, centre: c, xyz: xyzIn, uv: uvIn, config: .poseRefine)
        est.nBAInlierComponents = ba.inlierComponents

        // 5. O-SCD gate: more than min_num_inliers residual components (> 50 points).
        if ba.inlierComponents > minNumInliers {
            est.pose = Pose(r6: ba.r6, t: ba.t)
            est.accepted = true
        } else {
            est.failReason = "too_few_inliers"
        }
        return est
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter PoseEstimatorTests` then `swift test -c release --filter PoseEstimatorTests.testParity`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Add PoseEstimator: O-SCD per-frame pose with clean rejection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 12: Method A cues: SSIM pixel cue and SAM2 feature cue

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Cues/SSIM.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Cues/CuesA.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/CuesATests.swift`

**Interfaces:**
- Consumes: `FloatImage` (Task 3).
- Produces:
  - `SSIM.window: [Float]`, `SSIM.meanMap(_ a: FloatImage, _ b: FloatImage) -> [Float]`.
  - `CuesA.pixel(live:render:valid:) -> (normalized: [Float], raw: [Float], vmin: Float, vmax: Float)`.
  - `CuesA.feature(embRender:embLive:channels:grid:width:height:vmin:vmax:valid:) -> [Float]`.
  - `CuesA.compute(live:render:embRender:embLive:valid:) -> CueMapsA`, where `CueMapsA` has `combined`, `pixel`, `feature`, `rawMin`, `rawMax`.
  - `valid == nil` reproduces O-SCD exactly. A mask applies deviation D3: min/max are taken over valid pixels (and over feature cells that contain one), and invalid pixels get C = 0.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/CuesATests.swift`:
```swift
import XCTest
@testable import OSCDCore

final class CuesATests: XCTestCase {
    func constant(_ v: Float, _ w: Int = 32, _ h: Int = 24) -> FloatImage { FloatImage(width: w, height: h, channels: 3, fill: v) }

    func testSSIMIdenticalIsOneAndConstantsMatchFormula() {
        XCTAssertTrue(SSIM.meanMap(constant(0.3), constant(0.3)).allSatisfy { abs($0 - 1) < 1e-5 })
        // Interior pixel, constants 0.2 vs 0.6: (2·0.12 + C1) / (0.04 + 0.36 + C1).
        let m = SSIM.meanMap(constant(0.2), constant(0.6))
        XCTAssertEqual(m[12 * 32 + 16], Float((0.24 + 1e-4) / (0.40 + 1e-4)), accuracy: 1e-4)
    }

    func testIdenticalImagesGiveZeroCue() {
        let p = CuesA.pixel(live: constant(0.4), render: constant(0.4))
        XCTAssertTrue(p.normalized.allSatisfy { $0 == 0 })
    }

    /// Review Focus 3: an all-invalid mask returns zeros everywhere, with no NaN.
    func testAllInvalidMaskGivesZeros() {
        let n = 32 * 24
        let emb = [Float](repeating: 0.1, count: 256 * 64 * 64)
        let c = CuesA.compute(live: constant(0.1), render: constant(0.9), embRender: emb, embLive: emb.map { $0 + 0.5 },
                              valid: [Bool](repeating: false, count: n))
        XCTAssertTrue(c.combined.allSatisfy { $0 == 0 })
        XCTAssertFalse(c.combined.contains { $0.isNaN })
    }

    func testParityCuesA() throws {
        _ = try Fixtures.dir("cues")
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "cues/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let liveA = try Fixtures.npy(d, "live"), renA = try Fixtures.npy(d, "render")
            let H = liveA.shape[1], W = liveA.shape[2]
            let live = FloatImage(width: W, height: H, channels: 3, data: liveA.floats())
            let render = FloatImage(width: W, height: H, channels: 3, data: renA.floats())
            let ssim = SSIM.meanMap(render, live), wantSSIM = try Fixtures.npy(d, "ssim_map").floats()
            XCTAssertLessThan(zip(ssim, wantSSIM).map { abs($0 - $1) }.max()!, 1e-3, "frame \(k) ssim")
            let p = CuesA.pixel(live: live, render: render)
            XCTAssertLessThan(zip(p.raw, try Fixtures.npy(d, "delta_raw").floats()).map { abs($0 - $1) }.max()!, 1e-3, "frame \(k) delta")
            let emb = try Fixtures.npy(d, "emb").floats(), half = emb.count / 2
            let c = CuesA.compute(live: live, render: render, embRender: Array(emb[..<half]), embLive: Array(emb[half...]))
            XCTAssertLessThan(zip(c.combined, try Fixtures.npy(d, "C").floats()).map { abs($0 - $1) }.max()!, 2e-3, "frame \(k) C")
            // An all-true mask must reproduce O-SCD exactly.
            let masked = CuesA.compute(live: live, render: render, embRender: Array(emb[..<half]), embLive: Array(emb[half...]),
                                       valid: [Bool](repeating: true, count: W * H))
            XCTAssertEqual(masked.combined, c.combined)
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter CuesATests`
Expected: FAIL to compile (`cannot find 'SSIM' in scope`).

- [ ] **Step 3: Implement `SSIM.swift` and `CuesA.swift`**

`Sources/OSCDCore/Cues/SSIM.swift`:
```swift
import Foundation

/// O-SCD utils/loss_utils.py `ssim(..., window_size=11, map=True)`.
public enum SSIM {
    /// Normalised 1-D Gaussian, σ = 1.5, 11 taps (the 2-D window is its outer product).
    public static let window: [Float] = {
        let g = (0..<11).map { exp(-Double(($0 - 5) * ($0 - 5)) / (2 * 1.5 * 1.5)) }
        let s = g.reduce(0, +)
        return g.map { Float($0 / s) }
    }()

    /// Zero-padded separable Gaussian filter of one plane (F.conv2d with padding = 5).
    static func blur(_ x: [Float], _ w: Int, _ h: Int) -> [Float] {
        let k = window, r = 5
        var tmp = [Float](repeating: 0, count: w * h), out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x0 in 0..<w {
                var s: Float = 0
                for i in -r...r { let xx = x0 + i; if xx >= 0 && xx < w { s += k[i + r] * x[y * w + xx] } }
                tmp[y * w + x0] = s
            }
        }
        for y0 in 0..<h {
            for x in 0..<w {
                var s: Float = 0
                for i in -r...r { let yy = y0 + i; if yy >= 0 && yy < h { s += k[i + r] * tmp[yy * w + x] } }
                out[y0 * w + x] = s
            }
        }
        return out
    }

    /// Per-pixel SSIM averaged over channels (O-SCD `ssim(img1, img2, map=True).mean(dim=0)`).
    public static func meanMap(_ a: FloatImage, _ b: FloatImage) -> [Float] {
        let w = a.width, h = a.height, n = w * h
        let C1: Float = 0.01 * 0.01, C2: Float = 0.03 * 0.03
        var acc = [Float](repeating: 0, count: n)
        for c in 0..<a.channels {
            let x = Array(a.plane(c)), y = Array(b.plane(c))
            let mu1 = blur(x, w, h), mu2 = blur(y, w, h)
            let xx = blur(zip(x, x).map(*), w, h), yy = blur(zip(y, y).map(*), w, h), xy = blur(zip(x, y).map(*), w, h)
            for i in 0..<n {
                let m1 = mu1[i], m2 = mu2[i]
                let s1 = xx[i] - m1 * m1, s2 = yy[i] - m2 * m2, s12 = xy[i] - m1 * m2
                acc[i] += ((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 * m1 + m2 * m2 + C1) * (s1 + s2 + C2))
            }
        }
        return acc.map { $0 / Float(a.channels) }
    }
}
```

`Sources/OSCDCore/Cues/CuesA.swift`:
```swift
import Foundation

public struct CueMapsA: Sendable {
    public var combined: [Float]      // C_A in [0, 2]; 0 at invalid pixels
    public var pixel: [Float]         // normalised pixel cue
    public var feature: [Float]       // feature cue after O-SCD's re-normalisation, in [0, 1]
    public var rawMin: Float
    public var rawMax: Float
}

/// O-SCD oscd.py `generate_candidate_map`.
public enum CuesA {
    /// Pixel half: 0.8·mean_c|render − live| + 0.2·(1 − SSIM), min-max normalised.
    public static func pixel(live: FloatImage, render: FloatImage, valid: [Bool]? = nil)
        -> (normalized: [Float], raw: [Float], vmin: Float, vmax: Float) {
        let n = live.width * live.height
        let ssim = SSIM.meanMap(render, live)
        var raw = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var l1: Float = 0
            for c in 0..<live.channels { l1 += abs(render.data[c * n + i] - live.data[c * n + i]) }
            raw[i] = 0.8 * (l1 / Float(live.channels)) + 0.2 * (1 - ssim[i])
        }
        var vmin = Float.infinity, vmax = -Float.infinity
        for i in 0..<n where valid?[i] ?? true {
            vmin = min(vmin, raw[i])
            vmax = max(vmax, raw[i])
        }
        if vmin > vmax { return ([Float](repeating: 0, count: n), raw, 0, 0) }
        return (raw.map { ($0 - vmin) / (vmax - vmin + 1e-8) }, raw, vmin, vmax)
    }

    /// Feature half from SAM2 embeddings [channels, grid, grid] of the render and the live image:
    /// channel-mean |diff| → min-max → bilinear upsample → re-normalise with the pixel cue's raw
    /// min/max → clamp to [0, 1] (O-SCD code quirk, reproduced).
    public static func feature(embRender: [Float], embLive: [Float], channels: Int = 256, grid: Int = 64,
                               width: Int, height: Int, vmin: Float, vmax: Float, valid: [Bool]? = nil) -> [Float] {
        let cells = grid * grid
        var diff = [Float](repeating: 0, count: cells)
        for c in 0..<channels {
            for i in 0..<cells { diff[i] += abs(embRender[c * cells + i] - embLive[c * cells + i]) }
        }
        for i in 0..<cells { diff[i] /= Float(channels) }
        var cellValid = [Bool](repeating: valid == nil, count: cells)
        if let valid {
            for y in 0..<height {
                for x in 0..<width where valid[y * width + x] { cellValid[(y * grid / height) * grid + x * grid / width] = true }
            }
        }
        var dmin = Float.infinity, dmax = -Float.infinity
        for i in 0..<cells where cellValid[i] {
            dmin = min(dmin, diff[i])
            dmax = max(dmax, diff[i])
        }
        if dmin > dmax { return [Float](repeating: 0, count: width * height) }
        let norm = FloatImage(width: grid, height: grid, channels: 1, data: diff.map { ($0 - dmin) / (dmax - dmin + 1e-8) })
        return norm.resizedBilinear(width: width, height: height).data.map { min(1, max(0, ($0 - vmin) / (vmax - vmin + 1e-8))) }
    }

    public static func compute(live: FloatImage, render: FloatImage, embRender: [Float], embLive: [Float],
                               valid: [Bool]? = nil) -> CueMapsA {
        let p = pixel(live: live, render: render, valid: valid)
        let f = feature(embRender: embRender, embLive: embLive, width: live.width, height: live.height,
                        vmin: p.vmin, vmax: p.vmax, valid: valid)
        var combined = zip(p.normalized, f).map(+)
        if let valid { for i in combined.indices where !valid[i] { combined[i] = 0 } }
        return CueMapsA(combined: combined, pixel: p.normalized, feature: f, rawMin: p.vmin, rawMax: p.vmax)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter CuesATests` then `swift test -c release --filter CuesATests.testParity`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Port O-SCD's pixel and feature change cues with the D3 validity mask

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: SAM2.1-tiny encoder in Core ML

**Files:**
- Create: `oscd_mobile/tools/convert_sam2_coreml.py`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Cues/SAM2Encoder.swift`
- Modify: `oscd_mobile/README.md` (add the SAM2 line under "Models")
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/SAM2Tests.swift`

**Interfaces:**
- Consumes: `FloatImage.resizedNearest` (Task 3), `MLMultiArray.denseFloats`, `ModelError` (Task 6), `CuesA.compute` (Task 12).
- Produces:
  - `Models/SAM2_1Tiny_2x1024.mlpackage`, with input `images` Float32 [2,3,1024,1024] (raw [0,1]) and output `embedding` Float32 [2,256,64,64].
  - `SAM2Encoder(compiledModelURL:computeUnits:)` and `encode(render:live:) throws -> (render: [Float], live: [Float])`.

- [ ] **Step 1: Write `convert_sam2_coreml.py`**

```python
#!/usr/bin/env python
"""Convert SAM2.1-hiera-tiny's image encoder to Core ML exactly as O-SCD calls it.

O-SCD feeds raw [0, 1] images (no ImageNet normalisation), nearest-resized to 1024×1024, as one
batch [render, live], and keeps get_image_embeddings(...)[-1] (last FPN level plus the
no-memory embedding). The model below has the same input and output.
"""
import argparse
import os

import coremltools as ct
import numpy as np
import torch
from transformers import Sam2Model


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.get_image_embeddings(images)[-1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "..", "Models"))
    ap.add_argument("--fixtures", default=os.path.join(here, "..", "fixtures"))
    a = ap.parse_args()

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval()
    enc = Encoder(model).eval()
    example = torch.rand(2, 3, 1024, 1024)
    with torch.no_grad():
        assert tuple(enc(example).shape) == (2, 256, 64, 64)
        traced = torch.jit.trace(enc, example, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(2, 3, 1024, 1024), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "SAM2_1Tiny_2x1024.mlpackage")
    ml.save(path)
    print("saved", path)

    x_path = os.path.join(a.fixtures, "cues", "frame_000", "sam_input.npy")
    if not os.path.exists(x_path):
        print("no fixtures; skipped parity check")
        return
    x = np.load(x_path).astype(np.float32)
    got = ml.predict({"images": x})["embedding"]
    want = np.load(os.path.join(a.fixtures, "cues", "frame_000", "emb.npy"))
    d_got = np.abs(got[0] - got[1]).mean(axis=0).ravel()
    d_want = np.abs(want[0] - want[1]).mean(axis=0).ravel()
    r_emb = np.corrcoef(got.ravel(), want.ravel())[0, 1]
    r_diff = np.corrcoef(d_got, d_want)[0, 1]
    print(f"embedding r = {r_emb:.4f}, feature-difference map r = {r_diff:.4f}")
    assert r_diff >= 0.98, "feature-difference map differs from O-SCD"
    print("parity OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the conversion**

Run: `oscd_mobile/tools/.venv/bin/python oscd_mobile/tools/convert_sam2_coreml.py`
Expected: `saved .../Models/SAM2_1Tiny_2x1024.mlpackage`, then `feature-difference map r = 0.98x` (or higher) and `parity OK` when fixtures exist.
If tracing or conversion fails, **stop and report the error to the human.** Don't substitute Apple's `apple/coreml-sam2.1-tiny` package. Its input preprocessing isn't documented, so it can't be shown to match O-SCD's raw-[0,1] input (spec D4).

- [ ] **Step 3: Write the failing Swift test**

`Tests/OSCDCoreTests/SAM2Tests.swift`:
```swift
import XCTest
@testable import OSCDCore

final class SAM2Tests: XCTestCase {
    func testParitySAM2EncoderFeedsTheSameCue() throws {
        let d = try Fixtures.dir("cues/frame_000")
        let pkg = try Fixtures.modelPackage("SAM2_1Tiny_2x1024.mlpackage")
        let encoder = try SAM2Encoder(compiledModelURL: try XFeatModel.compile(pkg))
        let liveA = try Fixtures.npy(d, "live"), renA = try Fixtures.npy(d, "render")
        let H = liveA.shape[1], W = liveA.shape[2]
        let live = FloatImage(width: W, height: H, channels: 3, data: liveA.floats())
        let render = FloatImage(width: W, height: H, channels: 3, data: renA.floats())
        let emb = try encoder.encode(render: render, live: live)
        let ours = CuesA.compute(live: live, render: render, embRender: emb.render, embLive: emb.live)
        let want = try Fixtures.npy(d, "C").floats()
        let mae = zip(ours.combined, want).map { abs($0 - $1) }.reduce(0, +) / Float(want.count)
        XCTAssertLessThan(mae, 0.02)
        let ref = try Fixtures.npy(d, "emb").floats(), half = ref.count / 2
        let refCue = CuesA.compute(live: live, render: render, embRender: Array(ref[..<half]), embLive: Array(ref[half...]))
        XCTAssertGreaterThanOrEqual(Self.pearson(ours.feature, refCue.feature), 0.98)
    }

    static func pearson(_ a: [Float], _ b: [Float]) -> Double {
        let n = Double(a.count)
        let ma = a.reduce(0) { $0 + Double($1) } / n, mb = b.reduce(0) { $0 + Double($1) } / n
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for (x, y) in zip(a, b) { let dx = Double(x) - ma, dy = Double(y) - mb; sab += dx * dy; saa += dx * dx; sbb += dy * dy }
        return sab / (saa * sbb).squareRoot()
    }
}
```

Run: `swift test --filter SAM2Tests`
Expected: FAIL to compile (`cannot find 'SAM2Encoder' in scope`).

- [ ] **Step 4: Implement `SAM2Encoder.swift`**

```swift
import CoreML

/// SAM2.1-hiera-tiny image encoder as O-SCD uses it: both images nearest-resized to 1024×1024,
/// raw [0, 1] values, one batch [render, live] → embeddings [2, 256, 64, 64].
public final class SAM2Encoder {
    public static let side = 1024
    public let model: MLModel

    public init(compiledModelURL url: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    public func encode(render: FloatImage, live: FloatImage) throws -> (render: [Float], live: [Float]) {
        let a = render.resizedNearest(width: Self.side, height: Self.side)
        let b = live.resizedNearest(width: Self.side, height: Self.side)
        let arr = try MLMultiArray(shape: [2, 3, NSNumber(value: Self.side), NSNumber(value: Self.side)], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        a.data.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: $0.count) }
        b.data.withUnsafeBufferPointer { (dst + a.data.count).update(from: $0.baseAddress!, count: $0.count) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["images": MLFeatureValue(multiArray: arr)]))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else { throw ModelError.missingOutput("embedding") }
        let all = e.denseFloats()
        let half = all.count / 2
        return (Array(all[..<half]), Array(all[half...]))
    }
}
```

- [ ] **Step 5: Run the test, document, commit**

Run: `swift test -c release --filter SAM2Tests`
Expected: PASS (skips without fixtures or the model).

Append under "Models" in `oscd_mobile/README.md`:
```markdown
    tools/.venv/bin/python tools/convert_sam2_coreml.py      # → Models/SAM2_1Tiny_2x1024.mlpackage
```

```bash
git add oscd_mobile/tools/convert_sam2_coreml.py oscd_mobile/Packages/OSCDCore oscd_mobile/README.md
git commit -m "Convert SAM2.1-tiny to Core ML with O-SCD's raw input and add the encoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Method B cues: LiDAR depth and color

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Cues/CuesB.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/CuesBTests.swift`

**Interfaces:**
- Consumes: `FloatImage` (Task 3).
- Produces:
  - `CuesB.Params` holds every constant from the spec's §8.
  - `CuesB.upsampleDepth(_:confidence:width:height:toWidth:toHeight:) -> [Float]`. Invalid pixels come back as NaN.
  - `CuesB.depthCue(live:rendered:valid:params:) -> (cue: [Float], sign: [Int8])`, where sign is +1 for added and −1 for removed.
  - `CuesB.colorCue(live:render:liveDepth:renderDepth:valid:params:) -> [Float]`.
  - `CuesB.combine(cA:depth:color:valid:params:) -> [Float]`.
  - `CuesB.srgbToLab(_:_:_:) -> SIMD3<Float>`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/CuesBTests.swift`:
```swift
import XCTest
@testable import OSCDCore

final class CuesBTests: XCTestCase {
    func image(_ rgb: SIMD3<Float>, _ w: Int, _ h: Int) -> FloatImage {
        FloatImage(width: w, height: h, channels: 3, data: [Float](repeating: rgb.x, count: w * h) + [Float](repeating: rgb.y, count: w * h) + [Float](repeating: rgb.z, count: w * h))
    }

    func testLabReferenceValues() {
        let red = CuesB.srgbToLab(1, 0, 0)
        XCTAssertEqual(red.x, 53.24, accuracy: 0.05); XCTAssertEqual(red.y, 80.09, accuracy: 0.05); XCTAssertEqual(red.z, 67.20, accuracy: 0.05)
        let white = CuesB.srgbToLab(1, 1, 1)
        XCTAssertEqual(white.x, 100, accuracy: 0.05); XCTAssertEqual(white.y, 0, accuracy: 0.05)
    }

    func testDepthCueRampsAndSigns() {
        let r = CuesB.depthCue(live: [0.50, 0.47, 0.515, .nan], rendered: [0.5, 0.5, 0.5, 0.5], valid: [true, true, true, true])
        XCTAssertEqual(r.cue[0], 0)
        XCTAssertEqual(r.cue[1], 1, accuracy: 1e-5); XCTAssertEqual(r.sign[1], 1)     // 3 cm closer: added
        XCTAssertEqual(r.cue[2], 0.25, accuracy: 1e-4); XCTAssertEqual(r.sign[2], -1)  // 1.5 cm farther: removed
        XCTAssertEqual(r.cue[3], 0)                                                     // Review Focus 4: NaN depth is ignored
    }

    /// Review Focus 4: low-confidence or zero LiDAR depth becomes NaN, never a 0 m surface.
    func testUpsampleDepthDropsLowConfidence() {
        let d = CuesB.upsampleDepth([0.5, 0.0, 0.7, 0.8], confidence: [2, 2, 1, 2], width: 2, height: 2, toWidth: 4, toHeight: 4)
        XCTAssertEqual(d[0], 0.5)
        XCTAssertTrue(d[2].isNaN)           // depth 0
        XCTAssertTrue(d[2 * 4 + 0].isNaN)   // confidence 1
        XCTAssertEqual(d[3 * 4 + 3], 0.8)
    }

    func testColorCueFiresOnRecolorButNotOnLightnessOrExposure() {
        let w = 16, h = 16, n = w * h, valid = [Bool](repeating: true, count: n), nanDepth = [Float](repeating: .nan, count: n)
        let recolor = CuesB.colorCue(live: image(SIMD3(0, 0, 1), w, h), render: image(SIMD3(1, 0, 0), w, h),
                                     liveDepth: nanDepth, renderDepth: nanDepth, valid: valid)
        XCTAssertGreaterThan(recolor.min()!, 0.99)
        let lightness = CuesB.colorCue(live: image(SIMD3(0.6, 0.6, 0.6), w, h), render: image(SIMD3(0.9, 0.9, 0.9), w, h),
                                       liveDepth: nanDepth, renderDepth: nanDepth, valid: valid)
        XCTAssertLessThan(lightness.max()!, 0.05)
        // Exposure ×1.3 on 32×32 (≥ 500 agreeing-depth pixels): the gain alignment removes it.
        let W = 32, N = W * W, depth = [Float](repeating: 0.6, count: N)
        let exposure = CuesB.colorCue(live: image(SIMD3(0.52, 0.39, 0.26), W, W), render: image(SIMD3(0.4, 0.3, 0.2), W, W),
                                      liveDepth: depth, renderDepth: depth, valid: [Bool](repeating: true, count: N))
        XCTAssertLessThan(exposure.max()!, 0.05)
    }

    /// Review Focus 3: nothing valid → zeros.
    func testCombineClampsAndZeroesInvalid() {
        let c = CuesB.combine(cA: [1.5, 0.2, 1.0], depth: [1, 0, 1], color: [1, 0, 0], valid: [true, true, false])
        XCTAssertEqual(c, [2, 0.2, 0])
        let none = CuesB.colorCue(live: image(SIMD3(1, 0, 0), 4, 4), render: image(SIMD3(0, 0, 1), 4, 4),
                                  liveDepth: [Float](repeating: .nan, count: 16), renderDepth: [Float](repeating: .nan, count: 16),
                                  valid: [Bool](repeating: false, count: 16))
        XCTAssertEqual(none, [Float](repeating: 0, count: 16))
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter CuesBTests`
Expected: FAIL to compile (`cannot find 'CuesB' in scope`).

- [ ] **Step 3: Implement `CuesB.swift`**

```swift
import Foundation
import simd

/// Method B additions (spec §8): LiDAR depth cue and exposure-aligned, chroma-weighted color cue.
public enum CuesB {
    public struct Params: Sendable {
        public var depthMinTol: Float = 0.01
        public var depthRelTol: Float = 0.02
        public var depthRamp: Float = 0.02
        public var gainAgreeDepth: Float = 0.01
        public var gainMinPixels = 500
        public var blurSigma: Float = 1
        public var lightnessWeight: Float = 0.25
        public var deltaE0: Float = 10
        public var deltaE1: Float = 25
        public var minL: Float = 15
        public var maxL: Float = 95
        public var lambdaDepth: Float = 1
        public var lambdaColor: Float = 1
        public init() {}
    }

    /// LiDAR depth (metres) to working resolution by nearest neighbour. Pixels with ARKit confidence
    /// below high (2), or a non-finite or non-positive depth, become NaN.
    public static func upsampleDepth(_ d: [Float], confidence: [UInt8], width: Int, height: Int,
                                     toWidth: Int, toHeight: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: toWidth * toHeight)
        for y in 0..<toHeight {
            let sy = min(y * height / toHeight, height - 1)
            for x in 0..<toWidth {
                let i = sy * width + min(x * width / toWidth, width - 1)
                if confidence[i] >= 2, d[i].isFinite, d[i] > 0 { out[y * toWidth + x] = d[i] }
            }
        }
        return out
    }

    /// C_depth = clamp((|D_live − D_ren| − τ) / ramp, 0, 1) with τ = max(1 cm, 2%·D_ren).
    public static func depthCue(live: [Float], rendered: [Float], valid: [Bool],
                                params p: Params = Params()) -> (cue: [Float], sign: [Int8]) {
        var cue = [Float](repeating: 0, count: live.count), sign = [Int8](repeating: 0, count: live.count)
        for i in 0..<live.count where valid[i] && live[i].isFinite && rendered[i].isFinite && rendered[i] > 0 {
            let tau = max(p.depthMinTol, p.depthRelTol * rendered[i])
            let d = live[i] - rendered[i]
            cue[i] = min(1, max(0, (abs(d) - tau) / p.depthRamp))
            if cue[i] > 0 { sign[i] = d < 0 ? 1 : -1 }
        }
        return (cue, sign)
    }

    public static func colorCue(live: FloatImage, render: FloatImage, liveDepth: [Float], renderDepth: [Float],
                                valid: [Bool], params p: Params = Params()) -> [Float] {
        let n = live.width * live.height
        // 1. Per-channel gain aligning the render to the live frame, from pixels where depth agrees.
        var gain: [Float] = [1, 1, 1]
        var ratios: [[Float]] = [[], [], []]
        for i in 0..<n where valid[i] && liveDepth[i].isFinite && renderDepth[i].isFinite
                             && abs(liveDepth[i] - renderDepth[i]) < p.gainAgreeDepth {
            for c in 0..<3 where render.data[c * n + i] > 1e-3 { ratios[c].append(live.data[c * n + i] / render.data[c * n + i]) }
        }
        for c in 0..<3 where ratios[c].count >= p.gainMinPixels {
            let s = ratios[c].sorted()
            gain[c] = s[s.count / 2]
        }
        var adjusted = render
        for c in 0..<3 { for i in 0..<n { adjusted.data[c * n + i] = min(1, render.data[c * n + i] * gain[c]) } }
        // 2. Light blur. 3. CIELAB difference with lightness down-weighted. 4. Absolute ramp. 5. L* masks.
        let a = gaussianBlur(live, sigma: p.blurSigma), b = gaussianBlur(adjusted, sigma: p.blurSigma)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n where valid[i] {
            let la = srgbToLab(a.data[i], a.data[n + i], a.data[2 * n + i])
            let lb = srgbToLab(b.data[i], b.data[n + i], b.data[2 * n + i])
            if min(la.x, lb.x) < p.minL || max(la.x, lb.x) > p.maxL { continue }
            let d = SIMD3(p.lightnessWeight * (la.x - lb.x), la.y - lb.y, la.z - lb.z)
            out[i] = min(1, max(0, (simd_length(d) - p.deltaE0) / (p.deltaE1 - p.deltaE0)))
        }
        return out
    }

    /// C_B = clamp(C_A + λd·C_depth + λc·C_color, 0, 2); 0 outside the validity mask.
    public static func combine(cA: [Float], depth: [Float], color: [Float], valid: [Bool],
                               params p: Params = Params()) -> [Float] {
        (0..<cA.count).map { i in valid[i] ? min(2, max(0, cA[i] + p.lambdaDepth * depth[i] + p.lambdaColor * color[i])) : 0 }
    }

    /// sRGB (D65) in [0, 1] → CIELAB.
    public static func srgbToLab(_ r: Float, _ g: Float, _ b: Float) -> SIMD3<Float> {
        func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        let R = lin(r), G = lin(g), B = lin(b)
        let X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
        let Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
        let Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
        func f(_ t: Float) -> Float { t > 0.008856452 ? cbrt(t) : t / 0.12841855 + 4.0 / 29.0 }
        let fx = f(X), fy = f(Y), fz = f(Z)
        return SIMD3(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    /// Separable Gaussian blur, radius 3σ, clamp-to-edge.
    static func gaussianBlur(_ img: FloatImage, sigma: Float) -> FloatImage {
        let r = Int((3 * sigma).rounded(.up))
        var k = (-r...r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let s = k.reduce(0, +)
        k = k.map { $0 / s }
        let w = img.width, h = img.height
        var tmp = img, out = img
        for c in 0..<img.channels {
            for y in 0..<h { for x in 0..<w {
                var v: Float = 0
                for i in -r...r { v += k[i + r] * img[c, y, min(max(x + i, 0), w - 1)] }
                tmp[c, y, x] = v
            } }
            for y in 0..<h { for x in 0..<w {
                var v: Float = 0
                for i in -r...r { v += k[i + r] * tmp[c, min(max(y + i, 0), h - 1), x] }
                out[c, y, x] = v
            } }
        }
        return out
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter CuesBTests`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Add the Method B LiDAR depth and color cues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Session storage, keyframe selection, camera-image conversion

**Files:**
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Store/SessionStore.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Store/KeyframeSelector.swift`
- Create: `oscd_mobile/Packages/OSCDCore/Sources/OSCDCore/Image/CameraImage.swift`
- Test: `oscd_mobile/Packages/OSCDCore/Tests/OSCDCoreTests/StoreTests.swift`

**Interfaces:**
- Consumes: `NPYArray` (Task 1), `Intrinsics`, `Pose`, `simd_float4x4.columnMajor` (Task 2), `FloatImage` (Task 3).
- Produces:
  - `AABB(min:max:)` with `contains(_:)`.
  - `FrameRecord(id:timestamp:arkitCameraToWorld:pnpWorldToCamera:stats:)`.
  - `SessionManifest(deviceModel:captureWidth:captureHeight:depthWidth:depthHeight:intrinsics:)`, with mutable `roi`, `keyframes` and `sweep`.
  - `SessionStore(create:manifest:)` and `SessionStore(open:)`, with `manifest`, `append(_:record:rgb:depth:confidence:extra:)`, `setROI(_:)`, `rgb(_:id:)`, `depth(_:id:)`, `array(_:id:name:)`, and the nested types `Kind` (`.keyframes`, `.sweep`) and `StoreError`.
  - `KeyframeSelector()` with `consider(pose:time:trackingNormal:) -> Bool`.
  - `CameraImage.rgbArea(y:yStride:cbcr:cbcrStride:width:height:factor:) -> [UInt8]`.

The on-disk session format (also read by `tools/session_to_colmap.py`):
- `session.json` holds the manifest, with ISO-8601 dates and `SIMD3` values encoded as `[x, y, z]`.
- Each frame is stored as `keyframes/000012_rgb.npy` (u1, H×W×3), `_depth.npy` (f4, 192×256) and `_conf.npy` (u1). Optional extra arrays go in `_<name>.npy`.

- [ ] **Step 1: Write the failing tests**

`Tests/OSCDCoreTests/StoreTests.swift`:
```swift
import XCTest
import simd
@testable import OSCDCore

final class StoreTests: XCTestCase {
    func manifest() -> SessionManifest {
        SessionManifest(deviceModel: "test", captureWidth: 8, captureHeight: 4, depthWidth: 2, depthHeight: 1,
                        intrinsics: Intrinsics(f: 3, cx: 1.5, cy: 0.5, width: 4, height: 2))
    }

    func temp() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent("oscd-\(UUID().uuidString)") }

    func testRoundTrip() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        try store.setROI(AABB(min: SIMD3(-1, 0, -1), max: SIMD3(1, 0.5, 1)))
        let rec = FrameRecord(id: 0, timestamp: 1.5, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 255, count: 4 * 2 * 3), depth: [0.5, 0.6], confidence: [2, 1])
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes, [rec])
        XCTAssertEqual(back.manifest.roi?.max.y, 0.5)
        XCTAssertEqual(try back.rgb(.keyframes, id: 0).data.max(), 1)
        XCTAssertEqual(try back.depth(.keyframes, id: 0).confidence, [2, 1])
    }

    /// Review Focus 5: files written for a frame the manifest never recorded are ignored, and a
    /// listed frame with a missing file throws a clear error instead of crashing.
    func testPartialWriteIsIgnoredAndMissingFileThrows() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        let rec = FrameRecord(id: 0, timestamp: 0, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 0, count: 24), depth: [1, 1], confidence: [2, 2])
        try NPYArray(uint8: [UInt8](repeating: 0, count: 24), shape: [2, 4, 3])
            .write(to: root.appendingPathComponent("keyframes/000001_rgb.npy"))          // orphan, as after a crash
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes.map(\.id), [0])
        try FileManager.default.removeItem(at: root.appendingPathComponent("keyframes/000000_depth.npy"))
        XCTAssertThrowsError(try back.depth(.keyframes, id: 0))
        XCTAssertThrowsError(try SessionStore(open: temp()))
    }

    func testKeyframeSelector() {
        var s = KeyframeSelector()
        func pose(x: Float) -> Pose { Pose(R: matrix_identity_float3x3, t: SIMD3(-x, 0, 0)) }   // centre at (x, 0, 0)
        XCTAssertTrue(s.consider(pose: pose(x: 0), time: 0, trackingNormal: true))
        XCTAssertFalse(s.consider(pose: pose(x: 0.02), time: 0.2, trackingNormal: true))      // too close
        XCTAssertTrue(s.consider(pose: pose(x: 0.06), time: 0.4, trackingNormal: true))       // 6 cm on
        XCTAssertFalse(s.consider(pose: pose(x: 0.30), time: 0.5, trackingNormal: true))      // 2.4 m/s: too fast
        XCTAssertFalse(s.consider(pose: pose(x: 0.40), time: 1.5, trackingNormal: false))     // tracking limited
    }

    func testCameraImageConversion() {
        // 4×4 luma, 2×2 interleaved CbCr, factor 2 → 2×2 RGB.
        let gray = CameraImage.rgbArea(y: [UInt8](repeating: 128, count: 16), yStride: 4, cbcr: [UInt8](repeating: 128, count: 8), cbcrStride: 4,
                                       width: 4, height: 4, factor: 2)
        XCTAssertEqual(gray, [UInt8](repeating: 128, count: 12))
        var cbcr = [UInt8](repeating: 0, count: 8)
        for i in stride(from: 0, to: 8, by: 2) { cbcr[i] = 85; cbcr[i + 1] = 255 }        // pure red in BT.601 full range
        let red = CameraImage.rgbArea(y: [UInt8](repeating: 76, count: 16), yStride: 4, cbcr: cbcr, cbcrStride: 4, width: 4, height: 4, factor: 2)
        XCTAssertGreaterThanOrEqual(red[0], 250); XCTAssertLessThanOrEqual(red[1], 3); XCTAssertLessThanOrEqual(red[2], 3)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter StoreTests`
Expected: FAIL to compile (`cannot find 'SessionStore' in scope`).

- [ ] **Step 3: Implement the three source files**

`Sources/OSCDCore/Store/SessionStore.swift`:
```swift
import Foundation
import simd

public struct AABB: Codable, Equatable, Sendable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>
    public init(min: SIMD3<Float>, max: SIMD3<Float>) { self.min = min; self.max = max }
    public func contains(_ p: SIMD3<Float>) -> Bool { all(p .>= min) && all(p .<= max) }
}

public struct FrameRecord: Codable, Equatable, Sendable {
    public var id: Int
    public var timestamp: Double
    public var arkitCameraToWorld: [Float]           // 16 values, column-major
    public var pnpWorldToCamera: [Float]?            // 16 values when a PnP pose was accepted
    public var stats: [String: Double]

    public init(id: Int, timestamp: Double, arkitCameraToWorld: [Float], pnpWorldToCamera: [Float]? = nil,
                stats: [String: Double] = [:]) {
        self.id = id; self.timestamp = timestamp; self.arkitCameraToWorld = arkitCameraToWorld
        self.pnpWorldToCamera = pnpWorldToCamera; self.stats = stats
    }
}

public struct SessionManifest: Codable, Equatable, Sendable {
    public var version = 1
    public var deviceModel: String
    public var createdAt: Date
    public var captureWidth: Int
    public var captureHeight: Int
    public var depthWidth: Int
    public var depthHeight: Int
    public var intrinsics: Intrinsics                // at working resolution
    public var roi: AABB?
    public var keyframes: [FrameRecord] = []
    public var sweep: [FrameRecord] = []

    public init(deviceModel: String, captureWidth: Int, captureHeight: Int, depthWidth: Int, depthHeight: Int,
                intrinsics: Intrinsics, createdAt: Date = Date()) {
        self.deviceModel = deviceModel; self.captureWidth = captureWidth; self.captureHeight = captureHeight
        self.depthWidth = depthWidth; self.depthHeight = depthHeight; self.intrinsics = intrinsics; self.createdAt = createdAt
    }
}

/// A recorded session folder. Frame arrays are written before the manifest (atomically), so a crash
/// mid-frame leaves orphan files that no manifest entry points to.
public final class SessionStore {
    public enum Kind: String, Sendable { case keyframes, sweep }
    public enum StoreError: Error, Equatable { case notASession(String), missingFile(String) }

    public let root: URL
    public private(set) var manifest: SessionManifest
    static let manifestName = "session.json"

    public init(create root: URL, manifest: SessionManifest) throws {
        self.root = root
        self.manifest = manifest
        for kind in [Kind.keyframes, .sweep] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(kind.rawValue), withIntermediateDirectories: true)
        }
        try saveManifest()
    }

    public init(open root: URL) throws {
        self.root = root
        let url = root.appendingPathComponent(Self.manifestName)
        guard FileManager.default.fileExists(atPath: url.path) else { throw StoreError.notASession(root.path) }
        manifest = try Self.decoder.decode(SessionManifest.self, from: Data(contentsOf: url))
    }

    func url(_ kind: Kind, _ id: Int, _ name: String) -> URL {
        root.appendingPathComponent(kind.rawValue).appendingPathComponent(String(format: "%06d_%@.npy", id, name))
    }

    public func append(_ kind: Kind, record: FrameRecord, rgb: [UInt8], depth: [Float], confidence: [UInt8],
                       extra: [String: NPYArray] = [:]) throws {
        let K = manifest.intrinsics
        try NPYArray(uint8: rgb, shape: [K.height, K.width, 3]).write(to: url(kind, record.id, "rgb"))
        try NPYArray(floats: depth, shape: [manifest.depthHeight, manifest.depthWidth]).write(to: url(kind, record.id, "depth"))
        try NPYArray(uint8: confidence, shape: [manifest.depthHeight, manifest.depthWidth]).write(to: url(kind, record.id, "conf"))
        for (name, array) in extra { try array.write(to: url(kind, record.id, name)) }
        switch kind {
        case .keyframes: manifest.keyframes.append(record)
        case .sweep: manifest.sweep.append(record)
        }
        try saveManifest()
    }

    public func setROI(_ roi: AABB) throws {
        manifest.roi = roi
        try saveManifest()
    }

    public func rgb(_ kind: Kind, id: Int) throws -> FloatImage {
        let a = try load(kind, id, "rgb")
        return FloatImage(rgb8: a.uint8(), width: a.shape[1], height: a.shape[0])
    }

    public func depth(_ kind: Kind, id: Int) throws -> (depth: [Float], confidence: [UInt8]) {
        (try load(kind, id, "depth").floats(), try load(kind, id, "conf").uint8())
    }

    public func array(_ kind: Kind, id: Int, name: String) throws -> NPYArray { try load(kind, id, name) }

    func load(_ kind: Kind, _ id: Int, _ name: String) throws -> NPYArray {
        let u = url(kind, id, name)
        guard FileManager.default.fileExists(atPath: u.path) else { throw StoreError.missingFile(u.lastPathComponent) }
        return try NPYArray.load(u)
    }

    func saveManifest() throws {
        try Self.encoder.encode(manifest).write(to: root.appendingPathComponent(Self.manifestName), options: .atomic)
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
```

`Sources/OSCDCore/Store/KeyframeSelector.swift`:
```swift
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
```

`Sources/OSCDCore/Image/CameraImage.swift`:
```swift
/// ARKit `capturedImage` (420f: full-range BT.601 luma plane + interleaved CbCr plane at half size).
public enum CameraImage {
    /// RGB bytes (H/factor × W/factor × 3), each the mean of a factor×factor block (OpenCV INTER_AREA).
    public static func rgbArea(y: UnsafePointer<UInt8>, yStride: Int, cbcr: UnsafePointer<UInt8>, cbcrStride: Int,
                               width: Int, height: Int, factor: Int) -> [UInt8] {
        let ow = width / factor, oh = height / factor
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        let inv = 1 / Float(factor * factor)
        for oy in 0..<oh {
            for ox in 0..<ow {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for dy in 0..<factor {
                    let yy = oy * factor + dy
                    for dx in 0..<factor {
                        let xx = ox * factor + dx
                        let Y = Float(y[yy * yStride + xx])
                        let ci = (yy / 2) * cbcrStride + (xx / 2) * 2
                        let cb = Float(cbcr[ci]) - 128, cr = Float(cbcr[ci + 1]) - 128
                        r += min(255, max(0, Y + 1.402 * cr))
                        g += min(255, max(0, Y - 0.344136 * cb - 0.714136 * cr))
                        b += min(255, max(0, Y + 1.772 * cb))
                    }
                }
                let o = (oy * ow + ox) * 3
                out[o] = UInt8((r * inv).rounded()); out[o + 1] = UInt8((g * inv).rounded()); out[o + 2] = UInt8((b * inv).rounded())
            }
        }
        return out
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter StoreTests`
Expected: PASS (4 tests). The test passes `[UInt8]` arrays where `UnsafePointer<UInt8>` is expected; Swift's implicit array-to-pointer bridging handles that.

- [ ] **Step 5: Commit**

```bash
git add oscd_mobile/Packages/OSCDCore
git commit -m "Add session storage, keyframe selection and camera-image conversion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: iOS spike app: record sessions and benchmark every stage on the iPhone

**Files:**
- Create: `oscd_mobile/SpikeApp/project.yml`
- Create: `oscd_mobile/SpikeApp/Sources/OSCDSpikeApp.swift`
- Create: `oscd_mobile/SpikeApp/Sources/CaptureController.swift`
- Create: `oscd_mobile/SpikeApp/Sources/CaptureView.swift`
- Create: `oscd_mobile/SpikeApp/Sources/BenchmarkRunner.swift`
- Create: `oscd_mobile/SpikeApp/Sources/BenchmarkView.swift`
- Modify: `oscd_mobile/README.md` (add a "Spike app" section)

**Interfaces:**
- Consumes:
  - From Task 15: `SessionStore`, `SessionManifest`, `FrameRecord`, `AABB`, `KeyframeSelector`, `CameraImage.rgbArea`.
  - From Task 2: `Pose.fromARKitCameraToWorld`, `Intrinsics.fromARKit`.
  - Models and algorithms: `XFeatModel` (Task 6), `SAM2Encoder` (Task 13), `Matcher.countMutual` (Task 7), `P4PRANSAC.run` (Task 9), `MiniBA.solve` (Task 8), `CuesA.pixel` (Task 12), `AnchorBuilder.build` (Task 10).
- Produces:
  - Sessions in the app's `Documents/sessions/<ISO date>/`, visible in the Files app.
  - Benchmark CSVs in `Documents/benchmarks/bench_<date>.csv` with the columns `stage,median_ms,runs,note,device`.

- [ ] **Step 1: Write `project.yml`**

```yaml
name: OSCDSpike
options:
  bundleIdPrefix: edu.asu.oscd
  deploymentTarget:
    iOS: "17.0"
packages:
  OSCDCore:
    path: ../Packages/OSCDCore
targets:
  OSCDSpike:
    type: application
    platform: iOS
    sources:
      - path: Sources
      - path: ../Models
        optional: true
    dependencies:
      - package: OSCDCore
    info:
      path: Sources/Info.plist
      properties:
        CFBundleDisplayName: OSCD Spike
        NSCameraUsageDescription: Scans the tabletop for change detection.
        UIFileSharingEnabled: true
        LSSupportsOpeningDocumentsInPlace: true
        UILaunchScreen: {}
        UIRequiredDeviceCapabilities: [arkit]
        UISupportedInterfaceOrientations: [UIInterfaceOrientationPortrait]
    settings:
      base:
        CODE_SIGN_STYLE: Automatic
        PRODUCT_BUNDLE_IDENTIFIER: edu.asu.oscd.spike
        SWIFT_VERSION: "5.0"
```

- [ ] **Step 2: Write the app sources**

`SpikeApp/Sources/OSCDSpikeApp.swift`:
```swift
import SwiftUI

@main
struct OSCDSpikeApp: App {
    @StateObject private var capture = CaptureController()
    @StateObject private var bench = BenchmarkRunner()

    var body: some Scene {
        WindowGroup {
            TabView {
                CaptureView(controller: capture).tabItem { Label("Capture", systemImage: "camera.viewfinder") }
                BenchmarkView(runner: bench).tabItem { Label("Benchmark", systemImage: "speedometer") }
            }
        }
    }
}
```

`SpikeApp/Sources/CaptureController.swift`:
```swift
import ARKit
import AVFoundation
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Records reference-scan sessions: keyframes (RGB at 480×360, LiDAR depth + confidence, ARKit pose).
/// ARSession delegate callbacks arrive on the main thread; files are written on a serial queue.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    static let working = (width: 480, height: 360)

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Tap the table to place the scan region"
    @Published var keyframeCount = 0
    @Published var recording = false

    private var selector = KeyframeSelector()
    private var store: SessionStore?
    private var roi: AABB?
    private let writeQueue = DispatchQueue(label: "oscd.capture.write")

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.planeDetection = [.horizontal]
        config.isAutoFocusEnabled = false
        arView.session.delegate = self
        arView.session.run(config)
    }

    func placeROI(at point: CGPoint) {
        guard let query = arView.makeRaycastQuery(from: point, allowing: .existingPlaneGeometry, alignment: .horizontal),
              let hit = arView.session.raycast(query).first else {
            status = "No table plane there yet. Move the phone slowly over the table, then tap again."
            return
        }
        let c = hit.worldTransform.columns.3
        let p = SIMD3<Float>(c.x, c.y, c.z)
        roi = AABB(min: p - SIMD3(0.4, 0.02, 0.4), max: p + SIMD3(0.4, 0.35, 0.4))      // ARKit world is y-up
        status = "Region placed. Press Record and sweep around the table."
    }

    func toggleRecording() {
        if recording {
            recording = false
            store = nil
            status = "Saved \(keyframeCount) keyframes"
            return
        }
        guard roi != nil else { status = "Tap the table to place the scan region first"; return }
        lockCamera()
        selector = KeyframeSelector()
        keyframeCount = 0
        recording = true
        status = "Recording. Sweep slowly around the table."
    }

    /// Exposure, white balance and focus stay locked for the whole session (iOS 16+).
    private func lockCamera() {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
            if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
            device.unlockForConfiguration()
        } catch {
            status = "Could not lock the camera: \(error.localizedDescription)"
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard recording, let sceneDepth = frame.sceneDepth, let roi else { return }
        var normal = false
        if case .normal = frame.camera.trackingState { normal = true }
        let pose = Pose.fromARKitCameraToWorld(frame.camera.transform)
        guard selector.consider(pose: pose, time: frame.timestamp, trackingNormal: normal) else { return }
        if store == nil {
            do {
                store = try makeStore(frame: frame, depth: sceneDepth, roi: roi)
            } catch {
                status = "Could not create the session folder: \(error.localizedDescription)"
                recording = false
                return
            }
        }
        guard let store else { return }
        let rgb = Self.workingRGB(frame.capturedImage)
        let (depth, confidence) = Self.copyDepth(sceneDepth)
        let record = FrameRecord(id: keyframeCount, timestamp: frame.timestamp, arkitCameraToWorld: frame.camera.transform.columnMajor)
        keyframeCount += 1
        writeQueue.async {
            do {
                try store.append(.keyframes, record: record, rgb: rgb, depth: depth, confidence: confidence)
            } catch {
                DispatchQueue.main.async { self.status = "Writing keyframe \(record.id) failed: \(error.localizedDescription)" }
            }
        }
    }

    private func makeStore(frame: ARFrame, depth: ARDepthData, roi: AABB) throws -> SessionStore {
        let K = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let intrinsics = Intrinsics.fromARKit(fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                              captureWidth: Int(res.width), captureHeight: Int(res.height),
                                              workingWidth: Self.working.width, workingHeight: Self.working.height)
        var manifest = SessionManifest(deviceModel: Self.deviceModel(), captureWidth: Int(res.width), captureHeight: Int(res.height),
                                       depthWidth: CVPixelBufferGetWidth(depth.depthMap), depthHeight: CVPixelBufferGetHeight(depth.depthMap),
                                       intrinsics: intrinsics)
        manifest.roi = roi
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let name = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        return try SessionStore(create: docs.appendingPathComponent("sessions/\(name)"), manifest: manifest)
    }

    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }

    static func workingRGB(_ buffer: CVPixelBuffer) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0), h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let cbcr = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        return CameraImage.rgbArea(y: y, yStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                                   cbcr: cbcr, cbcrStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
                                   width: w, height: h, factor: w / working.width)
    }

    static func copyDepth(_ data: ARDepthData) -> ([Float], [UInt8]) {
        let dm = data.depthMap
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        let w = CVPixelBufferGetWidth(dm), h = CVPixelBufferGetHeight(dm)
        var depth = [Float](repeating: 0, count: w * h), conf = [UInt8](repeating: 0, count: w * h)
        let dp = CVPixelBufferGetBaseAddress(dm)!, dr = CVPixelBufferGetBytesPerRow(dm)
        for y in 0..<h {
            let row = (dp + y * dr).assumingMemoryBound(to: Float32.self)
            for x in 0..<w { depth[y * w + x] = row[x] }
        }
        if let cm = data.confidenceMap {
            CVPixelBufferLockBaseAddress(cm, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(cm, .readOnly) }
            let cp = CVPixelBufferGetBaseAddress(cm)!, cr = CVPixelBufferGetBytesPerRow(cm)
            for y in 0..<h {
                let row = (cp + y * cr).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { conf[y * w + x] = row[x] }
            }
        }
        return (depth, conf)
    }
}
```

`SpikeApp/Sources/CaptureView.swift`:
```swift
import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    let controller: CaptureController

    func makeUIView(context: Context) -> ARView {
        let view = controller.arView
        view.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tap(_:))))
        controller.start()
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(controller: controller) }

    final class Coordinator: NSObject {
        let controller: CaptureController
        init(controller: CaptureController) { self.controller = controller }
        @objc func tap(_ gesture: UITapGestureRecognizer) { controller.placeROI(at: gesture.location(in: gesture.view)) }
    }
}

struct CaptureView: View {
    @ObservedObject var controller: CaptureController

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(controller: controller).ignoresSafeArea()
            VStack(spacing: 12) {
                Text(controller.status)
                    .font(.callout)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                Text("Keyframes: \(controller.keyframeCount)").monospacedDigit()
                Button(controller.recording ? "Stop" : "Record") { controller.toggleRecording() }
                    .buttonStyle(.borderedProminent)
                    .tint(controller.recording ? .red : .accentColor)
            }
            .padding(.bottom, 32)
        }
    }
}
```

`SpikeApp/Sources/BenchmarkRunner.swift`:
```swift
import CoreML
import Foundation
import OSCDCore
import simd

struct BenchRow: Identifiable {
    let id = UUID()
    let stage: String
    let medianMs: Double
    let runs: Int
    let note: String
}

/// Times every Phase 1 stage on the device and writes Documents/benchmarks/bench_<date>.csv.
final class BenchmarkRunner: ObservableObject {
    @Published var rows: [BenchRow] = []
    @Published var running = false
    @Published var message = ""

    func run() {
        running = true
        rows = []
        message = "Running…"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let rows = try Self.measureAll { msg in DispatchQueue.main.async { self.message = msg } }
                DispatchQueue.main.async { self.rows = rows; self.running = false; self.message = "Saved to Documents/benchmarks" }
            } catch {
                DispatchQueue.main.async { self.running = false; self.message = "Failed: \(error.localizedDescription)" }
            }
        }
    }

    static func median(_ runs: Int, warmup: Int = 2, _ body: () throws -> Void) rethrows -> Double {
        for _ in 0..<warmup { try body() }
        var times: [Double] = []
        for _ in 0..<runs {
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        return times.sorted()[times.count / 2]
    }

    static func measureAll(progress: (String) -> Void) throws -> [BenchRow] {
        guard let xfeatURL = Bundle.main.url(forResource: "XFeat_480x352", withExtension: "mlmodelc"),
              let samURL = Bundle.main.url(forResource: "SAM2_1Tiny_2x1024", withExtension: "mlmodelc") else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Models are missing from the app. Run the two converters, then `xcodegen generate` again."])
        }
        var rows: [BenchRow] = []
        let session = try latestSession()
        let image: FloatImage
        if let session, let first = session.manifest.keyframes.first {
            image = try session.rgb(.keyframes, id: first.id)
        } else {
            image = FloatImage(width: 480, height: 360, channels: 3, fill: 0.5)
        }

        progress("XFeat")
        let xfeat = try XFeatModel(compiledModelURL: xfeatURL)
        var kp = DescribedKeypoints(kpts: [], feats: [])
        let xfeatMs = try median(20) { kp = try xfeat.detect(image) }
        rows.append(BenchRow(stage: "xfeat_detect", medianMs: xfeatMs, runs: 20, note: "\(kp.count) keypoints"))

        progress("SAM2")
        let sam = try SAM2Encoder(compiledModelURL: samURL)
        rows.append(BenchRow(stage: "sam2_encode_pair", medianMs: try median(10) { _ = try sam.encode(render: image, live: image) }, runs: 10, note: "batch of 2 at 1024²"))

        progress("Retrieval")
        let frame = kp
        let refs = [DescribedKeypoints](repeating: frame, count: 80)
        rows.append(BenchRow(stage: "retrieval_80_keyframes", medianMs: median(3, warmup: 1) {
            var counts = [Int](repeating: 0, count: refs.count)
            counts.withUnsafeMutableBufferPointer { c in
                DispatchQueue.concurrentPerform(iterations: refs.count) { c[$0] = Matcher.countMutual(refs[$0], frame) }
            }
        }, runs: 3, note: "\(frame.count) keypoints per image"))

        progress("P4P-RANSAC and mini-BA")
        let (uv, xyz, start) = syntheticCorrespondences(2000)
        let centre = SIMD2<Double>(239.5, 179.5)
        rows.append(BenchRow(stage: "p4p_ransac_2000", medianMs: median(5, warmup: 1) {
            var s = SeededSampler(seed: 0)
            _ = P4PRANSAC.run(uv: uv, xyz: xyz, confs: [Double](repeating: 1, count: uv.count), f: 360, centre: centre,
                              initR6: start.r6, initT: start.tDouble, maxError: 1.5, sampler: &s)
        }, runs: 5, note: "2000 hypotheses × 20 LM"))
        rows.append(BenchRow(stage: "mini_ba_2000", medianMs: median(5, warmup: 1) {
            _ = MiniBA.solve(r6: start.r6, t: start.tDouble, f: 360, centre: centre, xyz: xyz, uv: uv, config: .poseRefine)
        }, runs: 5, note: "20 iterations"))

        progress("Pixel cue")
        rows.append(BenchRow(stage: "pixel_cue_ssim", medianMs: median(5, warmup: 1) { _ = CuesA.pixel(live: image, render: image) }, runs: 5, note: "480×360"))

        if let session, session.manifest.keyframes.count >= 20 {
            progress("Anchors on the latest session")
            var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
            let t0 = DispatchTime.now().uptimeNanoseconds
            for r in session.manifest.keyframes {
                let img = try session.rgb(.keyframes, id: r.id)
                views.append((Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld)), try xfeat.detect(img)))
            }
            let t1 = DispatchTime.now().uptimeNanoseconds
            let anchors = AnchorBuilder.build(keyframes: views, intrinsics: session.manifest.intrinsics)
            let t2 = DispatchTime.now().uptimeNanoseconds
            rows.append(BenchRow(stage: "xfeat_all_keyframes", medianMs: Double(t1 - t0) / 1e6, runs: 1, note: "\(views.count) keyframes"))
            rows.append(BenchRow(stage: "anchor_build", medianMs: Double(t2 - t1) / 1e6, runs: 1,
                                 note: String(format: "yield %.3f, %d anchors", anchors.yield, anchors.anchorCount)))
        }
        try writeCSV(rows)
        return rows
    }

    static func latestSession() throws -> SessionStore? {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("sessions")
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path), let last = names.sorted().last else { return nil }
        return try SessionStore(open: dir.appendingPathComponent(last))
    }

    static func syntheticCorrespondences(_ n: Int) -> ([SIMD2<Double>], [SIMD3<Double>], Pose) {
        var s = SeededSampler(seed: 7)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0))), t: SIMD3(0.05, 0, 0.8))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = []
        for i in 0..<n {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i % 3 == 0 { p = SIMD2(Float(s.uniform()) * 480, Float(s.uniform()) * 360) }
            uv.append(SIMD2<Double>(p))
            xyz.append(SIMD3<Double>(X))
        }
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.08, axis: SIMD3(1, 0, 0))), t: truth.t + SIMD3(0.03, 0, 0))
        return (uv, xyz, start)
    }

    static func writeCSV(_ rows: [BenchRow]) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("benchmarks")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let device = CaptureController.deviceModel()
        var csv = "stage,median_ms,runs,note,device\n"
        for r in rows { csv += "\(r.stage),\(String(format: "%.2f", r.medianMs)),\(r.runs),\"\(r.note)\",\(device)\n" }
        let name = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        try csv.write(to: dir.appendingPathComponent("bench_\(name).csv"), atomically: true, encoding: .utf8)
    }
}
```

`SpikeApp/Sources/BenchmarkView.swift`:
```swift
import SwiftUI

struct BenchmarkView: View {
    @ObservedObject var runner: BenchmarkRunner

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button(runner.running ? "Running…" : "Run benchmarks") { runner.run() }
                        .disabled(runner.running)
                    if !runner.message.isEmpty {
                        Text(runner.message).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                Section("Median milliseconds") {
                    ForEach(runner.rows) { row in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(row.stage).font(.body.monospaced())
                                Text(row.note).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(String(format: "%.1f", row.medianMs)).monospacedDigit()
                        }
                    }
                }
            }
            .navigationTitle("Spike benchmarks")
        }
    }
}
```

- [ ] **Step 3: Generate the project and compile for a device without signing**

```bash
cd oscd_mobile/SpikeApp && xcodegen generate
xcodebuild -project OSCDSpike.xcodeproj -scheme OSCDSpike -destination 'generic/platform=iOS' -configuration Release build CODE_SIGNING_ALLOWED=NO | tail -3
```
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 4: Human step. Install, record, benchmark**

1. Open `oscd_mobile/SpikeApp/OSCDSpike.xcodeproj`, choose your team under Signing, and run on the iPhone Pro.
2. **Capture tab.** Put the blocks on a patterned mat. Move the phone over the table until a plane is found, tap the table, press **Record**, and sweep for about 30 s, aiming for 60–100 keyframes. Press **Stop**. Record two sessions: one of the clean table, and one after moving, adding and recoloring blocks.
3. **Benchmark tab.** Press **Run benchmarks**.
4. Copy the session folders and the benchmark CSV to the Mac via Finder (iPhone → Files → OSCD Spike), into `oscd_mobile/fixtures/sessions/` and `oscd_mobile/fixtures/benchmarks/`.

Expected: every row has a value, `anchor_build` shows a yield, and the session folders contain `session.json` plus about 3 files per keyframe.

- [ ] **Step 5: Document and commit**

Append to `oscd_mobile/README.md`:
```markdown
## Spike app
    cd SpikeApp && xcodegen generate && open OSCDSpike.xcodeproj
Capture tab: tap the table, Record, sweep, Stop → Documents/sessions/<date>/ (Files app).
Benchmark tab: times each stage on the device → Documents/benchmarks/bench_<date>.csv.
```

```bash
git add oscd_mobile/SpikeApp/project.yml oscd_mobile/SpikeApp/Sources oscd_mobile/README.md
git commit -m "Add the iOS spike app for session recording and on-device benchmarks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: Engine evaluation, anchor yield, spike report and go/no-go

**Files:**
- Modify: `oscd_mobile/Packages/OSCDCore/Package.swift` (add the `oscd-harness` executable target)
- Create: `oscd_mobile/Packages/OSCDCore/Sources/oscd-harness/main.swift`
- Create: `oscd_mobile/tools/session_to_colmap.py`
- Create: `oscd_mobile/docs/spike_report.md`

**Interfaces:**
- Consumes: Task 15's session format and the recorded sessions; `AnchorBuilder.build(…, disambiguationDepths:)` (Task 10); `XFeatModel` (Task 6).
- Produces:
  - `swift run -c release oscd-harness anchors <session-dir> <XFeat .mlpackage> [d1 d2]`.
  - A COLMAP text model of a session for the splat engines.
  - `docs/spike_report.md` with every decision Phase 2 depends on.

- [ ] **Step 1: Add the harness**

In `Package.swift`, add to `targets`:
```swift
        .executableTarget(name: "oscd-harness", dependencies: ["OSCDCore"]),
```

`Sources/oscd-harness/main.swift`:
```swift
import Foundation
import OSCDCore
import simd

// Usage: oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2]
// Builds O-SCD anchors for a recorded session and prints the anchor yield. The optional
// disambiguation depths test spec deviation D11 (O-SCD uses 1 and 10 world units).
let args = CommandLine.arguments
guard args.count >= 4, args[1] == "anchors" else {
    print("usage: oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2]")
    exit(2)
}
let store = try SessionStore(open: URL(fileURLWithPath: args[2]))
let xfeat = try XFeatModel(compiledModelURL: try XFeatModel.compile(URL(fileURLWithPath: args[3])))
let depths: (Float, Float) = args.count >= 6 ? (Float(args[4])!, Float(args[5])!) : (1, 10)
var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
for r in store.manifest.keyframes {
    let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
    views.append((pose, try xfeat.detect(try store.rgb(.keyframes, id: r.id))))
}
let t0 = Date()
let anchors = AnchorBuilder.build(keyframes: views, intrinsics: store.manifest.intrinsics, disambiguationDepths: depths)
print(String(format: "keyframes %d  anchors %d  yield %.3f  depths (%.2f, %.2f)  build %.1f s",
             views.count, anchors.anchorCount, anchors.yield, depths.0, depths.1, Date().timeIntervalSince(t0)))
```

Run: `cd oscd_mobile/Packages/OSCDCore && swift build -c release --product oscd-harness`
Expected: `Build complete!`

- [ ] **Step 2: Write `session_to_colmap.py`**

```python
#!/usr/bin/env python
"""Export a recorded oscd_mobile session as a COLMAP text model (PINHOLE, ARKit poses, LiDAR points)
so the candidate splat engines can train on it."""
import argparse
import json
import os

import numpy as np
from PIL import Image


def rot_to_qvec(R):
    """Rotation matrix → COLMAP (qw, qx, qy, qz)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.004)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.session, "session.json")))
    K = man["intrinsics"]
    W, H, f, cx, cy = K["width"], K["height"], K["f"], K["cx"], K["cy"]
    dw, dh = man["depthWidth"], man["depthHeight"]
    roi = man.get("roi")
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sparse", "0"), exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    pts, cols, img_lines = [], [], []
    for n, kf in enumerate(man["keyframes"]):
        stem = os.path.join(a.session, "keyframes", f"{kf['id']:06d}")
        rgb = np.load(stem + "_rgb.npy")
        depth = np.load(stem + "_depth.npy")
        conf = np.load(stem + "_conf.npy")
        name = f"{kf['id']:06d}.png"
        Image.fromarray(rgb).save(os.path.join(a.out, "images", name))
        c2w = np.array(kf["arkitCameraToWorld"], dtype=np.float64).reshape(4, 4).T @ flip   # column-major → OpenCV axes
        w2c = np.linalg.inv(c2w)
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        img_lines.append(f"{n + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
        # LiDAR → world points: depth-map intrinsics are the working ones scaled by dw / W.
        s = dw / W
        fd, cxd, cyd = f * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5
        v, u = np.nonzero((conf >= 2) & np.isfinite(depth) & (depth > 0))
        z = depth[v, u]
        cam = np.stack([(u - cxd) / fd * z, (v - cyd) / fd * z, z, np.ones_like(z)])
        world = (c2w @ cam)[:3].T
        uw = np.clip(np.round((u + 0.5) / s - 0.5).astype(int), 0, W - 1)
        vw = np.clip(np.round((v + 0.5) / s - 0.5).astype(int), 0, H - 1)
        pts.append(world)
        cols.append(rgb[vw, uw])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    if roi:
        lo, hi = np.array(roi["min"]), np.array(roi["max"])
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, cols = pts[keep], cols[keep]
    _, first = np.unique(np.floor(pts / a.voxel).astype(np.int64), axis=0, return_index=True)
    pts, cols = pts[first], cols[first]
    with open(os.path.join(a.out, "sparse", "0", "cameras.txt"), "w") as fh:
        fh.write(f"1 PINHOLE {W} {H} {f} {f} {cx} {cy}\n")
    with open(os.path.join(a.out, "sparse", "0", "images.txt"), "w") as fh:
        fh.writelines(img_lines)
    with open(os.path.join(a.out, "sparse", "0", "points3D.txt"), "w") as fh:
        for i, (p, c) in enumerate(zip(pts, cols)):
            fh.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"{len(man['keyframes'])} images, {len(pts)} points → {a.out}")


if __name__ == "__main__":
    main()
```

Run: `oscd_mobile/tools/.venv/bin/python oscd_mobile/tools/session_to_colmap.py oscd_mobile/fixtures/sessions/<clean-session> /tmp/table_colmap`
Expected: `N images, M points → /tmp/table_colmap`, with N = the keyframe count and M in the tens of thousands.

- [ ] **Step 3: Measure anchor yield (deviation D11)**

```bash
cd oscd_mobile/Packages/OSCDCore
S=../../fixtures/sessions/<clean-session>; M=../../Models/XFeat_480x352.mlpackage
swift run -c release oscd-harness anchors "$S" "$M"            # O-SCD default depths (1, 10) m
swift run -c release oscd-harness anchors "$S" "$M" 0.1 1.0    # tabletop-scaled alternative
```
Expected: two lines `keyframes … anchors … yield …`. **Decision rule:** if the default yield is below 0.15 and the scaled one is at least 2× higher, adopt (0.1, 1.0) as deviation D11. Otherwise keep O-SCD's values.

- [ ] **Step 4: Human + agent step. Evaluate the two splat engines on the exported table**

```bash
mkdir -p oscd_mobile/third_party && cd oscd_mobile/third_party
git clone https://github.com/PoSTMEDIA-AI/splatx-metal && git clone https://github.com/Plinth-it/msplat-ios
cd splatx-metal && swift run -c release splatx train --data /tmp/table_colmap --factor 1 --iters 3000 --max-points 200000 --out /tmp/table_splatx.ply --save-preview /tmp/table_splatx.png
cd ../msplat-ios && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ./build/msplat --help
```
- **msplat on the Mac:** run the same 3000-iteration training with `msplat`, using the flags its `--help` prints for data path, iterations and maximum Gaussians.
- **Both engines on the iPhone:** build each repository's iOS example app (see each README), import `/tmp/table_colmap`, and train 3000 iterations at 480×360. Record the wall time and peak memory from Xcode's gauges.
- **Capability check:** read each engine's source and answer these, citing `file:line`:

1. Can a custom per-pixel loss (our SSF kernel) be supplied?
2. Can each parameter have its own learning rate, including xyz = 0?
3. Can it render expected depth?
4. Can densify, clone/split, prune and opacity reset be triggered from outside the training loop?
5. Does it match O-SCD's rasterizer constants: α ≤ 0.99, skip α < 1/255, stop at T < 1e-4, +0.3 low-pass, 3σ?
6. Can it render SH degree 0?
7. Can it initialize from our points (PLY or COLMAP)?

**Decision rule:**
- Pick the engine that meets items 1–3 natively, or with at most one day of patching, and trains 3k iterations on the iPhone in 180 s or less.
- If neither trains on the iPhone in 180 s or less, the Mac fallback becomes the default reference builder (spec §7.4).

- [ ] **Step 5: Write `docs/spike_report.md` with the measured numbers and decisions**

```markdown
# Spike report: O-SCD Mobile (day 3)

Device: <model from bench CSV> · iOS <version> · date <YYYY-MM-DD> · fixtures scene <scene> · sessions <names>

## On-device timings (median ms, from Documents/benchmarks/bench_<date>.csv)
| Stage | Target (spec §11) | Measured | Pass |
|---|---|---|---|
| XFeat per frame | ≤ 15 | | |
| Retrieval vs 80 keyframes | ≤ 60 | | |
| P4P-RANSAC + mini-BA | ≤ 20 | | |
| SAM2 encoder (pair) | ≤ 240 | | |
| Pixel cue (SSIM) | n/a | | |
| XFeat on all keyframes + anchor build | ≤ 30 s total | | |

## Anchor yield (D11)
| Depths | Yield | Anchors | Decision |
|---|---|---|---|
| (1, 10) m, O-SCD | | | |
| (0.1, 1.0) m | | | |

## Splat engines
| Check | splatx-metal | msplat-ios |
|---|---|---|
| 3k iters at 480×360 on iPhone (s) | | |
| Peak memory (MB) | | |
| Mac 3k iters (s) | | |
| 1 Custom loss hook (file:line) | | |
| 2 Per-parameter LR incl. xyz = 0 | | |
| 3 Expected-depth render | | |
| 4 External densify/prune/reset | | |
| 5 Rasterizer constants match O-SCD | | |
| 6 SH degree 0 | | |
| 7 Init from our points | | |

## Decisions
- Engine: <splatx-metal | msplat-ios>, because <one line>.
- Reference build: <on device | Mac fallback by default>.
- D9 (fewer keypoints or lower resolution): <none | details>.
- D11 (disambiguation depths): <O-SCD (1, 10) | (0.1, 1.0)>.
- Go / no-go for Phase 2: <go | no-go + reason>.
```
Fill in every cell from the measurements in Steps 3–4 and the CSV. Leave no cell empty: write "not measured: <reason>" where a measurement failed.

- [ ] **Step 6: Commit and hand off**

```bash
git add oscd_mobile/Packages/OSCDCore/Package.swift oscd_mobile/Packages/OSCDCore/Sources/oscd-harness oscd_mobile/tools/session_to_colmap.py oscd_mobile/docs/spike_report.md
git commit -m "Add the spike harness and COLMAP export, and record the day-3 spike report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Then ask the human to review `oscd_mobile/docs/spike_report.md`. That report is the input to the Phase 2 plan.

---

## After Phase 1

Phase 2 gets its own plan, written from `docs/spike_report.md`. It covers:
- **Engine integration:** the `SplatEngine` adapter over the chosen engine.
- **Reference build:** `ReferenceBuilder` (LiDAR init, depth check), the anchors persisted next to it, and the Mac `RefServer`.
- **Fusion:** `R_change` + SSF online fusion with step-4 densification and the refine loop, plus their parity tests against fixtures that `dump_oscd_fixtures.py` is extended to save (an SSF step and refine).
- **Live app and Method B:** the three-stage live pipeline, the display bridging, the SweepStore-based Method B replay, and the A/B toggle and HUD.
- **Rehearsal:** the runbook, IoU checks on hand-labelled frames, and rehearsals.
