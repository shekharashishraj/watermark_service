import OSCDCore
import SwiftUI

/// Pick the before-scan (it needs its 3D model) and the after-scan, then run detection.
struct DetectScreen: View {
    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var runner: DetectionRunner
    @State private var afterID = ""
    @State private var refine = false
    @State private var withB = true
    @State private var showRun = false

    private var before: ScanInfo? { library.before }
    private var after: ScanInfo? {
        library.scans.first { $0.id == afterID && $0.id != before?.id } ?? library.scans.last { $0.id != before?.id }
    }

    var body: some View {
        VStack(spacing: 0) {
            VeriHeader(title: "Detect")
            List {
                Section {
                    if library.scans.contains(where: \.hasReference) {
                        ForEach(library.scans.filter(\.hasReference)) { scan in
                            pick(scan, selected: scan.id == before?.id) { library.beforeID = scan.id }
                        }
                    } else {
                        Text("No scan has a 3D model yet. Record the before-scan, then build its model on the Mac with tools/build_reference.sh.")
                            .font(.system(size: 13)).foregroundColor(VeriColor.nickel)
                    }
                } header: { Text("Before") } footer: { Text("The scan whose 3D model is the reference.") }

                Section("After: check for changes") {
                    ForEach(library.scans.filter { $0.id != before?.id }.reversed()) { scan in
                        pick(scan, selected: scan.id == after?.id) { afterID = scan.id }
                    }
                }

                Section("Method") {
                    HStack {
                        Text("A · O-SCD (exact)")
                        Spacer()
                        Image(systemName: "checkmark.circle.fill").foregroundColor(VeriColor.primary)
                    }
                    Toggle(isOn: $withB) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Also B · adds LiDAR depth and colour")
                            Text("Same frames and camera positions as A; compare them in the results").font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                        }
                    }
                    .tint(VeriColor.primary)
                    Toggle(isOn: $refine) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Refine at the end")
                            Text("O-SCD's final pass, about 1 minute more").font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                        }
                    }
                    .tint(VeriColor.primary)
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            VeriButton(title: "Run detection", systemImage: "play.fill", disabled: before == nil || after == nil) {
                guard let before, let after else { return }
                runner.start(before: before, after: after, refine: refine, withB: withB)
                showRun = true
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
        .background(VeriColor.background)
        .onAppear {
            library.refresh()
            autoRunFromLaunchArguments()
        }
        .fullScreenCover(isPresented: $showRun, onDismiss: { library.refresh() }) {
            DetectionSheet(runner: runner) { showRun = false }
        }
    }

    private func autoRunFromLaunchArguments() {
        let d = UserDefaults.standard
        guard !Self.autoRan, let b = d.string(forKey: "detectBefore"), let a = d.string(forKey: "detectAfter") else { return }
        Self.autoRan = true
        library.beforeID = b
        afterID = a
        guard let before = library.scans.first(where: { $0.id == b && $0.hasReference }),
              let after = library.scans.first(where: { $0.id == a }) else { return }
        runner.start(before: before, after: after, refine: d.bool(forKey: "detectRefine"), withB: d.bool(forKey: "detectB"))
        showRun = true
    }

    private static var autoRan = false

    private func pick(_ scan: ScanInfo, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(scan.label).foregroundColor(VeriColor.textPrimary)
                    Text("\(scan.frameCount) frames · \(Formatters.date(scan.createdAt))").font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                }
                Spacer()
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundColor(selected ? VeriColor.primary : VeriColor.disabledFill)
            }
        }
    }
}

/// Progress while detection runs, then the results.
struct DetectionSheet: View {
    @ObservedObject var runner: DetectionRunner
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VeriHeader(title: runner.result != nil ? "Results" : "Detecting", trailing: AnyView(
                    Button(runner.running ? "Cancel" : "Close") {
                        if runner.running { runner.cancel() } else { onClose() }
                    }
                    .font(.system(size: 15, weight: .medium)).foregroundColor(VeriColor.primary)
                ))
                if let result = runner.result {
                    ResultsView(result: result)
                } else {
                    progressList
                }
            }
            .background(VeriColor.background)
        }
        .interactiveDismissDisabled(runner.running)
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
    }

    private var progressList: some View {
        List {
            Section {
                ForEach(runner.steps) { step in stepRow(step) }
            }
            if let image = runner.latest {
                Section {
                    Image(uiImage: image).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 8))
                } footer: { Text("Latest frame. Red: changed. Dimmed: not judged (off the table or not in the 3D model).") }
            }
            if let error = runner.error {
                Section("Stopped") { Text(error).foregroundColor(VeriColor.danger).font(.system(size: 13)) }
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
    }

    private func stepRow(_ step: DetectionRunner.Step) -> some View {
        HStack(alignment: .top, spacing: 10) {
            switch step.state {
            case .pending: Image(systemName: "circle").foregroundColor(VeriColor.mediumGrey)
            case .active: ProgressView().controlSize(.small)
            case .done: Image(systemName: "checkmark.circle.fill").foregroundColor(VeriColor.darkGreen)
            case .failed: Image(systemName: "xmark.circle.fill").foregroundColor(VeriColor.danger)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(step.title).font(.system(size: 14)).foregroundColor(VeriColor.textPrimary)
                switch step.state {
                case .active(let s), .done(let s), .failed(let s):
                    if !s.isEmpty { Text(s).font(.system(size: 11)).foregroundColor(VeriColor.nickel) }
                case .pending: EmptyView()
                }
                if step.id == 1, case .active = step.state {
                    ProgressView(value: runner.progress).tint(VeriColor.primary)
                }
            }
        }
    }
}

/// The masks (one frame at a time, with a slider) and the timings, per stage.
struct ResultsView: View {
    let result: DetectionResult
    @State private var index = 0.0
    @State private var showB = false

    private var run: DetectionResult.Run { showB ? (result.b ?? result.a) : result.a }

    var body: some View {
        let s = run.summary
        let frames = run.frames
        List {
            if let b = result.b {
                Section {
                    Picker("Method", selection: $showB) {
                        Text(String(format: "A · %.1f%% changed", 100 * result.a.summary.changedFraction)).tag(false)
                        Text(String(format: "B · %.1f%% changed", 100 * b.summary.changedFraction)).tag(true)
                    }
                    .pickerStyle(.segmented)
                } footer: { Text("A: O-SCD exactly. B: O-SCD plus the LiDAR depth and colour cues, over the same frames.") }
            }
            Section {
                if frames.isEmpty {
                    Text("No frame of \(result.afterLabel) could be lined up with \(result.beforeLabel), so nothing was judged. The two scans need the same table, seen from similar distances.")
                        .font(.system(size: 13)).foregroundColor(VeriColor.nickel)
                } else {
                    let i = min(Int(index.rounded()), frames.count - 1)
                    if let data = try? Data(contentsOf: frames[i]), let image = UIImage(data: data), let cg = image.cgImage {
                        Image(uiImage: UIImage(cgImage: cg, scale: 1, orientation: .right))
                            .resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    if frames.count > 1 {
                        Slider(value: $index, in: 0...Double(frames.count - 1), step: 1).tint(VeriColor.primary)
                    }
                    Text("Frame \(i + 1) of \(frames.count) · red: changed · dimmed: not judged")
                        .font(.system(size: 11)).foregroundColor(VeriColor.nickel)
                }
            } header: { Text("\(result.afterLabel) against \(result.beforeLabel) · Method \(showB ? "B" : "A")") }

            Section("Summary") {
                row("Camera placed", "\(s.framesWithPose) of \(s.frames) frames")
                row("Changed", s.framesWithPose == 0 ? "not judged" : String(format: "%.1f%% of the judged table area", 100 * s.changedFraction))
                row("Refine", s.refined ? "Yes" : "No (online fusion only)")
                row("Total time", String(format: "%.0f s", s.totalSeconds))
            }

            if let objects = result.objects, !objects.events.isEmpty {
                ObjectsSection(report: objects, before: result.before)
            }

            let setup = s.timings.stages.filter { MethodA.Stage.setup.contains($0) }
            if !setup.isEmpty {
                Section("Setup (once)") {
                    ForEach(setup, id: \.self) { row($0, Formatters.ms(s.timings.total($0))) }
                }
            }
            let finishing = [MethodA.Stage.refine, "B: " + MethodA.Stage.refine, MethodA.Stage.finalMasks, DetectionRunner.findObjects]
            let perFrame = s.timings.stages.filter { !MethodA.Stage.setup.contains($0) && !finishing.contains($0) }
            if !perFrame.isEmpty {
                Section {
                    ForEach(perFrame, id: \.self) { stage in
                        row(stage, (s.timings.median(stage).map(Formatters.ms) ?? "–") + "  ×\(s.timings.count(stage))")
                    }
                } header: { Text("Per frame (median)") } footer: {
                    Text("×N is how many frames ran that step. Frames where the camera could not be placed stop after the pose steps. SAM2 on the live image runs alongside the pose, so it does not add to the frame time.")
                }
            }
            let after = s.timings.stages.filter { finishing.contains($0) }
            if !after.isEmpty {
                Section("After all frames") {
                    ForEach(after, id: \.self) { row($0, Formatters.ms(s.timings.total($0))) }
                }
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).foregroundColor(VeriColor.nickel)
            Spacer()
            Text(value).foregroundColor(VeriColor.textPrimary).monospacedDigit()
        }
        .font(.system(size: 13))
    }
}

/// The object events of a finished run, with a link to see them over the before-scan's 3D model.
struct ObjectsSection: View {
    let report: ObjectReport
    let before: ScanInfo?

    var body: some View {
        Section {
            ObjectEventList(report: report, compact: false, highlighted: .constant(nil))
            if let before {
                NavigationLink("Show in 3D") { ModelViewerScreen(scan: before, report: report) }
            }
        } header: {
            Text("Objects")
        } footer: {
            Text("Positions are in the before-scan's frame. Dashed: probably the same object. Hints are not counted.")
        }
    }
}

#if DEBUG
extension ObjectReport {
    /// One moved, one added, one removed and one altered event, around the origin on a table at y = 0.
    static var previewSample: ObjectReport {
        func place(_ x: Float, _ z: Float) -> Placement {
            let c = SIMD3<Float>(x, 0.03, z)
            return Placement(centroid: c, bbox: AABB(min: c - SIMD3(0.03, 0.03, 0.03), max: c + SIMD3(0.03, 0.03, 0.03)),
                             voxels: 120, frames: [0, 1, 2])
        }
        return ObjectReport(beforeID: "before", afterID: "after", method: "A", voxelSizeM: 0.01, events: [
            ObjectEvent(id: 1, kind: .moved, confidence: 0.8, hint: false, label: "Ball", labelConfidence: 0.9,
                        before: place(-0.12, 0.05), after: place(0.1, -0.03),
                        motion: Motion(translation: SIMD3(0.22, 0, -0.08), horizontalDistance: 0.23, yawDeg: nil, fitRMS: 0.004, inlierFraction: 0.9)),
            ObjectEvent(id: 2, kind: .added, confidence: 0.7, hint: false, label: "Cup", labelConfidence: 0.5, after: place(0.05, 0.12)),
            ObjectEvent(id: 3, kind: .removed, confidence: 0.7, hint: false, before: place(-0.1, -0.1)),
            ObjectEvent(id: 4, kind: .altered, confidence: 0.4, hint: true, label: "Book", labelConfidence: 0.8, after: place(0.14, 0.1)),
        ])
    }
}

#Preview("Objects section") {
    List { ObjectsSection(report: .previewSample, before: nil) }
}
#endif
