import OSCDCore
import SwiftUI

/// The scans list: every recorded scan with its thumbnail, frame count, time and status.
struct ScansScreen: View {
    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var runner: DetectionRunner
    @State private var pendingDelete: ScanInfo?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VeriHeader(title: "Scans", trailing: AnyView(
                    Button { library.refresh() } label: {
                        Image(systemName: "arrow.clockwise").font(.system(size: 17)).foregroundColor(VeriColor.primary)
                    }
                ))
                if library.scans.isEmpty {
                    emptyState
                } else {
                    List {
                        ForEach(library.scans.reversed()) { scan in
                            NavigationLink(value: scan) { ScanRow(scan: scan) }
                                .listRowBackground(VeriColor.cardFill)
                                .swipeActions(edge: .trailing) {
                                    Button(role: .destructive) { pendingDelete = scan } label: { Label("Delete", systemImage: "trash") }
                                }
                        }
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                }
            }
            .background(VeriColor.background)
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: ScanInfo.self) { ScanDetailScreen(scan: $0) }
            .confirmationDialog("Delete this scan?", isPresented: Binding(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
            ), presenting: pendingDelete) { scan in
                Button("Delete \(scan.label)", role: .destructive) { library.delete(scan); pendingDelete = nil }
                Button("Cancel", role: .cancel) { pendingDelete = nil }
            } message: { _ in
                Text("Removes the scan, its 3D model and its results from this phone. This cannot be undone.")
            }
        }
        .onAppear { library.refresh() }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "square.stack.3d.up").font(.system(size: 44)).foregroundColor(VeriColor.nickel)
            Text("No scans yet").font(.system(size: 17, weight: .semibold)).foregroundColor(VeriColor.textPrimary)
            Text("Open Capture, press Record and move slowly over the scene, then Stop.")
                .font(.system(size: 13)).foregroundColor(VeriColor.nickel)
                .multilineTextAlignment(.center).padding(.horizontal, 40)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}

struct ScanRow: View {
    @EnvironmentObject private var library: ScanLibrary
    let scan: ScanInfo

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let image = library.thumbnails[scan.id] {
                    Image(uiImage: image).resizable().scaledToFill()
                } else {
                    Image(systemName: "video.fill").font(.system(size: 20)).foregroundColor(VeriColor.nickel)
                }
            }
            .frame(width: 56, height: 64)
            .background(VeriColor.inputFill)
            .cornerRadius(10)
            .clipped()
            VStack(alignment: .leading, spacing: 4) {
                Text(scan.label).font(.system(size: 15, weight: .semibold)).foregroundColor(VeriColor.textPrimary)
                Text("\(scan.frameCount) frames · \(Formatters.duration(scan.durationSeconds)) · \(Formatters.date(scan.createdAt))")
                    .font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                ScanStatusChip(scan: scan)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }
}

struct ScanStatusChip: View {
    @EnvironmentObject private var library: ScanLibrary
    let scan: ScanInfo

    var body: some View {
        if let d = scan.latestDetection {
            let vs = library.scans.first { $0.id == d.beforeID }?.label ?? "a deleted scan"
            if d.framesWithPose == 0 {
                VeriChip(text: "Checked against \(vs) · couldn't line up any frame", bg: VeriColor.inputFill, fg: VeriColor.nickel)
            } else if d.changedPixels > 0 {
                VeriChip(text: String(format: "Checked against %@ · %.0f%% changed", vs, 100 * d.changedFraction),
                         bg: VeriColor.failChipBg, fg: VeriColor.failChipFg)
            } else {
                VeriChip(text: "Checked against \(vs) · no changes", bg: VeriColor.passChipBg, fg: VeriColor.passChipFg)
            }
        } else if scan.hasReference {
            VeriChip(text: "3D model ready · can be the before-scan", bg: VeriColor.infoBannerBg, fg: VeriColor.blue90)
        } else {
            VeriChip(text: "Not checked", bg: VeriColor.inputFill, fg: VeriColor.nickel)
        }
    }
}

/// One scan's details and its saved results.
struct ScanDetailScreen: View {
    @EnvironmentObject private var library: ScanLibrary
    @EnvironmentObject private var builder: ReferenceBuilder
    @State private var showBuild = false
    @Environment(\.dismiss) private var dismiss
    let scan: ScanInfo
    @State private var shown: DetectionResult?

    var body: some View {
        VStack(spacing: 0) {
            VeriHeader(title: scan.label, onBack: { dismiss() })
            List {
                Section("Scan") {
                    row("Frames", "\(scan.frameCount)")
                    row("Duration", Formatters.duration(scan.durationSeconds))
                    row("Recorded", Formatters.date(scan.createdAt))
                    row("Device", scan.deviceModel)
                    row("3D model", scan.hasReference ? "On this phone" : "Not built")
                }
                let runs = ((try? DetectionSummary.all(in: scan.url)) ?? []).sorted { $0.finishedAt > $1.finishedAt }
                if !runs.isEmpty {
                    Section("Results") {
                        ForEach(runs, id: \.beforeID) { run in
                            let before = library.scans.first { $0.id == run.beforeID }?.label ?? "a deleted scan"
                            Button {
                                shown = DetectionResult.load(run, beforeLabel: before, afterLabel: scan.label, afterURL: scan.url)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Against \(before)").foregroundColor(VeriColor.textPrimary)
                                        Text(String(format: "%.0f%% changed · camera placed in %d of %d frames · %@",
                                                    100 * run.changedFraction, run.framesWithPose, run.frames, Formatters.date(run.finishedAt)))
                                            .font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right").foregroundColor(VeriColor.nickel)
                                }
                            }
                        }
                    }
                }
                if scan.hasReference {
                    Section {
                        NavigationLink {
                            ModelViewerScreen(scan: scan)
                        } label: {
                            Label("View 3D model", systemImage: "cube.transparent")
                        }
                    }
                    Section {
                        Button(library.before?.id == scan.id ? "This is the before-scan" : "Use as the before-scan") {
                            library.beforeID = scan.id
                        }
                        .disabled(library.before?.id == scan.id)
                    }
                }
                Section {
                    Button(scan.hasReference ? "Rebuild 3D model on this phone" : "Build 3D model on this phone") {
                        builder.start(scan: scan)
                        showBuild = true
                    }
                    .disabled(builder.running || scan.frameCount < 20)
                } footer: {
                    Text(scan.frameCount < 20 ? "At least 20 frames are needed." :
                         "About 1–2 minutes. On the Mac instead: tools/build_reference.sh \(scan.id)")
                        .textSelection(.enabled)
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .background(VeriColor.background)
        .toolbar(.hidden, for: .navigationBar)
        .fullScreenCover(isPresented: $showBuild, onDismiss: { library.refresh() }) {
            BuildSheet(builder: builder) { showBuild = false }
        }
        .sheet(item: $shown) { result in
            NavigationStack {
                VStack(spacing: 0) {
                    VeriHeader(title: "Results", trailing: AnyView(Button("Close") { shown = nil }.foregroundColor(VeriColor.primary)))
                    ResultsView(result: result)
                }
                .background(VeriColor.background)
            }
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack { Text(title).foregroundColor(VeriColor.nickel); Spacer(); Text(value).foregroundColor(VeriColor.textPrimary) }
            .font(.system(size: 14))
    }
}
