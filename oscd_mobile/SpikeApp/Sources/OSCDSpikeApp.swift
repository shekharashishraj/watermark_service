import OSCDCore
import SwiftUI

@main
struct OSCDSpikeApp: App {
    @StateObject private var library = ScanLibrary()
    @StateObject private var capture = CaptureController()
    @StateObject private var bench = BenchmarkRunner()
    @StateObject private var preview = PreviewController()
    @StateObject private var runner = DetectionRunner()
    @StateObject private var builder = ReferenceBuilder()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView(capture: capture, bench: bench, preview: preview, runner: runner)
                .environmentObject(library)
                .environmentObject(builder)
                .preferredColorScheme(.light)        // the Romulus look is light-only
        }
        .onChange(of: scenePhase) { _, phase in
            // A run that ends while active (crash, memory kill, watchdog) leaves the marker set; a run that
            // was backgrounded first (swiped away, or evicted later) clears it. Read at the next launch.
            let d = UserDefaults.standard
            switch phase {
            case .active:
                if d.bool(forKey: Self.activeMarker) { AppLog.write("app: the previous run ended while active (crash, memory kill or watchdog)", sync: true) }
                d.set(true, forKey: Self.activeMarker)
            case .background:
                d.set(false, forKey: Self.activeMarker)
                AppLog.write("app: background", sync: true)
            default: break
            }
        }
    }

    static let activeMarker = "runEndedWhileActive"
}

/// Scans / Capture / Detect / Settings, as in Romulus.
struct RootView: View {
    enum Tab: Hashable { case scans, capture, detect, settings }

    @EnvironmentObject private var library: ScanLibrary
    @EnvironmentObject private var builder: ReferenceBuilder
    @ObservedObject var capture: CaptureController
    @ObservedObject var bench: BenchmarkRunner
    @ObservedObject var preview: PreviewController
    @ObservedObject var runner: DetectionRunner
    // Launch arguments `-detectBefore <scan folder> -detectAfter <scan folder>` open Detect and run that
    // pair (devicectl ... process launch ... -- -detectBefore X -detectAfter Y): a rehearsal without tapping.
    @State private var tab: Tab = UserDefaults.standard.string(forKey: "detectAfter") != nil ? .detect
        : UserDefaults.standard.string(forKey: "tab") == "capture" ? .capture : .scans     // `-tab capture` for checks
    @State private var showBuild = false
    @State private var viewScan: ScanInfo?
    @State private var viewReport: ObjectReport?
    @State private var showLive = false

    var body: some View {
        TabView(selection: $tab) {
            ScansScreen(runner: runner)
                .tabItem { Label("Scans", systemImage: "square.stack.3d.up") }
                .tag(Tab.scans)
            CaptureView(controller: capture)
                .onAppear { capture.start() }
                .onDisappear { capture.pause(); library.refresh() }
                .tabItem { Label("Capture", systemImage: "camera.viewfinder") }
                .tag(Tab.capture)
            DetectScreen(runner: runner)
                .tabItem { Label("Detect", systemImage: "viewfinder.trianglebadge.exclamationmark") }
                .tag(Tab.detect)
            SettingsScreen(bench: bench, preview: preview)
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
                .tag(Tab.settings)
        }
        .tint(VeriColor.primary)
        .onAppear {
            library.refresh()
            AppLog.write("app: launched, camera=\(AppLog.cameraAuthorization())")
            // `-buildReference <scan folder>` builds that scan's 3D model at launch (a check without tapping).
            // `-live YES` opens the Live view and starts it with the current before-scan (checks without tapping).
            showLive = UserDefaults.standard.bool(forKey: "live")
            // `-viewModel <scan folder>` opens that scan's 3D model viewer at launch.
            if let id = UserDefaults.standard.string(forKey: "viewModel") {
                viewScan = library.scans.first { $0.id == id && $0.hasReference }
            }
            // `-viewObjects <after-scan folder>` opens the 3D viewer on the before-scan of that scan's latest saved
            // detection, with that run's object events drawn over the model.
            if let id = UserDefaults.standard.string(forKey: "viewObjects") { openObjects(afterID: id) }
            if let id = UserDefaults.standard.string(forKey: "buildReference"), !builder.running,
               let scan = library.scans.first(where: { $0.id == id }) {
                builder.start(scan: scan)
                showBuild = true
            }
        }
        .fullScreenCover(isPresented: $showBuild, onDismiss: { library.refresh() }) {
            BuildSheet(builder: builder) { showBuild = false }
        }
        .fullScreenCover(item: $viewScan) { scan in ModelViewerScreen(scan: scan, report: viewReport) }
        .fullScreenCover(isPresented: $showLive) { PreviewView(controller: preview, autoStart: true) }
        .onChange(of: tab) { _, t in AppLog.write("app: tab \(t)") }
    }

    private func openObjects(afterID: String) {
        guard let after = library.scans.first(where: { $0.id == afterID }),
              let latest = ((try? DetectionSummary.all(in: after.url)) ?? []).max(by: { $0.finishedAt < $1.finishedAt }),
              let before = library.scans.first(where: { $0.id == latest.beforeID && $0.hasReference }) else {
            AppLog.write("viewer: no saved detection with a before-scan model for \(afterID)")
            return
        }
        viewReport = ["B", "A"].lazy.compactMap { ObjectReport.load(in: after.url, beforeID: latest.beforeID, method: $0) }.first
        viewScan = before
    }
}
