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

    var body: some Scene {
        WindowGroup {
            RootView(capture: capture, bench: bench, preview: preview, runner: runner)
                .environmentObject(library)
                .environmentObject(builder)
                .preferredColorScheme(.light)        // the Romulus look is light-only
        }
    }
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
            // `-viewModel <scan folder>` opens that scan's 3D model viewer at launch.
            if let id = UserDefaults.standard.string(forKey: "viewModel") {
                viewScan = library.scans.first { $0.id == id && $0.hasReference }
            }
            if let id = UserDefaults.standard.string(forKey: "buildReference"), !builder.running,
               let scan = library.scans.first(where: { $0.id == id }) {
                builder.start(scan: scan)
                showBuild = true
            }
        }
        .fullScreenCover(isPresented: $showBuild, onDismiss: { library.refresh() }) {
            BuildSheet(builder: builder) { showBuild = false }
        }
        .fullScreenCover(item: $viewScan) { scan in ModelViewerScreen(scan: scan) }
        .onChange(of: tab) { _, t in AppLog.write("app: tab \(t)") }
    }
}
