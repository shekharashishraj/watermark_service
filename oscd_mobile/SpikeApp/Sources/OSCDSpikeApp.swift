import SwiftUI

@main
struct OSCDSpikeApp: App {
    @StateObject private var library = ScanLibrary()
    @StateObject private var capture = CaptureController()
    @StateObject private var bench = BenchmarkRunner()
    @StateObject private var preview = PreviewController()
    @StateObject private var runner = DetectionRunner()

    var body: some Scene {
        WindowGroup {
            RootView(capture: capture, bench: bench, preview: preview, runner: runner)
                .environmentObject(library)
        }
    }
}

/// Scans / Capture / Detect / Settings, as in Romulus.
struct RootView: View {
    enum Tab: Hashable { case scans, capture, detect, settings }

    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var capture: CaptureController
    @ObservedObject var bench: BenchmarkRunner
    @ObservedObject var preview: PreviewController
    @ObservedObject var runner: DetectionRunner
    // Launch arguments `-detectBefore <scan folder> -detectAfter <scan folder>` open Detect and run that
    // pair (devicectl ... process launch ... -- -detectBefore X -detectAfter Y): a rehearsal without tapping.
    @State private var tab: Tab = UserDefaults.standard.string(forKey: "detectAfter") == nil ? .scans : .detect

    var body: some View {
        TabView(selection: $tab) {
            ScansScreen(runner: runner)
                .tabItem { Label("Scans", systemImage: "square.stack.3d.up") }
                .tag(Tab.scans)
            CaptureView(controller: capture)
                .onAppear { capture.resume() }
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
        .onAppear { library.refresh() }
    }
}
