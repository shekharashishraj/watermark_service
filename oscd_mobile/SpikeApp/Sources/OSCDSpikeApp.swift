import SwiftUI

@main
struct OSCDSpikeApp: App {
    @StateObject private var capture = CaptureController()
    @StateObject private var bench = BenchmarkRunner()
    @StateObject private var preview = PreviewController()

    var body: some Scene {
        WindowGroup {
            TabView {
                CaptureView(controller: capture)
                    .onAppear { capture.resume() }
                    .onDisappear { capture.pause() }
                    .tabItem { Label("Capture", systemImage: "camera.viewfinder") }
                PreviewView(controller: preview).tabItem { Label("Preview", systemImage: "eye") }
                BenchmarkView(runner: bench).tabItem { Label("Benchmark", systemImage: "speedometer") }
            }
        }
    }
}
