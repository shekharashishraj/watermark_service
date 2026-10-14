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
