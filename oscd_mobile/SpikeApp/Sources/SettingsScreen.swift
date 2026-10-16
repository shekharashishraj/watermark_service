import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var bench: BenchmarkRunner
    @ObservedObject var preview: PreviewController

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VeriHeader(title: "Settings")
                List {
                    Section("Tools") {
                        NavigationLink { PreviewView(controller: preview) } label: {
                            Label("Live view", systemImage: "eye")
                        }
                        NavigationLink { BenchmarkView(runner: bench) } label: {
                            Label("Stage benchmark", systemImage: "speedometer")
                        }
                    }
                    Section {
                        Text("""
                        1. Record the before-scan in Capture.
                        2. On the Mac, with the phone connected:
                           tools/build_reference.sh <scan folder>
                           It copies the scan off the phone, trains its 3D model (about 1 minute) and copies the model back.
                        3. The scan then shows "3D model ready" and can be picked as the before-scan in Detect.
                        """)
                        .font(.system(size: 12)).foregroundColor(VeriColor.nickel).textSelection(.enabled)
                    } header: { Text("Building a scan's 3D model") }
                    Section("About") {
                        row("Before-scan", library.before?.label ?? "none yet")
                        row("Method A", "O-SCD: XFeat, PnP, SAM2.1, change fusion")
                        row("Device", CaptureController.deviceModel())
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
            }
            .background(VeriColor.background)
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack { Text(title).foregroundColor(VeriColor.nickel); Spacer(); Text(value).foregroundColor(VeriColor.textPrimary) }
            .font(.system(size: 14))
    }
}
