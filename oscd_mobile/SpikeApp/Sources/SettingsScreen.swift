import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var bench: BenchmarkRunner
    @ObservedObject var preview: PreviewController
    @AppStorage(ReferenceBuilder.Quality.defaultsKey) private var quality = ReferenceBuilder.Quality.standard.rawValue
    @AppStorage(ObjectNamer.defaultsKey) private var nameObjects = true

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
                        Picker("Model quality", selection: $quality) {
                            ForEach(ReferenceBuilder.Quality.allCases) { q in Text(q.label).tag(q.rawValue) }
                        }
                        .pickerStyle(.inline).labelsHidden()
                        Text("""
                        Used by "Build 3D model on this phone". Standard is about 1 dB sharper than Quick; High is for a scan you want to look at. \
                        Capture matters as much: move slowly, and turn slowly. Nothing is saved while the phone turns faster than 30°/s.
                        """)
                        .font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                    } header: { Text("Building a scan's 3D model") }
                    Section {
                        Toggle("Name objects", isOn: $nameObjects)
                        Text("""
                        Names a changed object ("Tennis ball") with the phone's image classifier. A name under 60 % sure is shown with a "?". \
                        Off: objects are shown as "Object 1", "Object 2".
                        """)
                        .font(.system(size: 12)).foregroundColor(VeriColor.nickel)
                    } header: { Text("Objects that changed") }
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
