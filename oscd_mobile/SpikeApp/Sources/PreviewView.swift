import OSCDCore
import RealityKit
import SwiftUI

/// Live camera behind; on top, the latest processed frame with the changes in red (dimmed = not judged)
/// and the before-scan's 3D model rendered from where the camera is.
struct PreviewView: View {
    @EnvironmentObject private var library: ScanLibrary
    @ObservedObject var controller: PreviewController
    var autoStart = false
    @State private var showTimings = false
    @State private var appearToken = 0
    @State private var highlighted: Int?

    var body: some View {
        ZStack(alignment: .top) {
            SharedARViewSlot(onTap: nil).id(appearToken).ignoresSafeArea()
            VStack(spacing: 8) {
                beforePicker
                HStack(alignment: .top, spacing: 8) {
                    panel(controller.overlay, title: "Changes", showObjects: true)
                    panel(controller.reference, title: "\(library.before?.label ?? "Before-scan") from here")
                }
                if !controller.headline.isEmpty {
                    Text(controller.headline)
                        .font(.footnote)
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                }
                if let objects = controller.objects, !objects.events.isEmpty {
                    ScrollView {
                        ObjectEventList(report: objects, compact: true, highlighted: $highlighted)
                            .padding(8)
                    }
                    .frame(maxHeight: 96)
                    .fixedSize(horizontal: false, vertical: true)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                }
                if showTimings, !controller.details.isEmpty {
                    Text(controller.details)
                        .font(.caption2.monospaced())
                        .padding(6)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
                }
                Spacer()
                Text(controller.status)
                    .font(.callout)
                    .multilineTextAlignment(.center)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                HStack(spacing: 12) {
                    Button(controller.running ? "Stop" : "Start") {
                        if controller.running { controller.stop() } else if let before = library.before { controller.start(before: before) }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(controller.running ? .red : VeriColor.primary)
                    .disabled(!controller.running && library.before == nil)
                    Toggle("Timings", isOn: $showTimings).toggleStyle(.button).buttonStyle(.bordered)
                }
                .padding(.bottom, 24)
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
        }
        .onAppear {
            appearToken += 1
            controller.startSession()
            if autoStart, !controller.running, let before = library.before { controller.start(before: before) }
            if library.before == nil { controller.status = "No scan has a 3D model yet. See Settings › Building a scan's 3D model." }
        }
        .onDisappear { controller.stopSession() }
    }

    /// Which scan's 3D model the camera is compared with (only scans with a model; locked while running).
    private var beforePicker: some View {
        Menu {
            ForEach(library.scans.filter(\.hasReference).reversed()) { scan in
                Button {
                    library.beforeID = scan.id
                } label: {
                    Label("\(scan.label) · \(scan.frameCount) frames · \(Formatters.date(scan.createdAt))",
                          systemImage: scan.id == library.before?.id ? "checkmark" : "")
                }
            }
        } label: {
            HStack(spacing: 6) {
                Text("Comparing with").foregroundStyle(.secondary)
                Text(library.before.map { "\($0.label) · \(Formatters.date($0.createdAt))" } ?? "no scan with a 3D model")
                    .fontWeight(.semibold)
                if !controller.running { Image(systemName: "chevron.down") }
            }
            .font(.footnote)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
        .disabled(controller.running)
    }

    @ViewBuilder private func panel(_ image: UIImage?, title: String, showObjects: Bool = false) -> some View {
        VStack(spacing: 2) {
            if let image {
                Image(uiImage: image).resizable().scaledToFit()
                    .overlay {
                        // The latest placed frame's pose; the image shown is that frame (rotated .right), so the arrows
                        // line up with the red blob.
                        if showObjects, let report = controller.objects, let pose = controller.lastPose,
                           let intrinsics = controller.lastIntrinsics {
                            ObjectOverlay(report: report, pose: pose, intrinsics: intrinsics, orientation: .sensorRight,
                                          highlighted: highlighted)
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                RoundedRectangle(cornerRadius: 6).fill(.black.opacity(0.4)).aspectRatio(3.0 / 4.0, contentMode: .fit)
            }
            Text(title).font(.caption2).foregroundStyle(.white).shadow(radius: 2)
        }
    }
}
