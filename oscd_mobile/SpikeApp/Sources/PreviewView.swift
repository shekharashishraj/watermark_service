import RealityKit
import SwiftUI

struct ARViewHost: UIViewRepresentable {
    let arView: ARView
    func makeUIView(context: Context) -> ARView { arView }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// Live camera behind; on top, the latest processed frame with O-SCD's per-frame change mask in red
/// (dimmed = outside the judged table area) and the reference render at the O-SCD pose.
struct PreviewView: View {
    @ObservedObject var controller: PreviewController

    var body: some View {
        ZStack(alignment: .top) {
            ARViewHost(arView: controller.arView).ignoresSafeArea()
            VStack(spacing: 8) {
                HStack(alignment: .top, spacing: 8) {
                    panel(controller.overlay, title: "Changes (O-SCD fusion)")
                    panel(controller.reference, title: "Reference at O-SCD pose")
                }
                if !controller.hud.isEmpty {
                    Text(controller.hud)
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
                HStack(spacing: 16) {
                    Button(controller.running ? "Stop" : "Start") { controller.toggle() }
                        .buttonStyle(.borderedProminent)
                        .tint(controller.running ? .red : .accentColor)
                    if controller.running {
                        Button("Refine") { controller.refine() }.buttonStyle(.bordered)
                    }
                }
                .padding(.bottom, 24)
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
        }
        .onAppear { controller.startSession() }
        .onDisappear { controller.stopSession() }
    }

    @ViewBuilder private func panel(_ image: UIImage?, title: String) -> some View {
        VStack(spacing: 2) {
            if let image {
                Image(uiImage: image).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                RoundedRectangle(cornerRadius: 6).fill(.black.opacity(0.4)).aspectRatio(3.0 / 4.0, contentMode: .fit)
            }
            Text(title).font(.caption2).foregroundStyle(.white).shadow(radius: 2)
        }
    }
}
