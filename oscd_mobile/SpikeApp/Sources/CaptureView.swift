import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    let controller: CaptureController

    func makeUIView(context: Context) -> ARView {
        let view = controller.arView
        view.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tap(_:))))
        controller.start()
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(controller: controller) }

    final class Coordinator: NSObject {
        let controller: CaptureController
        init(controller: CaptureController) { self.controller = controller }
        @objc func tap(_ gesture: UITapGestureRecognizer) { controller.placeROI(at: gesture.location(in: gesture.view)) }
    }
}

struct CaptureView: View {
    @ObservedObject var controller: CaptureController

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(controller: controller).ignoresSafeArea()
            VStack(spacing: 12) {
                Text(controller.status)
                    .font(.callout)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                Text("Frames: \(controller.keyframeCount)").monospacedDigit()
                Button(controller.recording ? "Stop" : "Record") { controller.toggleRecording() }
                    .buttonStyle(.borderedProminent)
                    .tint(controller.recording ? .red : .accentColor)
            }
            .padding(.bottom, 32)
        }
    }
}
