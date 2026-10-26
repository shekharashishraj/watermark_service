import RealityKit
import SwiftUI

struct CaptureView: View {
    @ObservedObject var controller: CaptureController
    @State private var appearToken = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            SharedARViewSlot(onTap: nil).id(appearToken).ignoresSafeArea()
            VStack(spacing: 12) {
                Text(controller.status)
                    .font(.callout)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                Text("Frames: \(controller.keyframeCount)").monospacedDigit()
                Button(controller.recording ? "Stop" : "Record") { controller.toggleRecording() }
                    .buttonStyle(.borderedProminent)
                    .tint(controller.recording ? .red : .accentColor)
                    .disabled(controller.saving)
            }
            .padding(.bottom, 32)
        }
        .onAppear { appearToken += 1 }
    }
}
