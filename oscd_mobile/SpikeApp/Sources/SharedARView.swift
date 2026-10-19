import RealityKit
import SwiftUI

/// The app's only ARView. RealityKit drew the camera feed in just one of two ARView instances: with a
/// second one created for the Live view, the Capture screen stayed black although ARKit delivered
/// frames (first-frame mean luma 107/255, 2026-10-19). Capture and Live view never show at once, so
/// they share this view; whichever screen appears runs its own session configuration on it.
enum SharedAR {
    static let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
}

/// Hosts the shared ARView. Recreate it when its screen appears (`.id(token)`), so the view is moved
/// back from the other screen. `onTap` receives taps in view coordinates.
struct SharedARViewSlot: UIViewRepresentable {
    var onTap: ((CGPoint) -> Void)?

    func makeUIView(context: Context) -> UIView {
        let container = UIView()
        let view = SharedAR.view
        view.removeFromSuperview()
        view.frame = container.bounds
        view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        container.addSubview(view)
        view.gestureRecognizers?.forEach { view.removeGestureRecognizer($0) }
        if onTap != nil {
            view.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tap(_:))))
        }
        return container
    }

    func updateUIView(_ uiView: UIView, context: Context) { context.coordinator.onTap = onTap }

    func makeCoordinator() -> Coordinator { Coordinator(onTap: onTap) }

    final class Coordinator: NSObject {
        var onTap: ((CGPoint) -> Void)?
        init(onTap: ((CGPoint) -> Void)?) { self.onTap = onTap }
        @objc func tap(_ g: UITapGestureRecognizer) { onTap?(g.location(in: g.view)) }
    }
}
