import OSCDCore
import SwiftUI

/// Arrows (moved), boxes (added / removed) and dotted circles (altered) for an `ObjectReport`, projected with a
/// camera pose into an image shown aspect-fit in this view. The image is `intrinsics.width × height` in the sensor's
/// orientation, shown as `orientation` says. Does not know about `PreviewController`; the results screen and the
/// 3D viewer reuse it.
struct ObjectOverlay: View {
    let report: ObjectReport
    let pose: Pose
    let intrinsics: Intrinsics
    let orientation: ObjectProjection.Orientation
    var highlighted: Int? = nil

    /// Below this a `moved` arrow is dashed.
    private let lowConfidence: Float = 0.55

    var body: some View {
        Canvas { context, size in
            let image = ObjectProjection.displayedSize(width: intrinsics.width, height: intrinsics.height, orientation: orientation)
            guard image.width > 0, image.height > 0, size.width > 0, size.height > 0 else { return }
            // Aspect-fit, as `Image.resizable().scaledToFit()` does.
            let scale = min(size.width / CGFloat(image.width), size.height / CGFloat(image.height))
            let origin = CGPoint(x: (size.width - CGFloat(image.width) * scale) / 2,
                                 y: (size.height - CGFloat(image.height) * scale) / 2)
            func toView(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: origin.x + (CGFloat(p.x) + 0.5) * scale, y: origin.y + (CGFloat(p.y) + 0.5) * scale)
            }
            func point(_ world: SIMD3<Float>) -> CGPoint? {
                ObjectProjection.project(world, pose: pose, intrinsics: intrinsics, orientation: orientation).map(toView)
            }
            for event in report.events {
                draw(event, in: &context, point: point)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(_ event: ObjectEvent, in context: inout GraphicsContext, point: (SIMD3<Float>) -> CGPoint?) {
        let isHint = event.kind == .altered || event.hint
        let colour = VeriColor.color(for: event.kind).opacity(isHint ? 0.7 : 1)
        let width: CGFloat = highlighted == event.id ? 5 : 2.5
        let dash: [CGFloat] = event.kind == .moved && event.confidence < lowConfidence ? [7, 5] : []
        switch event.kind {
        case .moved:
            guard let tail = event.before.flatMap({ point($0.centroid) }),
                  let head = event.after.flatMap({ point($0.centroid) }) else { return }
            strokeArrow(from: tail, to: head, colour: colour, width: width, dash: dash, in: &context)
            label(shortName(event), at: head, colour: colour, in: &context)
        case .added, .removed:
            guard let placement = event.kind == .added ? event.after : event.before,
                  let rect = extent(of: placement.bbox, point: point) else { return }
            let box = Path(roundedRect: rect, cornerRadius: 3)
            context.stroke(box, with: .color(.black.opacity(0.6)), style: StrokeStyle(lineWidth: width + 2))
            context.stroke(box, with: .color(colour), style: StrokeStyle(lineWidth: width, dash: dash))
            label(event.kind == .added ? "+ \(shortName(event))" : "− \(shortName(event))",
                  at: CGPoint(x: rect.minX, y: rect.minY - 2), anchor: .bottomLeading, colour: colour, in: &context)
        case .altered:
            guard let placement = event.after ?? event.before, let centre = point(placement.centroid) else { return }
            let box = extent(of: placement.bbox, point: point)
            let radius = max(16, box.map { max($0.width, $0.height) / 2 } ?? 0)
            let circle = Path(ellipseIn: CGRect(x: centre.x - radius, y: centre.y - radius, width: 2 * radius, height: 2 * radius))
            context.stroke(circle, with: .color(.black.opacity(0.5)), style: StrokeStyle(lineWidth: width + 2, lineCap: .round, dash: [0.1, 6]))
            context.stroke(circle, with: .color(colour), style: StrokeStyle(lineWidth: width, lineCap: .round, dash: [0.1, 6]))
            label(event.summary(), at: CGPoint(x: centre.x, y: centre.y - radius - 2), anchor: .bottom, colour: colour, in: &context)
        }
    }

    /// The projected extent of a box's 8 corners; corners behind the camera (or far outside the image) are skipped and
    /// fewer than 2 projected corners means nothing is drawn.
    private func extent(of box: AABB, point: (SIMD3<Float>) -> CGPoint?) -> CGRect? {
        var pts: [CGPoint] = []
        for i in 0..<8 {
            let c = SIMD3<Float>(i & 1 == 0 ? box.min.x : box.max.x, i & 2 == 0 ? box.min.y : box.max.y, i & 4 == 0 ? box.min.z : box.max.z)
            if let p = point(c) { pts.append(p) }
        }
        guard pts.count >= 2 else { return nil }
        let xs = pts.map(\.x), ys = pts.map(\.y)
        return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
    }

    private func strokeArrow(from tail: CGPoint, to head: CGPoint, colour: Color, width: CGFloat, dash: [CGFloat],
                             in context: inout GraphicsContext) {
        let dx = head.x - tail.x, dy = head.y - tail.y
        let length = (dx * dx + dy * dy).squareRoot()
        guard length > 1 else { return }
        let ux = dx / length, uy = dy / length
        let headLength = min(max(10, width * 4), length * 0.6)
        let base = CGPoint(x: head.x - ux * headLength, y: head.y - uy * headLength)
        var shaft = Path(); shaft.move(to: tail); shaft.addLine(to: base)
        var tip = Path()
        tip.move(to: head)
        tip.addLine(to: CGPoint(x: base.x - uy * headLength * 0.45, y: base.y + ux * headLength * 0.45))
        tip.addLine(to: CGPoint(x: base.x + uy * headLength * 0.45, y: base.y - ux * headLength * 0.45))
        tip.closeSubpath()
        // A dark halo keeps the amber legible over the red mask.
        context.stroke(shaft, with: .color(.black.opacity(0.6)), style: StrokeStyle(lineWidth: width + 2, lineCap: .round, dash: dash))
        context.stroke(tip, with: .color(.black.opacity(0.6)), style: StrokeStyle(lineWidth: 2, lineJoin: .round))
        context.stroke(shaft, with: .color(colour), style: StrokeStyle(lineWidth: width, lineCap: .round, dash: dash))
        context.fill(tip, with: .color(colour))
        // The tail sits where the object was.
        context.fill(Path(ellipseIn: CGRect(x: tail.x - 3, y: tail.y - 3, width: 6, height: 6)), with: .color(colour))
    }

    private func label(_ text: String, at p: CGPoint, anchor: UnitPoint = .topLeading, colour: Color, in context: inout GraphicsContext) {
        let resolved = context.resolve(Text(text).font(.system(size: 11, weight: .semibold)).foregroundColor(.white))
        let size = resolved.measure(in: CGSize(width: 220, height: 40))
        var origin = CGPoint(x: p.x + 4, y: p.y + 4)
        if anchor == .bottomLeading { origin = CGPoint(x: p.x, y: p.y - size.height - 3) }
        if anchor == .bottom { origin = CGPoint(x: p.x - size.width / 2, y: p.y - size.height - 3) }
        let bg = CGRect(x: origin.x - 3, y: origin.y - 1, width: size.width + 6, height: size.height + 2)
        context.fill(Path(roundedRect: bg, cornerRadius: 4), with: .color(.black.opacity(0.65)))
        context.stroke(Path(roundedRect: bg, cornerRadius: 4), with: .color(colour), lineWidth: 1)
        context.draw(resolved, at: origin, anchor: .topLeading)
    }

    /// "Ball" from "Ball · moved 14 cm".
    private func shortName(_ event: ObjectEvent) -> String {
        event.summary().components(separatedBy: " · ").first ?? "Object \(event.id)"
    }
}
