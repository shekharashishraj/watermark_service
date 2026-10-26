import OSCDCore
import SwiftUI

extension VeriColor {
    static func color(for kind: ObjectEvent.Kind) -> Color {
        switch kind {
        case .moved: return moved
        case .added: return added
        case .removed: return removed
        case .altered: return altered
        }
    }
}

/// One row per event, in the report's order (first seen first) so rows do not jump between updates.
/// Tapping a row highlights its arrow or box in `ObjectOverlay` (tap again to clear). Hints (`altered`) are greyed.
struct ObjectEventList: View {
    let report: ObjectReport
    var compact: Bool
    @Binding var highlighted: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 2 : 6) {
            ForEach(Array(report.events.enumerated()), id: \.element.id) { _, event in
                row(event)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func row(_ event: ObjectEvent) -> some View {
        let isHint = event.kind == .altered || event.hint
        return HStack(spacing: 8) {
            Circle()
                .fill(VeriColor.color(for: event.kind).opacity(isHint ? 0.6 : 1))
                .frame(width: compact ? 8 : 10, height: compact ? 8 : 10)
            Text(event.summary())
                .font(compact ? .caption : .subheadline)
                .foregroundStyle(isHint ? .secondary : .primary)
                .fontWeight(highlighted == event.id ? .bold : .regular)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.vertical, compact ? 1 : 3)
        .contentShape(Rectangle())
        .onTapGesture { highlighted = highlighted == event.id ? nil : event.id }
    }
}
