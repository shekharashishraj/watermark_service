import SwiftUI

/// Progress of an on-phone 3D model build, in the Romulus processing-sheet style.
struct BuildSheet: View {
    @ObservedObject var builder: ReferenceBuilder
    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            VeriHeader(title: "Building 3D model", trailing: AnyView(
                Button(builder.running ? "Cancel" : "Close") { builder.running ? builder.cancel() : onClose() }
                    .font(.system(size: 15, weight: .medium)).foregroundColor(VeriColor.primary)
            ))
            List {
                Section {
                    Text(builder.scanLabel).font(.system(size: 16, weight: .semibold))
                    row("Prepare the scan (images, LiDAR points)", state: state(0))
                    row("Train the 3D model (\(ReferenceBuilder.iterations.formatted()) steps, on this phone)", state: state(1))
                    row("Save", state: state(2))
                }
                if case .done(let seconds, let train, let splats) = builder.stage {
                    Section("Result") {
                        kv("Gaussians", splats.formatted())
                        kv("Training", String(format: "%.0f s", train))
                        kv("Total", String(format: "%.0f s", seconds))
                    }
                    Section { Text("The scan can now be the before-scan in Detect and Live view.").font(.system(size: 13)).foregroundColor(VeriColor.nickel) }
                }
                if case .failed(let message) = builder.stage {
                    Section("Stopped") { Text(message).foregroundColor(VeriColor.danger).font(.system(size: 13)) }
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .background(VeriColor.background)
        .interactiveDismissDisabled(builder.running)
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
    }

    private enum RowState { case pending, active(String, Double?), done(String) }

    private func state(_ i: Int) -> RowState {
        switch builder.stage {
        case .idle: return .pending
        case .exporting: return i == 0 ? .active("", nil) : .pending
        case .training(let iter, let total, let splats, let left, let thermal):
            if i == 0 { return .done(String(format: "%.1f s", builder.exportSeconds)) }
            if i > 1 { return .pending }
            var detail = "\(iter) / \(total) · \(splats.formatted()) Gaussians"
            if left.isFinite { detail += " · \(Formatters.duration(left)) left" }
            if thermal >= 2 { detail += " · phone hot, slowing down" }
            return .active(detail, total > 0 ? Double(iter) / Double(total) : nil)
        case .saving: return i < 2 ? .done("") : .active("", nil)
        case .done: return .done("")
        case .failed: return .pending
        }
    }

    private func row(_ title: String, state: RowState) -> some View {
        HStack(alignment: .top, spacing: 10) {
            switch state {
            case .pending: Image(systemName: "circle").foregroundColor(VeriColor.mediumGrey)
            case .active: ProgressView().controlSize(.small)
            case .done: Image(systemName: "checkmark.circle.fill").foregroundColor(VeriColor.darkGreen)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 14))
                switch state {
                case .active(let d, let p):
                    if !d.isEmpty { Text(d).font(.system(size: 11)).foregroundColor(VeriColor.nickel) }
                    if let p { ProgressView(value: p).tint(VeriColor.primary) }
                case .done(let d): if !d.isEmpty { Text(d).font(.system(size: 11)).foregroundColor(VeriColor.nickel) }
                case .pending: EmptyView()
                }
            }
        }
    }

    private func kv(_ k: String, _ v: String) -> some View {
        HStack { Text(k).foregroundColor(VeriColor.nickel); Spacer(); Text(v) }.font(.system(size: 14))
    }
}
