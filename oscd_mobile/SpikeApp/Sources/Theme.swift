import SwiftUI

/// Veripheye colours and the shared pieces of the Romulus look (veripheye-romulus-ios Theme/VeriTheme.swift),
/// with the system font in place of Poppins.
enum VeriColor {
    static let primary = Color(hex: 0x47ACDA)          // VeriPhEye sky blue
    static let blue90 = Color(hex: 0x275E78)
    static let textPrimary = Color(hex: 0x1B1B1F)
    static let nickel = Color(hex: 0x6C6F7A)
    static let border = Color(hex: 0xDEDEDE)
    static let background = Color(hex: 0xF8F8F8)
    static let cardFill = Color.white
    static let inputFill = Color(hex: 0xECF0F3)
    static let disabledFill = Color(hex: 0xA4B0C3)
    static let mediumGrey = Color(hex: 0xD7D7D7)
    static let danger = Color(hex: 0xFB5454)
    static let darkGreen = Color(red: 21 / 255, green: 197 / 255, blue: 98 / 255)
    static let passChipBg = Color(hex: 0xE8F5E9)
    static let passChipFg = Color(hex: 0x2E7D32)
    static let failChipBg = Color(hex: 0xFFEBEE)
    static let failChipFg = Color(hex: 0xC62828)
    static let infoBannerBg = Color(red: 223 / 255, green: 241 / 255, blue: 253 / 255)
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255)
    }
}

/// Screen header: title on the left (or centred with a back chevron), optional trailing control.
struct VeriHeader: View {
    let title: String
    var onBack: (() -> Void)?
    var trailing: AnyView?

    var body: some View {
        HStack(spacing: 10) {
            if let onBack {
                Button(action: onBack) {
                    Image(systemName: "chevron.left").font(.system(size: 20, weight: .medium)).foregroundColor(VeriColor.textPrimary)
                }
            }
            Text(title)
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(VeriColor.textPrimary)
                .frame(maxWidth: .infinity, alignment: onBack == nil ? .leading : .center)
            if let trailing { trailing } else if onBack != nil { Image(systemName: "chevron.left").opacity(0) }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

/// Full-width primary button.
struct VeriButton: View {
    let title: String
    var systemImage: String?
    var disabled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title).font(.system(size: 17, weight: .semibold))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(13)
            .background(disabled ? VeriColor.disabledFill : VeriColor.primary)
            .cornerRadius(10)
        }
        .disabled(disabled)
    }
}

struct VeriChip: View {
    let text: String
    let bg: Color
    let fg: Color

    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold))
            .foregroundColor(fg)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(bg)
            .cornerRadius(8)
    }
}

enum Formatters {
    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return String(format: "%02d:%02d", s / 60, s % 60)
    }

    static func date(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "MMM d, HH:mm"
        return f.string(from: date)
    }

    static func ms(_ ms: Double) -> String {
        ms >= 1000 ? String(format: "%.1f s", ms / 1000) : String(format: "%.0f ms", ms)
    }
}
