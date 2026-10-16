import Combine
import OSCDCore
import SwiftUI
import UIKit

/// The recorded scans in Documents/sessions, for the Scans and Detect tabs.
final class ScanLibrary: ObservableObject {
    @Published private(set) var scans: [ScanInfo] = []
    @Published private(set) var thumbnails: [String: UIImage] = [:]
    /// The scan whose 3D model Detect and the live view compare against.
    @Published var beforeID = UserDefaults.standard.string(forKey: "beforeScanID") ?? "" {
        didSet { UserDefaults.standard.set(beforeID, forKey: "beforeScanID") }
    }

    static var sessionsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("sessions")
    }

    var before: ScanInfo? {
        scans.first { $0.id == beforeID && $0.hasReference } ?? scans.last { $0.hasReference }
    }

    func refresh() {
        let list = (try? ScanCatalog.list(Self.sessionsDir)) ?? []
        scans = list
        let missing = list.filter { thumbnails[$0.id] == nil }
        guard !missing.isEmpty else { return }
        DispatchQueue.global(qos: .utility).async {
            var made: [String: UIImage] = [:]
            for scan in missing { if let t = Self.thumbnail(scan) { made[scan.id] = t } }
            DispatchQueue.main.async { self.thumbnails.merge(made) { a, _ in a } }
        }
    }

    func delete(_ scan: ScanInfo) {
        try? FileManager.default.removeItem(at: scan.url)
        thumbnails[scan.id] = nil
        refresh()
    }

    /// The scan's middle keyframe, cached as thumb.jpg in the scan folder.
    static func thumbnail(_ scan: ScanInfo) -> UIImage? {
        let url = scan.url.appendingPathComponent("thumb.jpg")
        if let data = try? Data(contentsOf: url), let image = UIImage(data: data) { return image }
        guard let store = try? SessionStore(open: scan.url), !store.manifest.keyframes.isEmpty else { return nil }
        let r = store.manifest.keyframes[store.manifest.keyframes.count / 2]
        guard let a = try? store.array(.keyframes, id: r.id, name: "rgb"),
              let cg = PreviewController.cgImage(rgba(a.uint8()), a.shape[1], a.shape[0]) else { return nil }
        let image = UIImage(cgImage: cg, scale: 1, orientation: .right)
        let small = UIGraphicsImageRenderer(size: CGSize(width: 120, height: 160)).image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: 120, height: 160))
        }
        try? small.jpegData(compressionQuality: 0.8)?.write(to: url)
        return small
    }

    static func rgba(_ rgb: [UInt8]) -> [UInt8] {
        var out = [UInt8](repeating: 255, count: rgb.count / 3 * 4)
        for i in 0..<(rgb.count / 3) { out[i * 4] = rgb[i * 3]; out[i * 4 + 1] = rgb[i * 3 + 1]; out[i * 4 + 2] = rgb[i * 3 + 2] }
        return out
    }
}
