import AVFoundation
import Foundation

/// Documents/app-log.txt: camera and screen events, for debugging on the phone (pull with devicectl).
enum AppLog {
    private static let queue = DispatchQueue(label: "oscd.applog")
    private static let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("app-log.txt")

    static func write(_ s: String) { write(s, sync: false) }

    /// The app's resident memory in MB (phys_footprint, what iOS judges memory kills by).
    static func residentMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let rc = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return rc == KERN_SUCCESS ? Int(info.phys_footprint / 1_000_000) : -1
    }

    /// `sync` lands the line on disk before returning: for lines that must survive a crash right after them.
    static func write(_ s: String, sync: Bool) {
        let line = String(format: "%.3f ", Date().timeIntervalSince1970) + s + "\n"
        let work = {
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close()
            } else {
                try? Data(line.utf8).write(to: url)
            }
        }
        if sync { queue.sync(execute: work) } else { queue.async(execute: work) }
    }

    static func cameraAuthorization() -> String {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return "authorized"
        case .denied: return "denied"
        case .restricted: return "restricted"
        case .notDetermined: return "notDetermined"
        @unknown default: return "unknown"
        }
    }
}
