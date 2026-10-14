import SwiftUI

struct BenchmarkView: View {
    @ObservedObject var runner: BenchmarkRunner

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button(runner.running ? "Running…" : "Run benchmarks") { runner.run() }
                        .disabled(runner.running)
                    if !runner.message.isEmpty {
                        Text(runner.message).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                Section("Median milliseconds") {
                    ForEach(runner.rows) { row in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(row.stage).font(.body.monospaced())
                                Text(row.note).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(String(format: "%.1f", row.medianMs)).monospacedDigit()
                        }
                    }
                }
            }
            .navigationTitle("Spike benchmarks")
        }
    }
}
