// C API of Brush's `brush-c` crate (apps/brush-c/src/lib.rs), hand-written to match its #[repr(C)] types.
// The library is built from the remus checkout:
//   cargo build --release --target aarch64-apple-ios -p brush-c   →  target/aarch64-apple-ios/release/libbrush_c.a
#ifndef BRUSH_C_H
#define BRUSH_C_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BrushTrainExitCode { BrushTrainSuccess = 0, BrushTrainError = 1 } BrushTrainExitCode;

// #[repr(C)] enum ProgressMessage { NewProcess, Training { iter: u32 }, DoneTraining }
// Rust lays a data-carrying repr(C) enum out as { tag (C int), union { per-variant structs } }.
typedef enum BrushProgressTag { BrushProgressNewProcess = 0, BrushProgressTraining = 1, BrushProgressDoneTraining = 2 } BrushProgressTag;
typedef struct BrushProgressMessage {
    BrushProgressTag tag;
    union {
        struct { uint32_t iter; } training;
    };
} BrushProgressMessage;

typedef struct BrushTrainOptions {
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    const char* output_path;     // NULL = cwd
    uint32_t max_splats;         // 0 = Brush's default (10M); phones need a ceiling (dinovi's models are ~1M)
    uint32_t sh_degree;          // 0 = Brush's default (3); dinovi exports SH 2
} BrushTrainOptions;

typedef void (*BrushProgressCallback)(BrushProgressMessage progress_message, void* user_data);

// Trains from a COLMAP / Nerfstudio dataset folder and writes export_{iter}.ply into output_path.
// Blocks the calling thread until training is complete. Safe to call from a non-main thread.
BrushTrainExitCode train_and_save(const char* dataset_path, const BrushTrainOptions* options,
                                  BrushProgressCallback progress_callback, void* user_data);

#ifdef __cplusplus
}
#endif
#endif
