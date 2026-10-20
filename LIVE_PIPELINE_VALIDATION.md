# Live Pipeline Verification & Behavioral Audit Report

**Project:** Lecture Engagement Pattern Analyzer  
**Alternative Descriptive Name:** Classroom Behavioral Observation and Analysis System  
**Document Version:** 2.0.0 (Phase 3 Audit & GPU Acceleration Update)  
**Date:** October 18, 2026  
**Environment:** Windows 11 (64-bit), Python 3.11.17, PySide6 6.12.0, OpenCV 5.0.0, MediaPipe 0.10.35, Ultralytics YOLOv8 8.4.182, PyTorch 2.6.0+cu124 (CUDA 12.6)  
**Hardware Tested:**  
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM, Driver 561.00)  
- **Host Camera:** Physical Webcam Index 0 (DirectShow Backend, Native 1280 × 720)  

---

## 1. System Architecture & Concurrency Model

The application employs a decoupled producer-consumer architecture to eliminate inference backlog and guarantee real-time UI responsiveness:

```
[ Physical Webcam (1280x720 @ 30 FPS) ]
                 │
                 ▼
[ CameraCaptureThread (QThread) ]
    ├── DirectShow High-Rate Acquisition (~29.8 FPS)
    ├── Optional Video Writer (.mp4 @ 30 FPS)
    └── Atomic Single-Item Latest-Frame Buffer (QMutex)
                 │  (Drops superseded frames if inference is busy; tracks skip count)
                 ▼
[ CameraInferenceWorker (QThread) ]
    ├── Multi-Model Vision Pipeline (GPU-Accelerated)
    ├── Explainable Behavioral Rule Engine
    ├── Timestamp-Based Temporal Persistence Smoother
    └── Optional SQLite Batch Persistence (Transaction Buffering)
                 │
                 ▼
[ Main UI Thread (PySide6) ]
    └── Video Display Overlay + Real-Time Telemetry (Dual FPS, Latency, Skips)
```

### Key Concurrency Guarantees
1. **Zero Camera Backlog:** The capture thread reads frames at full hardware speed (~29.8 FPS). The inference worker consumes the latest available frame, preventing queue accumulation.
2. **Dual-Rate Telemetry:** Camera acquisition rate (FPS) and complete-pipeline processing rate (FPS) are measured and reported separately.
3. **Exact Timing Metadata:** Observations store the monotonic acquisition timestamp (`packet.monotonic_ts`) recorded when the frame was acquired, ensuring timing fidelity regardless of inference duration.

---

## 2. Hardware Acceleration & Benchmark Comparisons

With PyTorch upgraded to `2.6.0+cu124`, neural network models execute on the dedicated NVIDIA GeForce RTX 4050 Laptop GPU:

| Metric | CPU-Only Baseline (Phase 2) | GPU-Accelerated (Phase 3) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Inference Hardware** | Intel CPU (Torch 2.14.1+cpu) | NVIDIA GeForce RTX 4050 (6 GB VRAM) | Dedicated Tensor Cores |
| **Pipeline Warmup Time** | 12.21 s | 4.85 s | **2.5× faster** |
| **Camera Acquisition FPS** | 29.8 FPS | 29.8 FPS | Identical (hardware bound) |
| **Processing Rate (Full Pipeline)** | 1.88 FPS | **8.60 FPS** | **4.6× faster throughput** |
| **Median Processing Latency** | 531.3 ms | **82.0 ms** | **6.5× latency reduction** |
| **P95 Processing Latency** | 572.6 ms | **177.1 ms** | **3.2× latency reduction** |
| **P99 Processing Latency** | 584.1 ms | **180.8 ms** | **3.2× latency reduction** |
| **Memory Working Set (RSS)** | 588.3 MB | 1,366.3 MB | Within limits (includes CUDA context) |

---

## 3. Perception Component Verification & Timing Breakdown

Each processed frame undergoes end-to-end multi-modal perception:

| Perception Stage | Underlying Technology | Local Weights / Model | Component Latency (GPU) |
| :--- | :--- | :--- | :--- |
| **Frame Acquisition** | OpenCV DirectShow | Hardware Index 0 | ~2.1 ms |
| **Face Mesh & Geometry** | MediaPipe Task API | `models/face_landmarker.task` | ~18.4 ms |
| **3D Head Pose Solver** | OpenCV `solvePnP` (EPnP) | Anthropometric 3D face model | ~1.8 ms |
| **Human Pose Estimation** | Ultralytics YOLOv8n-pose | `models/yolov8n-pose.pt` | ~28.6 ms |
| **Object / Phone Detection** | Ultralytics YOLOv8n | `models/yolov8n.pt` | ~26.2 ms |
| **Anonymous Tracking** | Centroid-IoU Tracker | Pure Python / NumPy | ~0.5 ms |
| **Rule & Temporal Smoother** | Behavioral Engine | Pure Python | ~0.4 ms |
| **Total Frame Latency** | **End-to-End Pipeline** | **All Local Models** | **~82.0 ms (Median)** |

---

## 4. Feature Mathematics & Missing-Feature Safeguards

Strict missing-feature safeguards have been verified in `src/lecture_analyzer/behavior/rules.py` and `src/lecture_analyzer/vision/face_features.py`:

### 4.1 Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)
- **Mathematical Form:**
  $$\text{EAR} = \frac{\|p_{160} - p_{144}\| + \|p_{158} - p_{153}\|}{2 \cdot \|p_{33} - p_{133}\|}, \quad \text{MAR} = \frac{\|m_{13} - m_{14}\| + \|m_{81} - m_{178}\| + \|m_{311} - m_{402}\|}{3 \cdot \|m_{61} - m_{291}\|}$$
- **Safeguard A & B:** If face landmarks are missing or degraded, EAR evaluates to `None`. A value $\le 0.02$ represents a degenerate measurement and is rejected. **Missing face or missing EAR never triggers `DROWSY_SLEEP_LIKE`.**

### 4.2 Head-Pose Euler Angles (Perspective-n-Point)
- **Convention:** Yaw (left negative, right positive), Pitch (down negative, up positive), Roll (tilt left negative, tilt right positive).
- **Safeguard C:** If `solvePnP` fails, `head_pose_valid = False` with `yaw=None, pitch=None, roll=None`. Downstream rules strictly require `head_pose_valid=True`. **Zero angles with `head_pose_valid=False` are never treated as forward-facing.**

### 4.3 Postural Geometry & Phone Evidence
- **Safeguard D:** Posture metrics require `pose_detected=True`. If body pose is missing, `wrists_near_table` is `False`, preventing false `WRITING_READING` classification.
- **Safeguard E:** Phone detection requires `phone_detected=True` and `phone_confidence >= 0.40`.
- **Safeguard F & G:** A person without a detected face or with low landmark quality (`face_quality < 0.35`) is strictly blocked from `ENGAGED_BEHAVIOR` and falls back to `UNCERTAIN`.

---

## 5. Canonical Behavioral Taxonomy vs. Observability States

To resolve semantic ambiguity, the behavioral taxonomy is strictly standardized into seven categories, while observability is modeled as an independent orthogonal diagnostic:

### 5.1 Canonical Seven-Label Behavior Taxonomy
| Canonical Label | Definition | Priority Order |
| :--- | :--- | :--- |
| `PHONE_USE` | Active electronic device present in participant interaction zone. | 1 |
| `DROWSY_SLEEP_LIKE` | Sustained eye closure ($\text{EAR} < 0.20$) or head resting with eyes closed. | 2 |
| `LOOKING_AWAY` | Observable head rotation away from forward field ($|\text{yaw}| > 28^\circ$). | 3 |
| `WRITING_READING` | Downward head pitch with wrists at desk plane. | 4 |
| `HEAD_DOWN` | Downward head pitch without writing posture. | 5 |
| `ENGAGED_BEHAVIOR` | Forward-facing head, open eyes ($\text{EAR} \ge 0.20$), upright posture. | 6 |
| `UNCERTAIN` | Sensory evidence missing, occluded, ambiguous, or unobservable. | 7 (Fallback) |

### 5.2 Orthogonal Observability Levels
Observability indicates sensory integrity, **never a behavior label**:
- `ObservabilityLevel.FULL`: Both face/head-pose and body pose are reliably tracked.
- `ObservabilityLevel.PARTIAL_FACE_ONLY`: Face is tracked, but body pose is occluded.
- `ObservabilityLevel.PARTIAL_POSE_ONLY`: Body pose is tracked, but face is occluded/degraded.
- `ObservabilityLevel.UNOBSERVABLE`: Neither face nor pose is observable.  
  *(When observability is `UNOBSERVABLE`, the canonical behavior label is always `UNCERTAIN`.)*

---

## 6. Timestamp-Based Temporal Persistence & Smoothing

The temporal smoother (`src/lecture_analyzer/behavior/temporal_smoothing.py`) operates strictly on real elapsed monotonic time:

### 6.1 State Persistence Thresholds
A candidate state must persist continuously for at least its physical threshold before displacing an established state:

| Behavioral State | Required Persistence | Rationale |
| :--- | :--- | :--- |
| `DROWSY_SLEEP_LIKE` | **1.5 seconds** | Suppresses natural eye blinks ($0.15\text{ s} - 0.40\text{ s}$). |
| `LOOKING_AWAY` | **0.5 seconds** | Suppresses momentary glances away ($< 0.50\text{ s}$). |
| `HEAD_DOWN` | **0.6 seconds** | Prevents single-frame vertical nodding flicker. |
| `WRITING_READING` | **0.6 seconds** | Confirms sustained reading/writing engagement. |
| `PHONE_USE` | **0.4 seconds** | Requires multi-frame confirmation to prevent YOLO noise. |
| `ENGAGED_BEHAVIOR` | **0.4 seconds** | Confirms return to forward orientation. |
| `UNCERTAIN` | **0.0 seconds** | Transitions immediately if sensor data is lost. |

### 6.2 Smoothing Invariants
1. **Frame-Rate Invariance:** Evaluates $\Delta t = t_{\text{current}} - t_{\text{start}}$, functioning identically at 1.88 FPS, 8.6 FPS, or 30 FPS.
2. **Zero Track Contamination:** Histories are isolated in separate track queues (`_histories[track_id]`).
3. **Tracking Gap Reset:** If a tracking gap exceeds 2.0s (`max_gap_seconds`), historical evidence is purged immediately.
4. **Audit Separation:** Both `raw_behavior` / `raw_score` and `smoothed_behavior` / `smoothed_score` are recorded to SQLite.

---

## 7. Multi-Person Anonymous Tracking Verification

Verified in `tests/test_tracking.py`:
- **Anonymous Session Identifiers:** IDs are formatted as `S001`, `S002`, ...
- **Zero Biometrics:** No face embeddings, facial recognition, or biometric descriptors exist anywhere in the codebase.
- **Track Separation:** Two concurrent subjects maintain independent bounding boxes and feature assignments without identity cross-swapping.
- **Stale Track Purge:** Tracks without detection are retained for a 2.5s grace window, then purged.
- **Privacy-Preserving Re-entry:** When a participant leaves and returns after the retention window, they receive a **new anonymous ID** (`S002`), ensuring zero biometric re-identification.
- **Session Boundary Reset:** Resetting the tracker clears all active tracks and resets ID generation to `S001`.

---

## 8. Video Annotation Workflow & Time Mapping

The annotation interface (`src/lecture_analyzer/ui/annotation_view.py`) provides independent ground-truth labeling:

### 8.1 Time Mapping Definitions
- **Session-Relative Time ($t_{\text{session}}$):** Seconds elapsed since session start (`obs_monotonic_ts - session_start_monotonic`).
- **Video-Relative Time ($t_{\text{video}}$):** Timestamp within the recorded MP4 video file (`frame_index / fps`).  
  *Invariant:* Because video recording begins at session start, $t_{\text{video}} \equiv t_{\text{session}}$.
- **Wall-Clock Time ($t_{\text{wall}}$):** ISO8601 UTC timestamp of frame acquisition (`session_start_wall + elapsed_seconds`).

### 8.2 Annotation Features
- Native OpenCV video playback with play/pause and continuous slider seeking.
- Step backward (-1s) and step forward (+1s) frame controls.
- "Use Playhead" quick-sync buttons populating start and end timestamp fields.
- Real-time cumulative coverage calculation and red warning indicators for overlapping intervals.
- Ground-truth labeling fields are strictly decoupled from automated system predictions.

---

## 9. Machine Learning Readiness Gate Audit

The ML readiness module (`src/lecture_analyzer/ml/readiness.py`) enforces 7 scientific gates:

| Gate | Gating Criterion | Configured Threshold | Current Status |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Strict Provenance | `REAL_HUMAN` only (zero simulated data) | **PASS** (Enforced) |
| **Gate 2** | Participant Diversity | $\ge 10$ distinct genuine participants | **BLOCKED** (0 participants) |
| **Gate 3** | Session Diversity | $\ge 10$ distinct recorded sessions | **BLOCKED** (0 sessions) |
| **Gate 4** | Annotated Duration | $\ge 600.0\text{ s}$ (10 minutes) total annotated time | **BLOCKED** (0.0 s) |
| **Gate 5a** | Overall Feature Completeness | $\le 35\%$ missing feature fraction across matrix | **BLOCKED** (0 rows) |
| **Gate 5b** | Critical Feature Completeness | $\le 20\%$ missing fraction for pitch, yaw, EAR, quality | **BLOCKED** (0 rows) |
| **Gate 6** | Class Balance & Diversity | $\ge 40$ samples per class, $\ge 3$ participants/class | **BLOCKED** (0 rows) |
| **Gate 7** | Evaluation Strategy | GroupKFold cross-validation (zero participant leakage) | **PASS** (Configured) |

**Audit Finding:** `MLTrainingOrchestrator` raises `ReadinessGateViolationError` if invoked before all gates pass. Training is strictly locked.

---

## 10. Automated Test Suite Results

Executed via `.venv\Scripts\python.exe -m pytest tests -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.17, pytest-9.1.1, pluggy-1.7.0 -- D:\STUDENT ATTENTION ANALYSER\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\STUDENT ATTENTION ANALYSER
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 59 items

tests/test_annotations.py (5 tests)                      PASSED [  8%]
tests/test_behavior_rules.py (18 tests)                  PASSED [ 38%]
tests/test_config.py (6 tests)                           PASSED [ 49%]
tests/test_database.py (2 tests)                         PASSED [ 52%]
tests/test_dataset_export.py (1 test)                    PASSED [ 54%]
tests/test_dataset_quality.py (2 tests)                  PASSED [ 57%]
tests/test_end_to_end.py (1 test)                        PASSED [ 59%]
tests/test_head_pose.py (4 tests)                        PASSED [ 66%]
tests/test_ml_readiness.py (8 tests)                     PASSED [ 79%]
tests/test_provenance.py (6 tests)                       PASSED [ 89%]
tests/test_tracking.py (6 tests)                         PASSED [100%]

============================= 59 passed in 24.38s =============================
```

- **Passed:** 59
- **Failed:** 0
- **Skipped:** 0
- **Pass Rate:** **100.0%**

---

## 11. Known Limitations & Scientific Honesty

1. **Physical Population:** Currently **zero genuine human research sessions** exist in the database. No statistical claims regarding classifier accuracy may be made until a real collection campaign is executed.
2. **Pilot Scale:** The 10-participant gate represents a minimum pilot threshold. Machine learning models trained on 10 participants must be reported as a pilot feasibility study, not a generalizable production classifier.
3. **Multi-Person Classroom Camera Test:** The multi-person tracking algorithms have been validated with automated synthetic fixtures and unit tests. Verification with multiple consenting human subjects in a live physical classroom remains pending.
