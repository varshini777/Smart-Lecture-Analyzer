# Camera Architecture, Troubleshooting, and Lifecycle Guide

This guide documents the camera acquisition and preview subsystem for the **Lecture Engagement Pattern Analyzer**, including the root-cause diagnosis of earlier failure modes, the multi-threaded decoupled architecture, operational procedures, verification results, and lifecycle state management.

---

## 1. Root Cause Diagnosis

### The Symptom
After launching the desktop application (`python -m lecture_analyzer`), the webcam preview failed to display frames or remained permanently inactive/blank, despite diagnostic scripts (`check_environment.py --probe-camera`) successfully opening camera index `0` via DirectShow.

### The Real Technical Root Cause
Tracing the code path revealed four distinct structural flaws:

1. **Synchronous Inference Pipeline Blocking UI Delivery**:
   `CameraCaptureThread` was instantiated and started inside `CameraInferenceWorker.run()`. Immediately after launching capture, `CameraInferenceWorker` synchronously initialized `VisionPipeline`—importing PyTorch, Ultralytics YOLOv8n-pose, Google MediaPipe Face Landmarker, and running dummy warmup inference. On CPU/Windows, this initialization took 16–26 seconds. During this entire interval, `CameraCaptureThread` acquired frames into an internal buffer, but had **no signal connecting frames directly to the UI**. The UI only received frames via `CameraInferenceWorker.frame_processed.emit()`, which could not run until neural model initialization finished. To any user, the camera appeared completely broken, frozen, or failing to open.

2. **OpenCV Backend Priority on Windows**:
   `configs/config.yaml` configured `backend: "DEFAULT"`. In `CameraService.open()`, `"DEFAULT"` mapped directly to `cv2.CAP_ANY`. On modern Windows 10/11, OpenCV's `CAP_ANY` defaults to Media Foundation (`CAP_MSMF`), which frequently fails format negotiation on USB webcams without falling back, whereas DirectShow (`cv2.CAP_DSHOW`) was proven to negotiate 1280×720 at 30 FPS instantly.

3. **Missing First-Frame Hardware Verification**:
   `CameraService.open()` only checked `cap.isOpened()`. On Windows, certain drivers report `cap.isOpened() == True` even when device access is denied or format negotiation fails upon the first read. The opening routine failed to perform a verified test read `cap.read()` before declaring `is_opened = True`.

4. **Disposable Model Pipeline on Camera Toggle**:
   Stopping the camera destroyed `CameraInferenceWorker` and its internal `VisionPipeline`. Restarting the camera forced an entire reload of YOLO weights and MediaPipe tasks (15+ seconds), holding the Python GIL and starving the capture thread of frames upon restart.

---

## 2. Corrected Architecture & Lifecycle

### Architectural Diagram

```
+-------------------------------------------------------------------------+
|                              Main GUI Thread                            |
|                                                                         |
|   +-----------------------+              +--------------------------+   |
|   |    LiveMonitorTab     |              |    VideoDisplayWidget    |   |
|   |                       |              |                          |   |
|   |  - Start/Stop button  |              | - RGB aspect scaled      |   |
|   |  - Telemetry bar      |              | - Mirror comfort mode    |   |
|   |  - Session controls   |              | - Safe QImage deep-copy  |   |
|   +-----------+-----------+              +------------^-------------+   |
|               |                                       |                 |
+---------------|---------------------------------------|-----------------+
                |                                       |
    [Starts / Stops]                           [Direct Preview Frame]
                |                                       |
+---------------v-------------------+                   |
|       CameraCaptureThread         |                   |
|  (Exclusive Camera Hardware Owner)|-------------------+
|                                   |  (Signals: frame_acquired,
|   +--------------------------+    |   camera_started, camera_error)
|   |      CameraService       |    |
|   |   (cv2.VideoCapture)     |    |
|   |   - DirectShow / MSMF    |    |
|   |   - Monotonic timestamps |    |
|   +--------------------------+    |
+-----------------+-----------------+
                  |
         [Atomically Buffers]
                  |
                  v
+-----------------+-----------------+
|      CameraInferenceWorker        |
|  (Pure Neural & Heuristic Worker) |
|                                   |
|   - Reads latest buffered packet  |
|   - Runs YOLOv8 & MediaPipe       |
|   - Evaluates rule engine         |
|   - Temporal smoothing            |
|   - Emits overlay metadata        |
+-----------------+-----------------+
                  |
       [Inference Overlays]
                  |
                  v
       LiveMonitorTab Telemetry
```

### State Machine Transitions

The camera hardware lifecycle transitions through five formal states:

$$\text{CLOSED} \xrightarrow{\text{open()}} \text{OPENING} \xrightarrow{\text{test-read ok}} \text{READY} \xrightarrow{\text{read\_frame()}} \text{STREAMING} \xrightarrow{\text{release()}} \text{STOPPING} \rightarrow \text{CLOSED}$$

- **`CLOSED`**: Device handle released (`cap is None`), zero CPU/driver usage.
- **`OPENING`**: Probing candidates (DirectShow $\rightarrow$ Media Foundation $\rightarrow$ CAP_ANY).
- **`READY`**: Driver confirmed, requested resolution configured, first frame verified valid.
- **`STREAMING`**: High-frequency frame acquisition active at hardware FPS (30 FPS).
- **`STOPPING`**: Threads safely requested to terminate, buffer drained, handle releasing.

---

## 3. Files Changed

| File Path | Nature of Changes |
| :--- | :--- |
| `src/lecture_analyzer/vision/camera.py` | Added `CameraLifecycleState` enum, multi-backend fallback with DirectShow priority on Windows, test-read first-frame verification, robust `release()` idempotency, and high-resolution timing telemetry. |
| `src/lecture_analyzer/ui/live_monitor.py` | Decoupled `CameraCaptureThread` from `CameraInferenceWorker`. Enabled direct zero-lag frame preview via `frame_acquired` signal. Implemented thread-safe `VideoDisplayWidget` with QImage deep copying. Added cached `VisionPipeline` preservation across camera restarts. |
| `src/lecture_analyzer/ui/main_window.py` | Updated `closeEvent` to invoke `tab_live.stop_camera()` ensuring complete resource cleanup upon application exit. |
| `tests/test_camera.py` | Added comprehensive regression test suite (15 tests) covering open failure, read failure, fallback backends, thread lifecycles, error emissions, missing models, and restart recovery. |
| `CAMERA_TROUBLESHOOTING.md` | Created technical documentation detailing diagnosis, lifecycle, operational steps, and verification metrics. |

---

## 4. Operational Instructions

### Exact Startup Command
To launch the desktop application, run:
```powershell
.venv\Scripts\python.exe -m lecture_analyzer
```

### Accessing the Live Monitor
Upon startup, the application displays the **Live Monitor** as the default primary tab. You can also switch to it at any time by clicking the **Live Monitor** tab in the top navigation bar.

### Camera Startup & Preview Mode
- **Does preview begin automatically?**  
  **No.** For research privacy and participant protection, the webcam does not silently record or open on application startup.
- **How to start preview:**  
  Click the green **"Start Camera"** button in the *Camera Device* panel on the right.
- **What happens immediately:**  
  Within 300–500 ms, the physical camera opens, the status indicator turns green (`Active: 1280x720 (DSHOW)`), and live camera video appears in the preview window.
- **AI Model Status:**  
  While the neural network models (YOLO and MediaPipe) are warming up in the background, a non-intrusive status pill displays: `"AI Models Warming Up... Live Feed Active"`. The video feed streams smoothly at 30 FPS. Once warmed up, the badge clears and real-time behavioral classifications and rule scores appear.

### How to Restart the Camera
1. Click the red **"Stop Camera"** button. The camera device is immediately released.
2. Click **"Start Camera"** again. The camera re-opens in ~200 ms and resumes live streaming immediately.

### Starting a Research Recording Session
1. Confirm participant consent by checking **Participant Consent Confirmed**.
2. Enter an anonymous participant identifier (e.g., `P001`).
3. Click **"Start Recording Session"**.
4. If the camera was not running, the application will prompt you to start the camera and ensure it is ready before activating the session.
5. To stop, click **"Stop Recording Session"**. Video and timestamp sidecars are saved cleanly to `data/recordings/`.

---

## 5. Interpreting Status & Telemetry Indicators

| Indicator | Location | Interpretation |
| :--- | :--- | :--- |
| **Acq: XX.X FPS** | Telemetry Bar | Hardware acquisition rate from webcam (nominally 28–32 FPS). |
| **Proc: XX.X FPS** | Telemetry Bar | Neural inference throughput (typically 20–35 FPS depending on hardware). |
| **Latency: XX ms** | Telemetry Bar | Processing latency for pose, face geometry, and 3D head pose. |
| **Skipped: X** | Telemetry Bar | Number of frames skipped by inference to prevent backlog while camera streams at full FPS. |
| **Active: 1280x720 (DSHOW)** | Camera Box | Active physical resolution and backend driver protocol. |
| **Session: ACTIVE (SES_...)**| Telemetry Bar | Active data collection session. Records observations to SQLite. |
| **Session: Inactive** | Telemetry Bar | Camera is in preview mode only. No research records or videos are saved. |

---

## 6. Verification Results

All verification tests were executed against the actual environment and physical webcam:

1. **Environment Probe**:
   ```powershell
   .venv\Scripts\python.exe scripts/check_environment.py --probe-camera
   ```
   *Result:* `[PASS] Camera index 0 captured test frame (1280x720).`

2. **Automated Test Suite**:
   ```powershell
   .venv\Scripts\python.exe -m pytest tests -v
   ```
   *Result:* `85 passed in 32.17s` (including all 15 new camera regression tests).

3. **Physical GUI Application Verification**:
   - Initial button state verified (`Start Camera`).
   - Hardware opened via DirectShow at index `0`.
   - First frame acquired and rendered to `VideoDisplayWidget` in GUI within 500 ms.
   - Live stream continuously updated at **32.4 FPS**.
   - Stop camera cleanly released device and reset display.
   - Camera restart verified successfully (re-opened and acquired 54+ frames).
   - Database integrity verified (`PRAGMA integrity_check -> ok`; zero mock records written to production database).

---

## 7. Known Limitations & Hardware Recommendations
- **Single Camera Ownership**: OpenCV requires exclusive access to the webcam device index. If another program (such as Zoom, Teams, or another Python script) has opened index 0, the application will report a clear `Camera Error` dialog advising the user to close competing processes.
- **Low-Light Environments**: In severely underexposed lighting, camera hardware auto-exposure may reduce native sensor FPS to 15–20 FPS. For research collection, ensure adequate ambient classroom illumination.
