# Smart Lecture Analyzer
*(Lecture Engagement Pattern & Classroom Attention Analyzer)*

A real-time computer-vision research platform designed to observe visible classroom behavior, estimate 3D head pose and posture, track participants anonymously within sessions, classify observable behavioral patterns using an explainable rule engine, audit temporal data quality, manage independent human annotations, enforce strict data provenance, and safeguard participant-held-out machine learning workflows.

---

## 1. Scientific Disclaimer & Ethical Principles

1. **Observable Physical Behavior Only**: This system measures observable visual cues (head orientation, eye aspect ratio, mouth aspect ratio, torso uprightness, phone presence). **It does NOT measure, quantify, or diagnose internal psychological attention, comprehension, learning, cognition, or neurological drowsiness.**
2. **Strict Anonymity**: No facial recognition, face verification, biometric identification, or face embeddings are ever computed or stored. Person tracking is session-local and represented as anonymous IDs (`S001`, `S002`, ...). Tracking states are automatically wiped at session boundaries.
3. **Local Privacy**: All inference runs locally on the host machine. No raw video, biometric descriptors, or classroom data are ever uploaded to cloud services.
4. **Data Provenance Gating**: Synthetic or simulated data is explicitly tagged and barred from entering genuine human research machine learning datasets. Machine learning training remains strictly locked until a genuine human dataset satisfies all scientific readiness criteria.

---

## 2. Observable Behavior Taxonomy

The behavioral engine classifies visual evidence into seven standardized observable categories:

| Category | Operational Definition | Primary Visual Evidence |
| :--- | :--- | :--- |
| `ENGAGED_BEHAVIOR` | Forward-facing orientation consistent with attending to instruction. | Head yaw/pitch within forward threshold, eyes open (`EAR >= 0.20`), upright torso. |
| `LOOKING_AWAY` | Head turned away from forward instructional field. | Head yaw magnitude exceeds threshold (`\|yaw\| > 28°`). |
| `HEAD_DOWN` | Downward head tilt without clear desk writing context. | Head pitch below threshold (`pitch < -18°`). |
| `PHONE_USE` | Cell phone or electronic device detected in interaction zone. | YOLO object detection (COCO class 67) associated with person track. |
| `WRITING_READING` | Downward head angle accompanied by writing/reading posture. | Head pitch below threshold with wrists/hands positioned at table/desk level. |
| `DROWSY_SLEEP_LIKE`| Prolonged eye closure and/or head resting on desk/arm. | Low eye aspect ratio (`EAR < 0.20`) with downward or resting posture. |
| `UNCERTAIN` | Sensory evidence is missing, partial, occluded, or ambiguous. | Face unobservable, landmarks degenerate, or sensory conflict. |

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.11 (64-bit Windows compatible).
- **GUI Framework**: PySide6 (Qt for Python) with multi-threaded architecture (worker threads prevent camera inference from freezing UI).
- **Perception Pipeline**:
  - **Human Pose**: Ultralytics YOLOv8n-pose (17 COCO body keypoints).
  - **Facial Geometry**: Google MediaPipe FaceLandmarker (478 3D landmarks, eye/mouth aspect ratios).
  - **3D Head Pose**: OpenCV `cv2.solvePnP` with 3D canonical anthropometric face model (Euler yaw, pitch, roll).
  - **Object Detection**: Ultralytics YOLOv8n object detection (cell phone, laptop, book).
  - **Tracking**: Anonymous IoU-based session-local tracker.
- **Storage**: Local SQLite with Write-Ahead Logging (WAL) and foreign key integrity.
- **Data & ML**: Pandas, NumPy, SciPy, Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `SimpleImputer`, `GroupKFold`).

---

## 4. Installation & Setup

### 4.1 Prerequisites
- Windows 10/11 64-bit
- Python 3.11 installed
- Physical USB or integrated webcam

### 4.2 Clone & Environment Setup
```powershell
# Open project directory
cd "d:\STUDENT ATTENTION ANALYSER"

# Create virtual environment with uv or python
uv venv --python 3.11 .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies in editable mode
uv pip install -e .
```

### 4.3 Download Offline Model Weights
Model weights are stored locally in `models/` and are never silently downloaded during application startup.
```powershell
python scripts/download_models.py
```
This script downloads and verifies:
1. `models/face_landmarker.task` (Google MediaPipe, Apache 2.0)
2. `models/yolov8n-pose.pt` (Ultralytics, AGPL-3.0)
3. `models/yolov8n.pt` (Ultralytics, AGPL-3.0)

### 4.4 Verify Environment & Hardware
Run the diagnostic utility:
```powershell
python scripts/check_environment.py --probe-camera
```

---

## 5. Running the Application

### Launch GUI
```powershell
python -m lecture_analyzer
# or
python scripts/run_app.py
```

### User Interface Navigation
The desktop application consists of five operational tabs:
1. **Live Monitor**: Live webcam stream with pose skeleton, 3D head pose orientation axes, active behavioral state badges, heuristic rule scores, and session controls.
2. **Sessions and Reports**: Interactive browser of stored sessions with effective FPS, frame intervals, duration metrics, and behavioral distributions.
3. **Human Annotation**: Independent ground-truth labeling workspace featuring synchronized OpenCV video playback, timeline scrubbing slider, playhead interval sync, and annotator attribution.
4. **Dataset & Quality Audit**: Multi-point session audit tool (timestamp jitter, face quality, annotation coverage, system-annotation comparison) and provenance-filtered CSV/manifest export.
5. **ML Readiness & Evaluation**: Automated gating verification (checks participant diversity, session diversity, annotated duration, class support, and provenance) and participant-held-out baseline model training.

---

## 6. CLI Utilities

| Command | Purpose |
| :--- | :--- |
| `python scripts/check_environment.py --probe-camera` | Diagnostics for Python packages, GPU, models, and webcam. |
| `python scripts/download_models.py` | Downloads and verifies offline model files. |
| `python scripts/smoke_test_live_pipeline.py` | Live physical camera perception pipeline smoke test and benchmarks. |
| `python scripts/validate_database.py` | Verifies SQLite schema integrity and foreign keys. |
| `python scripts/run_quality_audit.py` | Performs data-quality audits on recorded sessions. |
| `python scripts/export_dataset.py --provenance REAL_HUMAN` | Exports provenance-filtered dataset and forensic manifest. |
| `pytest tests -v` | Executes the complete automated test suite (85 tests including camera regression). |

For webcam troubleshooting, hardware backend configuration, and lifecycle details, see [CAMERA_TROUBLESHOOTING.md](file:///D:/STUDENT%20ATTENTION%20ANALYSER/CAMERA_TROUBLESHOOTING.md).

---

## 7. Machine Learning Readiness Safeguards

To prevent premature training on unverified or synthetic data, the application enforces the following hard gates (`configs/ml_readiness.yaml`):
- Minimum **10 distinct genuine human participants** (`REAL_HUMAN`).
- Minimum **10 distinct sessions**.
- Minimum **600 seconds of human-annotated duration**.
- Minimum **40 samples per evaluated category** and **>= 3 participants per class**.
- Maximum overall missing feature fraction <= 35%.
- Maximum per-feature critical missingness <= 20% (head pose, EAR, face quality).
- Strict exclusion of `SIMULATED`, `PIPELINE_VALIDATION`, and `UNKNOWN` data.
- Strict `GroupKFold` evaluation ensuring **zero participant leakage** across train/test splits.

If any gate fails, the model training workflow is automatically locked with detailed diagnostic reasons.
