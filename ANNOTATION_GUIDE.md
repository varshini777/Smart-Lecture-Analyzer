# Independent Human Ground-Truth Annotation Guide

**Lecture Engagement Pattern Analyzer**  
*(Classroom Behavioral Observation and Analysis System)*

---

## 1. Principles of Independent Annotation

1. **Strict Decoupling**: Ground-truth human annotations must be generated independently of automated system outputs. Annotators must inspect visual evidence (live session or recorded video) and assign categories without the interface pre-filling or biasing toward automated predictions.
2. **Observable Physical Evidence**: Annotations describe observable physical postures, head orientations, and tool interactions. Annotators must never attempt to guess a student's internal thoughts, understanding, or psychological state.
3. **Conservative Uncertainty**: When visual evidence is ambiguous, obstructed, or unclear, assign `UNCERTAIN`.

---

## 2. Behavioral Label Taxonomy & Operational Criteria

### 2.1 `ENGAGED_BEHAVIOR`
- **Definition**: Participant exhibits forward-facing visual orientation consistent with attending to the primary instructional space (lecturer, whiteboard, or presentation screen).
- **Inclusion Criteria**:
  - Head orientation is generally facing forward (yaw within roughly ±25°, pitch within upright forward limits).
  - Eyes are visible and open (not resting closed).
  - Body posture is generally upright or oriented toward the board.
- **Scientific Limitation**: Forward orientation does **not** prove cognitive comprehension or active listening. It merely reflects an outward forward-attending posture.

### 2.2 `LOOKING_AWAY`
- **Definition**: Participant's head is turned distinctly away from the instructional space.
- **Inclusion Criteria**:
  - Head turned significantly to the left or right (yaw > 28°).
  - Gaze directed out a window, toward a classmate, or away from the presentation area.
- **Exclusion**: Brief transient saccades or slight head movements (< 1.0 second) should not be labeled as looking away.

### 2.3 `HEAD_DOWN`
- **Definition**: Participant's head is tilted downward without visible evidence of note-taking or desk study.
- **Inclusion Criteria**:
  - Head pitch tilted significantly downward (< -18°).
  - Gaze directed toward lap or floor.
  - Hands are not actively holding a pen, writing on a notebook, or typing on a keyboard.

### 2.4 `PHONE_USE`
- **Definition**: Participant is handling, interacting with, or visually attending to a cell phone or similar handheld mobile device.
- **Inclusion Criteria**:
  - Phone is visible in the participant's hands, on the lap, or held at desk level.
  - Head and gaze are oriented toward the device.

### 2.5 `WRITING_READING`
- **Definition**: Participant exhibits downward head tilt accompanied by posture consistent with reading desk material, writing notes, or working on assignments.
- **Inclusion Criteria**:
  - Head tilted downward toward desk level.
  - Hands or wrists positioned on paper, notebook, book, or keyboard with observable writing or typing actions.

### 2.6 `DROWSY_SLEEP_LIKE`
- **Definition**: Observable prolonged eye closure or physical posture resembling resting/sleeping.
- **Inclusion Criteria**:
  - Continuous eye closure for >= 2.0 seconds while upright, or head resting on the desk, arm, or hand.
  - Head nodding with intermittent eye closure.
- **Caution**: Strictly an observable posture category; not a clinical or neurological sleep diagnosis.

### 2.7 `UNCERTAIN`
- **Definition**: Visual evidence is inadequate, obstructed, or contradictory.
- **Inclusion Criteria**:
  - Participant's face or body is occluded by another person or object.
  - Lighting is too dim to discern facial features.
  - Ambiguous posture that cannot be reliably distinguished.

---

## 3. Annotation Workflow in the Desktop GUI

1. Open the **Human Annotation** tab.
2. Select the target recorded session from the dropdown.
3. Review the video playback or session timeline.
4. Specify the **Start Timestamp** and **End Timestamp** in seconds.
5. Select the observed **Behavior Category** from the dropdown.
6. Enter your unique **Annotator ID** (e.g., `ANN_01`).
7. Enter any contextual notes (e.g., "Student was adjusting chair at 04:12").
8. Click **Save Ground-Truth Annotation**.
9. The interface updates the **Annotation Coverage** percentage and highlights any overlapping intervals.
