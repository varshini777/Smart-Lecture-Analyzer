"""Database schema definitions and SQL DDL for Lecture Engagement Pattern Analyzer."""

CURRENT_SCHEMA_VERSION = 1

CREATE_TABLES_SQL = [
    # Migrations table
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,

    # Sessions table
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        participant_code TEXT,
        start_time TEXT NOT NULL,
        end_time TEXT,
        duration_seconds REAL,
        collection_source TEXT NOT NULL,
        provenance TEXT NOT NULL,
        collection_mode TEXT NOT NULL,
        consent_confirmed INTEGER NOT NULL DEFAULT 0,
        recording_path TEXT,
        feature_schema_version TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        model_versions TEXT,
        quality_status TEXT DEFAULT 'PENDING_AUDIT',
        error_message TEXT,
        notes TEXT
    );
    """,

    # Session-local anonymous tracks
    """
    CREATE TABLE IF NOT EXISTS session_tracks (
        session_id TEXT NOT NULL,
        track_id TEXT NOT NULL,
        first_seen_ts TEXT NOT NULL,
        last_seen_ts TEXT NOT NULL,
        total_observations INTEGER DEFAULT 0,
        PRIMARY KEY (session_id, track_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """,

    # Incremental Frame-level Observations
    """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        track_id TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        wall_timestamp TEXT NOT NULL,
        monotonic_ts REAL NOT NULL,
        elapsed_seconds REAL NOT NULL,
        bbox_x1 INTEGER,
        bbox_y1 INTEGER,
        bbox_x2 INTEGER,
        bbox_y2 INTEGER,
        ear_avg REAL,
        ear_left REAL,
        ear_right REAL,
        mar REAL,
        face_quality REAL,
        face_detected INTEGER,
        head_yaw REAL,
        head_pitch REAL,
        head_roll REAL,
        head_pose_valid INTEGER,
        pose_detected INTEGER,
        shoulder_tilt_deg REAL,
        torso_upright_ratio REAL,
        head_shoulder_offset REAL,
        wrists_near_head INTEGER,
        wrists_near_table INTEGER,
        phone_detected INTEGER,
        phone_confidence REAL,
        observability_level TEXT NOT NULL,
        observability_reasons TEXT,
        raw_behavior TEXT NOT NULL,
        raw_score REAL NOT NULL,
        smoothed_behavior TEXT NOT NULL,
        smoothed_score REAL NOT NULL,
        score_type TEXT NOT NULL DEFAULT 'HEURISTIC_RULE_SCORE',
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """,

    # Independent Human Annotations
    """
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        track_id TEXT,
        start_seconds REAL NOT NULL,
        end_seconds REAL NOT NULL,
        behavior_label TEXT NOT NULL,
        annotator_id TEXT NOT NULL,
        confidence_level TEXT DEFAULT 'HIGH',
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """,

    # Session Quality Audit Reports
    """
    CREATE TABLE IF NOT EXISTS quality_reports (
        session_id TEXT PRIMARY KEY,
        verdict TEXT NOT NULL,
        audit_timestamp TEXT NOT NULL,
        observation_count INTEGER NOT NULL,
        mean_interval_ms REAL,
        fps_effective REAL,
        face_quality_mean REAL,
        observability_fraction REAL,
        annotation_coverage_fraction REAL,
        warning_count INTEGER DEFAULT 0,
        warnings_json TEXT,
        report_json TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """,

    # Dataset Exports Metadata
    """
    CREATE TABLE IF NOT EXISTS dataset_exports (
        export_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        export_type TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        csv_path TEXT,
        session_count INTEGER NOT NULL,
        participant_count INTEGER NOT NULL,
        observation_count INTEGER NOT NULL,
        notes TEXT
    );
    """,

    # Machine Learning Models Metadata
    """
    CREATE TABLE IF NOT EXISTS ml_models (
        model_id TEXT PRIMARY KEY,
        model_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        model_path TEXT NOT NULL,
        feature_columns TEXT NOT NULL,
        parameters TEXT,
        readiness_gate_passed INTEGER NOT NULL,
        notes TEXT
    );
    """,

    # Machine Learning Evaluation Results
    """
    CREATE TABLE IF NOT EXISTS ml_evaluations (
        eval_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        evaluated_at TEXT NOT NULL,
        split_strategy TEXT NOT NULL,
        num_participants INTEGER NOT NULL,
        num_sessions INTEGER NOT NULL,
        accuracy REAL,
        balanced_accuracy REAL,
        macro_f1 REAL,
        macro_precision REAL,
        macro_recall REAL,
        metrics_json TEXT NOT NULL,
        FOREIGN KEY (model_id) REFERENCES ml_models(model_id) ON DELETE CASCADE
    );
    """
]

CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_obs_session_track ON observations (session_id, track_id);",
    "CREATE INDEX IF NOT EXISTS idx_obs_elapsed ON observations (session_id, elapsed_seconds);",
    "CREATE INDEX IF NOT EXISTS idx_annot_session ON annotations (session_id, start_seconds, end_seconds);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_provenance ON sessions (provenance);",
]
