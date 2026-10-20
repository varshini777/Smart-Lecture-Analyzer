"""Machine learning training orchestrator with mandatory readiness gating."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from lecture_analyzer.ml.evaluate import GroupedEvaluationReport, evaluate_participant_held_out
from lecture_analyzer.ml.features import build_preprocessor, prepare_feature_matrix
from lecture_analyzer.ml.readiness import MLReadinessChecker, ReadinessGateReport
from lecture_analyzer.storage.database import Database
from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.ml.train")


class ReadinessGateViolationError(Exception):
    """Raised when an attempt is made to train a model on an ineligible dataset."""
    pass


class MLTrainingOrchestrator:
    """Manages baseline model training, evaluation, and serialization under strict gating."""

    def __init__(self, db: Database, models_dir: str = "models"):
        self.db = db
        self.models_dir = resolve_path(models_dir)
        self.readiness_checker = MLReadinessChecker()

    def train_and_evaluate(
        self,
        candidate_df: Optional[pd.DataFrame] = None,
        model_type: str = "random_forest",
        bypass_gate_for_synthetic_debug_only: bool = False,
    ) -> Tuple[Pipeline, GroupedEvaluationReport]:
        """Gated model training workflow.

        Args:
            candidate_df: Candidate DataFrame with features and labels. If None, queries DB.
            model_type: 'random_forest', 'logistic_regression', 'decision_tree', or 'majority'
            bypass_gate_for_synthetic_debug_only: If True, explicitly tags model as DEBUG_ONLY.
        """
        # 1. Run Readiness Assessment
        if candidate_df is None:
            gate_report = self.readiness_checker.check_database_readiness(self.db)
        else:
            gate_report = self.readiness_checker.check_dataframe_readiness(candidate_df)

        if not gate_report.is_ready_for_training and not bypass_gate_for_synthetic_debug_only:
            reasons_str = "\n  - " + "\n  - ".join(gate_report.failed_gates)
            raise ReadinessGateViolationError(
                f"ML Training Gate FAILED:\n{gate_report.verdict_message}{reasons_str}\n\n"
                "System policy strictly prohibits automated model training on incomplete or unverified research data."
            )

        if candidate_df is None:
            # Query data from DB
            sql = """
            SELECT o.*, s.participant_code as participant_id, s.provenance, a.behavior_label as human_behavior_label
            FROM observations o
            JOIN sessions s ON o.session_id = s.session_id
            JOIN annotations a ON o.session_id = a.session_id
                AND o.elapsed_seconds >= a.start_seconds
                AND o.elapsed_seconds <= a.end_seconds;
            """
            rows = self.db.execute_query(sql)
            candidate_df = pd.DataFrame([dict(r) for r in rows])

        X, y, groups = prepare_feature_matrix(candidate_df)
        rule_preds = candidate_df["smoothed_behavior"] if "smoothed_behavior" in candidate_df.columns else None

        # 2. Select Classifier
        if model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        elif model_type == "logistic_regression":
            clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        elif model_type == "decision_tree":
            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
        elif model_type == "majority":
            clf = DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'")

        # 3. Assemble Pipeline
        preprocessor = build_preprocessor()
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # 4. Run Participant-Held-Out Evaluation
        eval_report = evaluate_participant_held_out(
            pipeline=full_pipeline,
            X=X,
            y=y,
            groups=groups,
            rule_predictions=rule_preds,
            n_splits=5,
        )

        # 5. Fit Final Model on all candidate data
        full_pipeline.fit(X, y)

        # 6. Serialize Model and Metadata
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"active_{model_type}.joblib"
        model_path = self.models_dir / model_filename
        joblib.dump(full_pipeline, model_path)

        meta_filename = f"active_{model_type}_metadata.json"
        meta_path = self.models_dir / meta_filename
        metadata = {
            "model_type": model_type,
            "created_at": now_utc_iso(),
            "is_debug_only": bypass_gate_for_synthetic_debug_only,
            "readiness_gate_passed": gate_report.is_ready_for_training,
            "evaluation_summary": eval_report.to_dict(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Trained and evaluated {model_type}: Macro F1={eval_report.test_metrics.macro_f1:.3f}")
        return full_pipeline, eval_report
