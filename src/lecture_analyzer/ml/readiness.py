"""Machine learning readiness assessment enforcing scientific gating conditions."""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

from lecture_analyzer.storage.database import Database
from lecture_analyzer.utils.paths import resolve_path

logger = logging.getLogger("lecture_analyzer.ml.readiness")


@dataclass
class ReadinessGateReport:
    is_ready_for_training: bool
    passed_gates: List[str]
    failed_gates: List[str]
    gate_details: Dict[str, Any]
    verdict_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLReadinessChecker:
    """Evaluates whether dataset meets mandatory criteria for genuine ML training."""

    def __init__(self, readiness_config_path: str = "configs/ml_readiness.yaml"):
        cfg_path = resolve_path(readiness_config_path)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f).get("readiness_gates", {})
        else:
            self.config = {
                "min_distinct_participants": 10,
                "min_distinct_sessions": 10,
                "min_annotated_duration_seconds": 600.0,
                "min_samples_per_class": 40,
                "min_participants_per_class": 3,
                "max_missing_feature_fraction": 0.35,
                "allowed_provenances": ["REAL_HUMAN"],
            }

    def check_dataframe_readiness(self, df: Optional[pd.DataFrame]) -> ReadinessGateReport:
        """Evaluates readiness criteria against an exported or prepared candidate dataset."""
        passed: List[str] = []
        failed: List[str] = []
        details: Dict[str, Any] = {}

        if df is None or len(df) == 0:
            return ReadinessGateReport(
                is_ready_for_training=False,
                passed_gates=[],
                failed_gates=["DATASET_NON_EMPTY: Dataset is empty or null."],
                gate_details={"row_count": 0},
                verdict_message="ML training blocked: No records available.",
            )

        details["row_count"] = len(df)

        # Gate 1: Provenance Check
        allowed_provs = set(self.config.get("allowed_provenances", ["REAL_HUMAN"]))
        actual_provs = set(df["provenance"].unique()) if "provenance" in df else {"UNKNOWN"}
        details["actual_provenances"] = list(actual_provs)
        disallowed = actual_provs - allowed_provs
        if disallowed:
            failed.append(f"STRICT_PROVENANCE: Dataset contains disallowed provenance types: {disallowed}. Only {allowed_provs} allowed.")
        else:
            passed.append(f"STRICT_PROVENANCE: All records match approved provenance ({allowed_provs}).")

        # Gate 2: Participant Diversity
        min_parts = self.config.get("min_distinct_participants", 10)
        participants = df["participant_id"].dropna().unique() if "participant_id" in df else []
        n_participants = len(participants)
        sessions = df["session_id"].dropna().unique() if "session_id" in df else []
        n_sessions = len(sessions)
        details["distinct_participants"] = n_participants
        details["distinct_sessions"] = n_sessions
        if n_participants < min_parts:
            failed.append(
                f"PARTICIPANT_DIVERSITY: Found {n_participants} unique participant(s) across {n_sessions} session(s) "
                f"(required: >= {min_parts} unique participants). Multiple sessions from the same participant do not increase participant diversity."
            )
        else:
            passed.append(f"PARTICIPANT_DIVERSITY: {n_participants} distinct participants meets requirement (>= {min_parts}).")

        # Gate 3: Session Count
        min_sess = self.config.get("min_distinct_sessions", 10)
        if n_sessions < min_sess:
            failed.append(f"SESSION_DIVERSITY: Found {n_sessions} distinct sessions (required: >= {min_sess}).")
        else:
            passed.append(f"SESSION_DIVERSITY: {n_sessions} distinct sessions meets requirement (>= {min_sess}).")

        # Gate 4: Annotated Duration
        min_duration = float(self.config.get("min_annotated_duration_seconds", 600.0))
        if "elapsed_seconds" in df and len(df) > 0:
            total_duration = 0.0
            if "session_id" in df:
                for _, grp in df.groupby("session_id"):
                    if len(grp) > 1:
                        total_duration += float(grp["elapsed_seconds"].max() - grp["elapsed_seconds"].min())
            else:
                total_duration = float(df["elapsed_seconds"].max() - df["elapsed_seconds"].min())
            details["annotated_duration_seconds"] = round(total_duration, 1)
            if total_duration < min_duration:
                failed.append(
                    f"ANNOTATED_DURATION: Total annotated duration is {total_duration:.1f}s (required: >= {min_duration:.1f}s)."
                )
            else:
                passed.append(
                    f"ANNOTATED_DURATION: Total annotated duration of {total_duration:.1f}s meets requirement (>= {min_duration:.1f}s)."
                )

        # Gate 4: Class Representation & Support
        min_samples_per_class = self.config.get("min_samples_per_class", 40)
        min_parts_per_class = self.config.get("min_participants_per_class", 3)
        if "human_behavior_label" in df:
            label_counts = df["human_behavior_label"].value_counts().to_dict()
            details["label_distribution"] = label_counts

            undersupported_classes = [c for c, cnt in label_counts.items() if cnt < min_samples_per_class]
            if undersupported_classes:
                failed.append(
                    f"CLASS_SUPPORT: Classes {undersupported_classes} have < {min_samples_per_class} annotated samples."
                )
            else:
                passed.append(f"CLASS_SUPPORT: All evaluated classes have >= {min_samples_per_class} annotated samples.")

            # Participant diversity per class
            low_diversity_classes = []
            for lbl in label_counts.keys():
                sub_parts = df[df["human_behavior_label"] == lbl]["participant_id"].nunique()
                if sub_parts < min_parts_per_class:
                    low_diversity_classes.append((lbl, sub_parts))

            if low_diversity_classes:
                failed.append(
                    f"CLASS_PARTICIPANT_DIVERSITY: Classes {low_diversity_classes} lack required participant diversity (>= {min_parts_per_class})."
                )
            else:
                passed.append(f"CLASS_PARTICIPANT_DIVERSITY: Each class represented by >= {min_parts_per_class} participants.")
        else:
            failed.append("GROUND_TRUTH_LABELS: Column 'human_behavior_label' missing from dataset.")

        # Gate 5: Missing Feature Quality (Overall and Per-Feature Critical Audit)
        max_overall = self.config.get("max_overall_missing_fraction", self.config.get("max_missing_feature_fraction", 0.35))
        max_crit = self.config.get("max_critical_feature_missing_fraction", 0.20)
        crit_features = self.config.get("critical_features", ["head_pitch", "head_yaw", "ear_avg", "face_quality"])

        numeric_cols = [c for c in df.columns if c in [
            "ear_avg", "mar", "head_yaw", "head_pitch", "head_roll", "face_quality", "torso_upright_ratio"
        ]]
        if numeric_cols:
            overall_missing = float(df[numeric_cols].isna().mean().mean())
            details["overall_missing_fraction"] = round(overall_missing, 3)
            if overall_missing > max_overall:
                failed.append(
                    f"DATA_COMPLETENESS: Overall missing feature fraction ({overall_missing:.1%}) exceeds maximum ({max_overall:.1%})."
                )
            else:
                passed.append(
                    f"DATA_COMPLETENESS: Overall missing feature fraction ({overall_missing:.1%}) is within tolerance (<= {max_overall:.1%})."
                )

            # Audit per-feature critical completeness
            crit_failures = []
            per_feat_stats = {}
            for feat in crit_features:
                if feat in df.columns:
                    feat_missing = float(df[feat].isna().mean())
                    per_feat_stats[feat] = round(feat_missing, 3)
                    if feat_missing > max_crit:
                        crit_failures.append(f"'{feat}' ({feat_missing:.1%} > {max_crit:.1%})")
                else:
                    crit_failures.append(f"'{feat}' (column missing entirely)")

            details["critical_feature_missingness"] = per_feat_stats
            if crit_failures:
                failed.append(
                    f"CRITICAL_FEATURE_COMPLETENESS: Critical features exceed {max_crit:.1%} missingness tolerance: {', '.join(crit_failures)}."
                )
            else:
                passed.append(
                    f"CRITICAL_FEATURE_COMPLETENESS: All critical predictive features meet <= {max_crit:.1%} completeness requirement."
                )

        is_ready = len(failed) == 0
        if is_ready:
            msg = "READY: Dataset satisfies all scientific readiness gates for participant-held-out ML training."
        else:
            msg = f"BLOCKED: Dataset fails {len(failed)} mandatory readiness gate(s). Training cannot proceed."

        return ReadinessGateReport(
            is_ready_for_training=is_ready,
            passed_gates=passed,
            failed_gates=failed,
            gate_details=details,
            verdict_message=msg,
        )

    def check_database_readiness(self, db: Database) -> ReadinessGateReport:
        """Evaluates readiness directly against records stored in the SQLite database."""
        # Query genuine human records with annotations
        sql = """
        SELECT o.*, s.participant_code as participant_id, s.provenance, a.behavior_label as human_behavior_label
        FROM observations o
        JOIN sessions s ON o.session_id = s.session_id
        JOIN annotations a ON o.session_id = a.session_id
            AND o.elapsed_seconds >= a.start_seconds
            AND o.elapsed_seconds <= a.end_seconds
        WHERE s.provenance = 'REAL_HUMAN';
        """
        rows = db.execute_query(sql)
        if not rows:
            return ReadinessGateReport(
                is_ready_for_training=False,
                passed_gates=[],
                failed_gates=["REAL_HUMAN_DATA_EXISTS: No genuine human annotated records found in database."],
                gate_details={"row_count": 0},
                verdict_message="ML training blocked: No genuine human research data collected or annotated yet.",
            )

        df = pd.DataFrame([dict(r) for r in rows])
        return self.check_dataframe_readiness(df)
