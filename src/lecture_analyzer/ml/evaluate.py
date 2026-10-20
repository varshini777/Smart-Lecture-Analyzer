"""Participant-held-out evaluation workflow and metrics computation."""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("lecture_analyzer.ml.evaluate")


@dataclass
class EvaluationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: List[List[int]]
    classes: List[str]


@dataclass
class GroupedEvaluationReport:
    model_name: str
    n_splits: int
    num_participants: int
    num_sessions: int
    total_samples: int
    test_metrics: EvaluationMetrics
    rule_baseline_metrics: Optional[EvaluationMetrics]
    fold_details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_participant_held_out(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    rule_predictions: Optional[pd.Series] = None,
    n_splits: int = 5,
) -> GroupedEvaluationReport:
    """Performs rigorous GroupKFold evaluation strictly holding out entire participants."""
    unique_groups = groups.unique()
    actual_splits = min(n_splits, len(unique_groups))

    if actual_splits < 2:
        raise ValueError(
            f"Cannot perform participant-held-out evaluation with only {len(unique_groups)} distinct participants. Minimum 2 required."
        )

    gkf = GroupKFold(n_splits=actual_splits)
    classes = sorted(list(y.unique()))

    all_y_true = []
    all_y_pred = []
    all_rule_pred = [] if rule_predictions is not None else None
    fold_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_parts = set(groups.iloc[train_idx].unique())
        test_parts = set(groups.iloc[test_idx].unique())

        # Assert zero participant leakage
        overlap = train_parts.intersection(test_parts)
        assert len(overlap) == 0, f"Critical integrity error: Participant leakage detected in fold {fold_idx}: {overlap}"

        # Fit model on training fold only
        pipeline.fit(X_train, y_train)

        # Predict test fold
        preds = pipeline.predict(X_test)
        all_y_true.extend(y_test)
        all_y_pred.extend(preds)

        if all_rule_pred is not None:
            all_rule_pred.extend(rule_predictions.iloc[test_idx])

        fold_summaries.append({
            "fold": fold_idx + 1,
            "train_participants": len(train_parts),
            "test_participants": len(test_parts),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 3),
        })

    # Overall Metrics for ML Model
    ml_metrics = _compute_metrics(all_y_true, all_y_pred, classes)

    # Baseline comparison with Rule Engine
    rule_metrics = None
    if all_rule_pred is not None:
        rule_metrics = _compute_metrics(all_y_true, all_rule_pred, classes)

    return GroupedEvaluationReport(
        model_name=getattr(pipeline.named_steps.get("classifier", None), "__class__", type(pipeline)).__name__,
        n_splits=actual_splits,
        num_participants=len(unique_groups),
        num_sessions=X["session_id"].nunique() if "session_id" in X.columns else 0,
        total_samples=len(X),
        test_metrics=ml_metrics,
        rule_baseline_metrics=rule_metrics,
        fold_details=fold_summaries,
    )


def _compute_metrics(y_true: list, y_pred: list, classes: list) -> EvaluationMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    _, _, class_f1s, class_supps = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return EvaluationMetrics(
        accuracy=round(acc, 3),
        balanced_accuracy=round(bal_acc, 3),
        macro_precision=round(float(prec), 3),
        macro_recall=round(float(rec), 3),
        macro_f1=round(float(f1), 3),
        per_class_f1={c: round(float(score), 3) for c, score in zip(classes, class_f1s)},
        per_class_support={c: int(supp) for c, supp in zip(classes, class_supps)},
        confusion_matrix=cm,
        classes=classes,
    )
