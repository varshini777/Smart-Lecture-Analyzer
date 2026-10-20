"""Guarded ML inference runtime with automatic rule fallback."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import pandas as pd

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.utils.paths import resolve_path

logger = logging.getLogger("lecture_analyzer.ml.inference")


@dataclass
class MLInferenceResult:
    predicted_behavior: BehaviorCategory
    class_probabilities: Dict[str, float]
    model_version: str
    is_fallback: bool = False


class MLInferenceEngine:
    """Provides guarded ML inference when a validated model is explicitly loaded."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = resolve_path(model_path) if model_path else None
        self.pipeline = None
        self.is_loaded = False
        self.metadata = {}

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        try:
            self.pipeline = joblib.load(self.model_path)
            meta_path = self.model_path.with_name(self.model_path.stem + "_metadata.json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            self.is_loaded = True
            logger.info(f"Loaded active ML model from {self.model_path.name}")
        except Exception as e:
            logger.error(f"Failed to load ML model from {self.model_path}: {e}")
            self.is_loaded = False
            self.pipeline = None

    def predict_observation(self, feats: ObservationFeatures) -> Optional[MLInferenceResult]:
        """Runs ML inference on a single observation feature set."""
        if not self.is_loaded or self.pipeline is None:
            return None

        # Build 1-row DataFrame
        df_row = pd.DataFrame([feats.to_dict()])

        try:
            pred_label = self.pipeline.predict(df_row)[0]
            cat = BehaviorCategory(pred_label)

            probs: Dict[str, float] = {}
            if hasattr(self.pipeline, "predict_proba"):
                proba = self.pipeline.predict_proba(df_row)[0]
                classes = self.pipeline.classes_
                probs = {str(c): round(float(p), 3) for c, p in zip(classes, proba)}

            return MLInferenceResult(
                predicted_behavior=cat,
                class_probabilities=probs,
                model_version=self.metadata.get("model_type", "unknown"),
                is_fallback=False,
            )
        except Exception as e:
            logger.debug(f"ML inference error: {e}")
            return None
