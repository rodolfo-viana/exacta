"""Stateless metric functions. Each returns a (B, C) array of per-case, per-class values."""

from exacta.functional.overlap import (
    dice,
    iou,
    precision,
    relative_volume_difference,
    sensitivity,
    specificity,
    volume_similarity,
)
from exacta.functional.surface import assd, hausdorff_distance, hd95, nsd
from exacta.functional.area import predicted_area, target_area, area_error
from exacta.functional.volume import predicted_volume, target_volume, volume_error
from exacta.functional.ranking import RankingCurve, RankingResult, voxel_ranking, average_precision, auroc
from exacta.functional.collection import evaluate
from exacta.functional.topology import cldice, skeletonize
from exacta.functional.calibration import (
    ReliabilityDiagram, brier_score, negative_log_likelihood, expected_calibration_error, reliability_diagram,
)

__all__ = [
    "RankingCurve", "RankingResult", "voxel_ranking", "average_precision", "auroc",
    "predicted_area", "target_area", "area_error",
    "predicted_volume", "target_volume", "volume_error",
    "cldice", "skeletonize", "ReliabilityDiagram", "brier_score", "negative_log_likelihood",
    "expected_calibration_error", "reliability_diagram",
    "assd",
    "dice",
    "evaluate",
    "hausdorff_distance",
    "hd95",
    "iou",
    "nsd",
    "precision",
    "relative_volume_difference",
    "sensitivity",
    "specificity",
    "volume_similarity",
]
