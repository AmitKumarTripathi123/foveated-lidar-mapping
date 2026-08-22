"""
Phase 2 Semantic Segmentation Evaluator.
"""
from typing import Dict, Any, Optional
import numpy as np
from src.types import SuperClass


class Phase2SemanticEvaluator:
    def __init__(self, num_classes: int = 4, ignore_label: int = SuperClass.IGNORE_LABEL):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = ["drivable_terrain", "non_drivable_terrain", "static_obstacle", "dynamic_object"]

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        ranges: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        valid_mask = (targets != self.ignore_label)
        preds = predictions[valid_mask]
        targs = targets[valid_mask]

        if len(targs) == 0:
            return {"mIoU": 0.0, "overall_accuracy": 0.0}

        oa = float(np.mean(preds == targs))

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for p, t in zip(preds, targs):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                cm[t, p] += 1

        ious, precisions, recalls, f1s = {}, {}, {}, {}

        for c in range(self.num_classes):
            c_name = self.class_names[c]
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp

            denom_iou = tp + fp + fn
            iou_val = float(tp / denom_iou) if denom_iou > 0 else 0.0
            ious[f"{c_name}_IoU"] = round(iou_val, 4)

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            precisions[f"{c_name}_Precision"] = round(prec, 4)
            recalls[f"{c_name}_Recall"] = round(rec, 4)
            f1s[f"{c_name}_F1"] = round(f1, 4)

        miou = float(np.mean(list(ious.values())))

        result = {
            "mIoU": round(miou, 4),
            "overall_accuracy": round(oa, 4),
            "confusion_matrix": cm.tolist(),
            **ious,
            **precisions,
            **recalls,
            **f1s
        }

        if ranges is not None:
            r_valid = ranges[valid_mask]
            result["distance_bands"] = {
                "near_0_10m": self._eval_band(preds, targs, r_valid, 0.0, 10.0),
                "mid_10_40m": self._eval_band(preds, targs, r_valid, 10.0, 40.0),
                "far_40_100m": self._eval_band(preds, targs, r_valid, 40.0, 100.0)
            }

        if probabilities is not None:
            probs_valid = probabilities[valid_mask]
            conf = np.max(probs_valid, axis=-1)
            correct = (preds == targs)

            result["confidence_stats"] = {
                "mean_confidence": round(float(np.mean(conf)), 4),
                "correct_mean_confidence": round(float(np.mean(conf[correct])), 4) if np.any(correct) else 0.0,
                "incorrect_mean_confidence": round(float(np.mean(conf[~correct])), 4) if np.any(~correct) else 0.0,
                "ece": round(self._calc_ece(conf, correct), 4)
            }

        return result

    def _eval_band(self, preds: np.ndarray, targs: np.ndarray, ranges: np.ndarray, r_min: float, r_max: float) -> Dict[str, float]:
        b_mask = (ranges >= r_min) & (ranges < r_max if r_max < 100.0 else ranges <= r_max)
        p_b, t_b = preds[b_mask], targs[b_mask]
        if len(t_b) == 0:
            return {"mIoU": 0.0, "points": 0}

        b_ious = []
        band_dict = {}
        for c in range(self.num_classes):
            c_name = self.class_names[c]
            tp = np.sum((p_b == c) & (t_b == c))
            fp = np.sum((p_b == c) & (t_b != c))
            fn = np.sum((p_b != c) & (t_b == c))
            iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
            band_dict[f"{c_name}_IoU"] = round(iou, 4)
            b_ious.append(iou)

        band_dict["mIoU"] = round(float(np.mean(b_ious)), 4)
        band_dict["points"] = int(len(t_b))
        return band_dict

    def _calc_ece(self, conf: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n = len(conf)
        for i in range(num_bins):
            bin_mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if np.any(bin_mask):
                bin_acc = np.mean(correct[bin_mask])
                bin_conf = np.mean(conf[bin_mask])
                bin_size = np.sum(bin_mask)
                ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
        return float(ece)
