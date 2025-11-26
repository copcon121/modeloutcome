"""
Evaluation Metrics
Custom metrics for outcome model evaluation
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Compute precision, recall, F1 for each class

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict with metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )

    metrics = {
        'long': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': support[0]},
        'short': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': support[1]},
        'skip': {'precision': precision[2], 'recall': recall[2], 'f1': f2], 'support': support[2]},
    }

    # Overall metrics
    metrics['macro_avg'] = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1)
    }

    return metrics


def compute_expected_R(
    y_true: List[int],
    y_pred: List[int],
    max_up_R: List[float],
    max_down_R: List[float],
    target_R: float = 2.0,
    stop_R: float = 1.0
) -> float:
    """
    Compute expected R-multiple based on predictions

    This simulates what would happen if we traded based on model predictions

    Args:
        y_true: True labels (0=long, 1=short, 2=skip)
        y_pred: Predicted labels
        max_up_R: Maximum upside R for each sample
        max_down_R: Maximum downside R for each sample
        target_R: Target R-multiple
        stop_R: Stop loss R-multiple

    Returns:
        Expected R-multiple per trade
    """
    total_R = 0.0
    num_trades = 0

    for i in range(len(y_pred)):
        pred = y_pred[i]
        true_label = y_true[i]

        # Skip if model says skip
        if pred == 2:
            continue

        num_trades += 1

        # Predicted LONG
        if pred == 0:
            if true_label == 0:  # Correct long
                total_R += target_R
            elif true_label == 1:  # Wrong (should be short)
                total_R -= stop_R
            else:  # Should have skipped
                # Use actual outcome
                if max_up_R[i] >= target_R:
                    total_R += target_R
                else:
                    total_R -= stop_R

        # Predicted SHORT
        elif pred == 1:
            if true_label == 1:  # Correct short
                total_R += target_R
            elif true_label == 0:  # Wrong (should be long)
                total_R -= stop_R
            else:  # Should have skipped
                if max_down_R[i] <= -target_R:
                    total_R += target_R
                else:
                    total_R -= stop_R

    if num_trades == 0:
        return 0.0

    expected_R = total_R / num_trades
    return expected_R


def precision_at_confidence(
    y_true: List[int],
    y_probs: np.ndarray,
    confidence_threshold: float = 0.6
) -> Dict:
    """
    Compute precision for predictions above confidence threshold

    Args:
        y_true: True labels
        y_probs: Predicted probabilities [N, num_classes]
        confidence_threshold: Minimum confidence to consider

    Returns:
        Dict with precision metrics
    """
    y_pred = np.argmax(y_probs, axis=1)
    max_probs = np.max(y_probs, axis=1)

    # Filter by confidence
    high_conf_mask = max_probs >= confidence_threshold
    y_true_filtered = np.array(y_true)[high_conf_mask]
    y_pred_filtered = y_pred[high_conf_mask]

    if len(y_true_filtered) == 0:
        return {'precision': 0.0, 'num_samples': 0}

    correct = np.sum(y_true_filtered == y_pred_filtered)
    precision = correct / len(y_true_filtered)

    return {
        'precision': precision,
        'num_samples': len(y_true_filtered),
        'coverage': len(y_true_filtered) / len(y_true)
    }
