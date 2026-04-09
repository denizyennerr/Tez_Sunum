import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from typing import Dict, List
from costum_metric import BalancedAccuracy


def get_model_paths_grouped(base_path=None) -> dict:
    if base_path is None:
        base_path = os.path.join(os.getcwd(), "model_paths")
    grouped = {}
    if not os.path.exists(base_path):
        return grouped

    for folder in os.listdir(base_path):
        models_path = os.path.join(base_path, folder, "models")
        if os.path.isdir(models_path):
            grouped[folder] = [
                os.path.join(models_path, file)
                for file in os.listdir(models_path) if file.endswith(('.h5', '.keras'))
            ]
    return grouped



def run_decision_fusion_inference(X_data, model_paths_dict):
    """
    Runs inference on incoming X_data using all available models.
    """
    all_predictions = []

    flat_model_paths = []
    for group, paths in model_paths_dict.items():
        flat_model_paths.extend(paths)

    if not flat_model_paths:
        raise ValueError("No model found to load!")

    for path in flat_model_paths:
        try:
            model = tf.keras.models.load_model(
                filepath=path,
                custom_objects={'BalancedAccuracy': BalancedAccuracy.BalancedAccuracy}
            )

            preds = model.predict(X_data, verbose=0)

            if preds.shape[-1] > 1:
                preds = preds[:, 1]
            else:
                preds = preds.flatten()

            all_predictions.append(preds)

        except Exception as e:
            # UI kurallarına uygun olarak loglama dili İngilizce
            print(f"Error loading or predicting with model: {path} | Error: {e}")

    # Soft Voting
    all_predictions = np.array(all_predictions)
    fused_probabilities = np.mean(all_predictions, axis=0)
    fused_classes = (fused_probabilities >= 0.5).astype(int)

    return fused_probabilities, fused_classes

def window_level_align(coarse_probs: np.ndarray, target_length: int) -> np.ndarray:
    n_coarse = len(coarse_probs)
    if n_coarse == target_length:
        return coarse_probs.copy()
    if target_length % n_coarse == 0:
        ratio = target_length // n_coarse
        return np.repeat(coarse_probs, ratio)
    else:
        # Interpolasyon (fringe cases where not exact multiples)
        fine_indices = np.floor(np.linspace(0, n_coarse, target_length, endpoint=False)).astype(int)
        fine_indices = np.clip(fine_indices, 0, n_coarse - 1)
        return coarse_probs[fine_indices]

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc, auprc = np.nan, np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'Decision Threshold': round(float(threshold), 4),
        'AUROC': round(float(auroc), 4),
        'AUPRC': round(float(auprc), 4),
        'Sensitivity': round(float(sensitivity), 4),
        'Specificity': round(float(specificity), 4),
        'Balanced Accuracy': round(float(balanced_acc), 4),
        'Seizure F1-Score': round(float(f1), 4),
        'Precision': round(float(precision), 4),
    }

def soft_ensemble(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs_list), axis=0)

def hard_vote_ensemble(probs_list: List[np.ndarray], threshold: float = 0.35) -> np.ndarray:
    votes = (np.vstack(probs_list) >= threshold).astype(int)
    return np.mean(votes, axis=0)