"""학습 루프 밖에서 돌리는 지표들."""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 맞은 비율
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        # 진짜 t, 예측 p 칸에 +1
        cm[t, p] += 1
    return cm


def snr_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_db: np.ndarray,
) -> dict[float, float]:
    result = {}
    for s in sorted(np.unique(snr_db).tolist()):
        mask = snr_db == s
        # 그 SNR만 모아서 acc
        result[float(s)] = accuracy(y_true[mask], y_pred[mask])
    return result
