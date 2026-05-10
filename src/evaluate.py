from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import load_dataset

MODEL_FILE = Path('data/train/model.json')
OUTPUT_DIR = Path('data/evaluate')

if __name__ == '__main__':
    X_test, y_test = load_dataset('X_test.csv', 'y_test.csv')
    y_true = y_test.squeeze()

    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(MODEL_FILE)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }).to_csv(OUTPUT_DIR / 'predictions.csv', index=False)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_proba):.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

    threshold_rows = []
    for tpr_target in np.arange(0.1, 1.0, 0.1):
        idx = np.argmax(tpr >= tpr_target)
        thr = thresholds[idx]
        ax.scatter(fpr[idx], tpr[idx], color='red', s=20, zorder=3)
        ax.annotate(
            f'thr={thr:.2f}',
            (fpr[idx], tpr[idx]),
            textcoords='offset points',
            xytext=(6, -4),
            fontsize=8,
        )

        y_pred_thr = (y_proba >= thr).astype(int)
        threshold_rows.append({
            'threshold': thr,
            'fpr': fpr[idx],
            'tpr': tpr[idx],
            'accuracy': accuracy_score(y_true, y_pred_thr),
            'precision': precision_score(y_true, y_pred_thr, zero_division=0),
            'recall': recall_score(y_true, y_pred_thr),
            'f1': f1_score(y_true, y_pred_thr),
        })

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT_DIR / 'roc_curve.png', bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / 'threshold_metrics.csv', index=False
    )
