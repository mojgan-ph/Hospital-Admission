from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay

MODEL_FILE = Path('data/train/model.json')
PREDICTIONS_FILE = Path('data/evaluate/predictions.csv')
OUTPUT_DIR = Path('data/analyse')

if __name__ == '__main__':
    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(MODEL_FILE)

    predictions = pd.read_csv(PREDICTIONS_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(clf, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
    plt.close(fig)

    for threshold in (0.1, 0.2):
        y_pred_thr = (predictions['y_proba'] >= threshold).astype(int)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(
            predictions['y_true'], y_pred_thr, ax=ax
        )
        ax.set_title(f'Confusion Matrix (threshold = {threshold})')
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f'confusion_matrix_thr_{threshold:.1f}.png', dpi=150
        )
        plt.close(fig)
