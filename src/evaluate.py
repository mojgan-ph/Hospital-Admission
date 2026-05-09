from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }).to_csv(OUTPUT_DIR / 'predictions.csv', index=False)

    pd.DataFrame(
        [{'metric': k, 'value': v} for k, v in metrics.items()]
    ).to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
