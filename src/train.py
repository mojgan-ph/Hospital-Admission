from pathlib import Path

import xgboost as xgb

from src.utils import load_dataset, set_seeds

OUTPUT_DIR = Path('data/train')

if __name__ == '__main__':
    set_seeds()

    X_train, y_train = load_dataset('X_train.csv', 'y_train.csv')
    X_val, y_val = load_dataset('X_val.csv', 'y_val.csv')

    clf = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        device="cuda",
        early_stopping_rounds=10,
    )
    clf.fit(
        X_train,
        y_train.squeeze(),
        eval_set=[(X_val, y_val.squeeze())],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(OUTPUT_DIR / 'model.json')
