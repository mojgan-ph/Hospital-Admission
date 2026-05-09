import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from ucimlrepo import fetch_ucirepo

from src.utils import RANDOM_SEED, set_seeds

OUTPUT_DIR = Path('data/split')

if __name__ == '__main__':
    set_seeds()

    # fetch dataset
    dataset = fetch_ucirepo(id=296)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets == '<30'
    groups = dataset.data.ids['patient_nbr']

    train_val_split = GroupShuffleSplit(
        n_splits=1, test_size=0.4, random_state=RANDOM_SEED
    )
    train_idx, temp_idx = next(train_val_split.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]

    val_test_split = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=RANDOM_SEED
    )
    val_idx, test_idx = next(val_test_split.split(X_temp, y_temp, groups=groups_temp))

    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset.variables.to_csv(OUTPUT_DIR / 'variables.csv', index=False)

    categorical_features = dataset.variables.loc[
        dataset.variables['type'] == 'Categorical', 'name'
    ]
    categories = {
        col: sorted(X[col].dropna().unique().tolist())
        for col in categorical_features if col in X.columns
    }
    with open(OUTPUT_DIR / 'categories.json', 'w') as f:
        json.dump(categories, f)

    X_train.to_csv(OUTPUT_DIR / 'X_train.csv', index=False)
    y_train.to_csv(OUTPUT_DIR / 'y_train.csv', index=False)
    X_val.to_csv(OUTPUT_DIR / 'X_val.csv', index=False)
    y_val.to_csv(OUTPUT_DIR / 'y_val.csv', index=False)
    X_test.to_csv(OUTPUT_DIR / 'X_test.csv', index=False)
    y_test.to_csv(OUTPUT_DIR / 'y_test.csv', index=False)
