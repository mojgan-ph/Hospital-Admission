import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 67
SPLIT_DIR = Path('data/split')
CATEGORIES_FILE = SPLIT_DIR / 'categories.json'


def set_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def load_dataset(features_file, target_file):
    X = pd.read_csv(SPLIT_DIR / features_file)
    y = pd.read_csv(SPLIT_DIR / target_file)

    with open(CATEGORIES_FILE) as f:
        categories = json.load(f)

    for feature, levels in categories.items():
        if feature in X.columns:
            X[feature] = pd.Categorical(X[feature], categories=levels)

    return X, y
