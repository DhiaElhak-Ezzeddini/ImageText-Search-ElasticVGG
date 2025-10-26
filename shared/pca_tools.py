# pca_tools.py
from sklearn.decomposition import IncrementalPCA # type: ignore
import joblib # type: ignore
import numpy as np
from tqdm import tqdm # type: ignore
from typing import Iterable
from pathlib import Path

def train_ipca(feature_generator: Iterable[np.ndarray], n_components: int, batch_size: int = 1024):
    ipca = IncrementalPCA(n_components=n_components)
    buf = []
    for i, feat in enumerate(tqdm(feature_generator)):
        buf.append(feat)
        if len(buf) >= batch_size:
            X = np.vstack(buf)
            ipca.partial_fit(X)
            buf = []
    if buf:
        ipca.partial_fit(np.vstack(buf))
    return ipca

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
