# features.py
import io as _io
import base64
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import requests
import cv2
from skimage.feature import local_binary_pattern, hog # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image as kimage # type: ignore

# ---------- Helpers ----------
def _read_image(input: Union[str, bytes, Image.Image]):
    """
    Accepts:
      - local path string
      - url string (http/https)
      - base64 bytes/string
      - PIL.Image
      - numpy array (H, W, C)
    Returns: PIL.Image (RGB) or raises.
    """
    if isinstance(input, Image.Image):
        return input.convert("RGB")

    if isinstance(input, bytes):
        return Image.open(_io.BytesIO(input)).convert("RGB")

    if isinstance(input, str):
        # base64?
        try:
            # a very small heuristic: base64 usually contains many '=' at end or '/'
            if input.strip().startswith(("iVBOR", "/9j/")) or ("base64," in input):
                # treat as base64
                b = input.split("base64,")[-1]
                raw = base64.b64decode(b)
                return Image.open(_io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass

        # URL?
        if input.startswith("http://") or input.startswith("https://"):
            resp = requests.get(input, timeout=10)
            resp.raise_for_status()
            return Image.open(_io.BytesIO(resp.content)).convert("RGB")

        # local path
        p = Path(input)
        if p.exists():
            return Image.open(p).convert("RGB")
        # else assume raw file-like path string was misused
        raise ValueError(f"Cannot read image from string: {input}")

    # numpy array
    if isinstance(input, np.ndarray):
        if input.ndim == 2:  # gray
            return Image.fromarray(input).convert("RGB")
        return Image.fromarray(input.astype("uint8")).convert("RGB")

    raise TypeError("Unsupported image input type")


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.astype("float32")  # return as-is (avoid divide by zero)
    return (vec / (norm + 0.0)).astype("float32")


# ---------- Extractors ----------
class VGGExtractor:
    def __init__(self, target_size=(224, 224)):
        # include_top=False with pooling='avg' -> 512-d
        base = VGG16(weights="imagenet", include_top=False, pooling="avg")
        self.model = base
        self.size = target_size

    def _preprocess(self, pil_img: Image.Image):
        img = pil_img.resize(self.size)
        arr = kimage.img_to_array(img)  # RGB array
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return arr

    def extract(self, data: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        pil = _read_image(data)
        x = self._preprocess(pil)
        feat = self.model.predict(x, verbose=0)
        feat = feat.reshape(-1)
        return _l2_normalize(feat)


class LBPExtractor:
    def __init__(self, P=24, R=3, target_size=(128, 128), method="uniform"):
        self.P = P
        self.R = R
        self.size = target_size
        self.method = method

    def extract(self, data: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        pil = _read_image(data).convert("L").resize(self.size)
        arr = np.asarray(pil, dtype=np.uint8)
        lbp = local_binary_pattern(arr, self.P, self.R, method=self.method)
        # histogram bins: uniform patterns count = P+2 (using skimage uniform)
        n_bins = int(self.P + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float32")
        if hist.sum() > 0:
            hist /= (hist.sum() + 1e-12)
        return hist.astype("float32")


class HOGExtractor:
    def __init__(self, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9, target_size=(128, 128)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations
        self.size = target_size

    def extract(self, data: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        pil = _read_image(data).convert("L").resize(self.size)
        arr = np.asarray(pil, dtype=np.uint8)
        fd = hog(arr,
                 orientations=self.orientations,
                 pixels_per_cell=self.pixels_per_cell,
                 cells_per_block=self.cells_per_block,
                 block_norm="L2-Hys",
                 feature_vector=True)
        fd = np.asarray(fd, dtype="float32")
        return _l2_normalize(fd)
