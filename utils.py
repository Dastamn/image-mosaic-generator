from glob import glob
from typing import List, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from typing_extensions import Literal


def get_image_paths(dir: str, ext: List[str] = ['jpg', 'jpeg', 'png']) -> List[str]:
    ext = [(e, e.upper()) for e in ext]
    ext = tuple(f'.{i}' for item in ext for i in item)
    return list(filter(lambda x: x.endswith(ext), glob(f'{dir}/**/*', recursive=True)))


def image_as_array(path: str, crop: int = None) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img, dtype=np.uint8)
    if crop:
        crop = min(crop, min(img.shape[:2]))
        h, w, _ = img.shape
        h_start = h//2 - crop//2
        w_start = w//2 - crop//2
        img = img[h_start:h_start+crop, w_start:w_start+crop, :]
    return img


def tile(x: np.ndarray, window: Tuple[int, int] = (2, 2), stride: int = 2) -> np.ndarray:
    s0, s1 = x.strides[:2]
    h, w = x.shape[:2]
    h_w, w_w = window[:2]
    view_shape = (1 + (h-h_w) // stride, 1 + (w-w_w) // stride, h_w, w_w) \
        + x.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1) + x.strides[2:]
    return as_strided(x, view_shape, strides, writeable=False)


def pool2d(x: np.ndarray, pool_size: int = 2, stride: int = 2, mode: Literal['max', 'avg'] = 'avg') -> np.ndarray:
    view = tile(x, (pool_size,)*2, stride)
    return np.mean(view, axis=(2, 3)).astype(np.uint8) if mode == 'avg' else np.max(view, axis=(2, 3))
