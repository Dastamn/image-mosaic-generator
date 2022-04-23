import random
from glob import glob
from typing import List, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from typing_extensions import Literal


def get_image_paths(dir: str, ext: List[str] = ['jpg', 'jpeg', 'png'], limit: int = None) -> List[str]:
    ext = [(e.lower(), e.upper()) for e in ext]
    ext = tuple(f'.{i}' for item in ext for i in item)
    paths = glob(f'{dir}/**/*', recursive=True)
    if limit:
        paths = random.sample(paths, limit)
    return list(filter(lambda x: x.endswith(ext), paths))


def image_as_array(path: str, size: int = None) -> np.ndarray:
    im = Image.open(path)
    w, h = im.size
    w_start, h_start = 0, 0
    w_size, h_size = w, h
    if size:
        # resize so that smallest dim matches 'size'
        w, h = (size*w//h, size) if w > h else (size, size*h//w)
        w_start, h_start = w//2 - min(size, w)//2, h//2 - min(size, h)//2
        w_size, h_size = size, size
        im = im.resize((w, h))
    im = np.asarray(im, dtype=np.uint8)
    # center crop
    im = im[h_start:h_start+h_size, w_start:w_start+w_size, :]
    return im


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
