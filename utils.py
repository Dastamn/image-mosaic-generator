import random
import re
from glob import glob
from typing import List, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from augment import to_bgr, to_grayscale, to_high_contrast, to_negative


def get_image_paths(dir: str, ext: List[str] = ['jpg', 'jpeg', 'png'], limit: int = None) -> List[str]:
    ext = [(e.lower(), e.upper()) for e in ext]
    ext = tuple(f'.{i}' for item in ext for i in item)
    paths = glob(f'{dir}/**/*', recursive=True)
    if limit:
        paths = random.sample(paths, min(len(paths), limit))
    return list(filter(lambda x: x.endswith(ext), paths))


def image_as_array(path: str, size: int = None, crop: bool = False) -> np.ndarray:
    im = Image.open(path)
    w, h = im.size
    w_start, h_start = 0, 0
    w_size, h_size = w, h
    if size:
        # resize so that smallest dim matches 'size'
        w, h = (size*w//h, size) if w > h else (size, size*h//w)
        im = im.resize((w, h), Image.ANTIALIAS)
        if crop:
            w_start, h_start = w//2 - min(size, w)//2, h//2 - min(size, h)//2
            w_size, h_size = size, size
    im = np.asarray(im, dtype=np.uint8)
    # center crop
    im = im[h_start:h_start+h_size, w_start:w_start+w_size, :]
    return im


def load_source(src_path: str, augment: bool = False, size: int = None, limit: int = None, ext: Union[str, List[str]] = ['jpg', 'jpeg', 'png']) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(ext, str):
        ext = re.split(r', *', ext)
    paths = get_image_paths(src_path, ext, limit)
    images = np.asarray([image_as_array(path, size, crop=True)
                         for path in tqdm(paths)])
    cat = images if not augment \
        else np.concatenate((images,
                             to_grayscale(images),
                             to_high_contrast(images),
                             to_bgr(images),
                             to_negative(images)), axis=0)
    return cat, np.mean(cat, axis=(1, 2))


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
