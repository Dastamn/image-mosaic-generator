import numpy as np


def to_grayscale(images: np.ndarray) -> np.ndarray:
    _images = np.dot(images[..., :3],
                     [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    _images = np.expand_dims(_images, axis=-1)
    return np.repeat(_images, repeats=3, axis=-1)


def to_bgr(images: np.ndarray) -> np.ndarray:
    _images = images.copy()
    R, B = images[:, :, :, 0],  images[:, :, :, 2]
    _images[:, :, :, 0], _images[:, :, :, 2] = B, R
    return _images


def to_high_contrast(images: np.ndarray,  factor=2.) -> np.ndarray:
    R, G, B = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
    _images = np.empty(images.shape, dtype=np.uint8)
    _images[:, :, :, 0] = np.clip(128 + factor*R - factor*128, 0, 255)
    _images[:, :, :, 1] = np.clip(128 + factor*G - factor*128, 0, 255)
    _images[:, :, :, 2] = np.clip(128 + factor*B - factor*128, 0, 255)
    return _images


def to_negative(images: np.ndarray) -> np.ndarray:
    return 255 - images
