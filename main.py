import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from augment import to_bgr, to_grayscale, to_high_contrast, to_negative
from utils import get_image_paths, image_as_array, pool2d


def main(args: argparse.Namespace):
    ext = ['jpg', 'jpeg', 'png'] if not args.ext \
        else re.split(r', *', args.ext)
    paths = get_image_paths(args.source, ext, args.limit)
    print('Loading source images...')
    images = np.asarray([image_as_array(path, args.crop)
                         for path in tqdm(paths)])
    print('Processing...')
    cat = np.concatenate((images, to_grayscale(images), to_high_contrast(images),
                          to_bgr(images), to_negative(images)), axis=0)
    h_s, w_s = cat.shape[1:3]
    cat_avg = np.mean(cat, axis=(1, 2))
    print('Loading target image...')
    target_name = Path(args.target).stem
    target = image_as_array(args.target)
    h_in, w_in = target.shape[:2]
    target = pool2d(target, pool_size=args.poolsize,
                    stride=args.stride, mode=args.poolmode)
    h_t, w_t, d = target.shape
    view = target.reshape(h_t*w_t, d)
    print('Selecting relevant source images...')
    idx = [np.argmin(np.sqrt((np.mean(cat_avg, axis=1) - np.mean(rgb))**2))
           for rgb in tqdm(view)]
    h, w = h_t*h_s, w_t*w_s
    out = np.zeros((h, w, d), dtype=np.uint8)
    print('Generating mosaic...')
    y = -1
    for pos, i in enumerate(tqdm(idx)):
        im = cat[i]
        k = pos*w_s % w
        if k == 0:
            y += 1
        j = y*h_s % h
        out[j:j+h_s, k:k+w_s, :] = im
    out = Image.fromarray(out)
    if args.keepsize:
        out = out.resize((w_in, h_in))
    out_name = f'{target_name}_mosaic.jpg'
    out.save(out_name)
    print(f"Saved to '{out_name}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Mosaic Generator')
    parser.add_argument('--target', type=str, default=None,
                        help='Path to target image')
    parser.add_argument('--source', type=str, default=None,
                        help='Source image directory')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of source images used')
    parser.add_argument('--ext', type=str, default='jpg,jpeg,png',
                        help='Coma separated string of acceptable image file extensions')
    parser.add_argument('--crop', type=int, default=32,
                        help='Source image size')
    parser.add_argument('--stride', type=int, default=2,
                        help='Pooling factor of target image')
    parser.add_argument('--poolsize', type=int, default=2,
                        help='Pooling window size of target image')
    parser.add_argument('--poolmode', choices=['avg', 'max'], default='avg',
                        help='Pooling mode of target image')
    parser.add_argument('--keepsize', action='store_true',
                        help='Whether to keep the same size as the target image or not')
    args = parser.parse_args()
    print(args)
    main(args)
