import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import image_as_array, load_source, pool2d


def gen(args: argparse.Namespace):
    print('Loading source images...')
    src, src_avg = load_source(
        args.source, True, args.srcsize, args.limit, args.ext)
    h_s, w_s = src.shape[1:3]
    print('Loading target image...')
    target_name = Path(args.target).stem
    target = image_as_array(args.target, args.resize)
    h_in, w_in = target.shape[:2]
    target = pool2d(target, pool_size=args.poolsize,
                    stride=args.stride, mode=args.poolmode)
    h_t, w_t, d = target.shape
    view = target.reshape(h_t*w_t, d)
    print('Selecting relevant source images...')
    idx = [np.argmin(np.mean(np.sqrt((rgb - src_avg)**2), axis=1))
           for rgb in tqdm(view)]
    h, w = h_t*h_s, w_t*w_s
    out = np.zeros((h, w, d), dtype=np.uint8)
    print('Generating mosaic...')
    y = -1
    for pos, i in enumerate(tqdm(idx)):
        im = src[i]
        k = pos*w_s % w
        if k == 0:
            y += 1
        j = y*h_s % h
        out[j:j+h_s, k:k+w_s, :] = im
    out = Image.fromarray(out)
    if args.keepsize:
        out.resize((w_in, h_in), Image.ANTIALIAS)
    out_name = f'{target_name}_mosaic.jpg'
    out.save(out_name)
    print(f"Saved to '{out_name}'.")


def test(args: argparse.Namespace):
    print('Loading source images...')
    src, src_avg = load_source(
        args.source, False, args.srcsize, args.limit, args.ext)
    rgb_arr = np.random.randint(0, 256, size=(args.samples, 3), dtype=np.uint8)
    idx = [np.argmin(np.mean(np.sqrt((rgb - src_avg)**2), axis=1))
           for rgb in tqdm(rgb_arr)]
    _, axes = plt.subplots(args.samples, 3, layout='constrained')
    for i, rgb in enumerate(rgb_arr):
        index = idx[i]
        if i == 0:
            axes[i, 0].set_title('Target Color')
            axes[i, 1].set_title('Found Color')
            axes[i, 2].set_title('Source Image')
        # target
        axes[i, 0].imshow(rgb.reshape(1, 1, 3))
        axes[i, 0].axis('off')
        # found
        axes[i, 1].imshow(src_avg[index].reshape(1, 1, 3).astype(np.uint8))
        axes[i, 1].axis('off')
        # source
        axes[i, 2].imshow(src[index])
        axes[i, 2].axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Mosaic Generator')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')
    # generator
    gen_parser = subparsers.add_parser('gen', help='Generate an image mosaic')
    gen_parser.add_argument('--target', type=str, default=None,
                            help='Path to target image')
    gen_parser.add_argument('--resize', type=int, default=None,
                            help='Resize target image while preserving aspect ratio')
    gen_parser.add_argument('--source', type=str, default=None,
                            help='Source image directory')
    gen_parser.add_argument('--limit', type=int, default=None,
                            help='Limit the number of source images used')
    gen_parser.add_argument('--ext', type=str, default='jpg,jpeg,png',
                            help='Coma separated string of acceptable image file extensions')
    gen_parser.add_argument('--srcsize', type=int, default=32,
                            help='Source image size')
    gen_parser.add_argument('--stride', type=int, default=2,
                            help='Pooling factor of target image')
    gen_parser.add_argument('--poolsize', type=int, default=2,
                            help='Pooling window size of target image')
    gen_parser.add_argument('--poolmode', choices=['avg', 'max'], default='avg',
                            help='Pooling mode of target image')
    gen_parser.add_argument('--keepsize', action='store_true',
                            help='Whether to keep the same size as the target image or not')
    # tesing
    test_parser = subparsers.add_parser(
        'test', help='Test the image/pixel approximation')
    test_parser.add_argument('--source', type=str, default=None,
                             help='Source image directory')
    test_parser.add_argument('--srcsize', type=int, default=32,
                             help='Source image size')
    test_parser.add_argument('--ext', type=str, default='jpg',
                             help='Coma separated string of acceptable image file extensions')
    test_parser.add_argument('--limit', type=int, default=1000,
                             help='Limit the number of source images used')
    test_parser.add_argument('--augment', action='store_false',
                             help='Whether to augment source images or not')
    test_parser.add_argument('--samples', type=int, default=5,
                             help='The number of random colors to approximate')

    args = parser.parse_args()
    if (args.subcommand == 'gen'):
        gen(args)
    elif (args.subcommand == 'test'):
        test(args)
