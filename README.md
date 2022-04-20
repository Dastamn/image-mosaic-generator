# Image Mosaic Generator

Implementation of an image mosaic generator using NumPy. The algorithm takes as input a target image and a directory of source images, and returns the target image where each pixel is replaced by its closest image in the image source set.

To this end, the algorithm computes, for each image in the source set, the average value of each RGB layer and compares them to the target image pixels by computing a Root Mean Square Error. For a better color variation, the source images are augmented by generating: BGR equivalents, higher contrast, grayscale and negative images. Moreover, to better handle larger images, pooling (average and maximum) as well as strides can be used.

## Requirements

- Python 3.7
- NumPy 1.21.5
- Pillow 9.0.1

## Basic Usage

```bash
$ python main.py --source <path_to_image_directory> --target <path_to_image> --poolsize <pooling_size> --stride <stride_size>
```

Check [`main.py`](https://github.com/Dastamn/image-mosaic/blob/main/main.py#L53) to see all the available options.

## Mosaic Sample Using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

<section align='center'>
    <img src='samples/nier.jpg' height='300'/>
    <img src='samples/nier_mosaic.jpg' height='300'/>
</section>
