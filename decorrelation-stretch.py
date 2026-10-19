# modified from https://github.com/lbrabec/decorrstretch/

import argparse
import ast
import sys
from functools import reduce

import numpy as np
from PIL import Image


def decorrelation_stretch(A, tol):
    """
    Apply decorrelation stretch to image

    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- [[band1_min, band1_max], [band2_min, band2_max], ...] all floats, upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    try:
        num_bands = orig_shape[2]
    except IndexError:
        print("Error: Image must have more than one color band.")
        sys.exit()

    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1,num_bands)).astype(np.float64)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    eigval, V = eigval.real, V.real
    # stretch matrix
    # compute mean of each color
    S = np.diag(1/np.sqrt(eigval))
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for band in range(num_bands):
        # apply contrast stretching if requested
        try:
            # TODO: figure out why sometimes low and high become equal
            # TODO: write better comments
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:,:,band], tol[band][0]), np.percentile(B[:,:,band], tol[band][1])
            if low >= high:
                # don't accept too high of contrast value
                print("contrast value too high")
            else:
                # update contrast
                B[B<low] = low
                B[B>high] = high
        except Exception:  # noqa: BLE001, S110
            pass
        # rescale the color values to 0..255
        diff = B[:,:,band].max() - B[:,:,band].min()
        if diff != 0:
            B[:,:,band] = 255 * (B[:,:,band] - B[:,:,band].min())/diff
        else:
            B[:,:,band] = B[:,:,band] - B[:,:,band].min()
    # return it as uint8 (byte) image
    out = B.astype(np.uint8)
    # TODO: consider output file name command line argument option
    # TODO: figure out why image file size 3x's after running
    im = Image.fromarray(out)
    im.save("out.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file name or path of input image", type=str)
    parser.add_argument("tol", help="TODO", nargs="*", type=ast.literal_eval)

    args = parser.parse_args()

    file = args.file
    tol = args.tol

    img = Image.open(file)
    img_arr = np.asarray(img)
    decorrelation_stretch(img_arr, tol)
    # # uv run decorrelation-stretch.py test/test3.jpg [band1_min, band1_max] [band2_min, band2_max] [band3_min, band3_max]
