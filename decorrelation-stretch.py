# modified from https://github.com/lbrabec/decorrstretch/

import sys
from functools import reduce

# TODO: remove usage of cv2 for arcgis pro compat
import cv2
import numpy as np


def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image

    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- float, 1 - 99; upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1,3)).astype(np.float64)
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
    # TODO: figure out tiffs and single band rasters
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # TODO: figure out why sometimes low and high become equal and write better comments
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:,:,b], tol), np.percentile(B[:,:,b], 100-tol)
            if low == high:
                # don't accept too high of contrast value
                print("contrast value too high")
            else:
                # update contrast
                B[B<low] = low
                B[B>high] = high
        # rescale the color values to 0..255
        diff = B[:,:,b].max() - B[:,:,b].min()
        if diff != 0:
            B[:,:,b] = 255 * (B[:,:,b] - B[:,:,b].min())/diff
        else:
            B[:,:,b] = B[:,:,b] - B[:,:,b].min()
    # return it as uint8 (byte) image
    out = B.astype(np.uint8)
    # TODO: consider output file name command line argument option
    cv2.imwrite("out.png", out)


if __name__ == "__main__":
    file = sys.argv[1]
    tol = float(sys.argv[2]) / 2 if len(sys.argv) > 2 else None
    img = cv2.imread(file)
    img_arr = np.asarray(img)
    decorrstretch(img_arr, tol)
