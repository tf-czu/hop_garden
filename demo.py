"""
    Demo code for hop plants detection
"""
import cv2
import numpy as np


def norm_green(im):
    """
    returns normalized green 8 bit image:
    Gn = G/(R+G+R) * 255
    """
    b, g, r = cv2.split(im)
    b = b.astype(float)
    g = g.astype(float)
    r = r.astype(float)
    sum_arr = b + g + r
    sum_arr[sum_arr == 0] = 1  # avoid division by 0
    norm_g = g / sum_arr * 255
    return norm_g.astype(np.uint8)


def show_im(im):
    cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    cv2.imshow('Img', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def threshold(im):
    im_1d = im.ravel()
    # Image contains an unnecessary black background
    mask = im_1d != 0
    thr_value, th = cv2.threshold(im_1d[mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Used threshold value: %f" %thr_value)
    bin = np.zeros(im_1d.shape, dtype=np.uint8)
    bin[mask] = th.ravel()
    return np.reshape(bin, im.shape)


def detect_plants(im_file):
   im = cv2.imread(im_file)
   norm_g = norm_green(im)
   bin_im = threshold(norm_g)

   show_im(bin_im)
   cv2.imwrite("bin_image.png", bin_im)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imfile', help='path to image files')
    args = parser.parse_args()

    detect_plants(args.imfile)
