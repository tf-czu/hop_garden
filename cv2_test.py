import cv2 as cv
import numpy as np


def show(im):
    cv.namedWindow("Img", cv.WINDOW_NORMAL)
    cv.imshow('Img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


im = np.zeros((20, 20), np.uint8)
cv.rectangle(im, (5,5), (14,14), 255, -1)

element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
im2 = cv.erode(im, element)

show(im)
show(im2)