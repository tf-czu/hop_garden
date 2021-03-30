import cv2 as cv
import numpy as np
from demo import rotate_points


def show(im):
    cv.namedWindow("Img", cv.WINDOW_NORMAL)
    cv.imshow('Img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def try_erode():
    im = np.zeros((20, 20), np.uint8)
    cv.rectangle(im, (5,5), (14,14), 255, -1)

    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    im2 = cv.erode(im, element)

    show(im)
    show(im2)

def draw_rot_rec():
    size_data = [((15, 25), (20.0, 10.0), 61), ((35, 5), (4.0, 4.0), 0)]
    b = np.zeros((40,40))
    b_r = np.zeros((40,40))
    for r in size_data:
        box = cv.boxPoints(r)
        rec_xR, rec_yR = rotate_points(box[:, 0], box[:, 1], 45)
        print(rec_xR, rec_yR)
        box_r = []
        for x, y in zip(rec_xR, rec_yR):
            box_r.append([x,y])
        box = np.int32(box)
        box_r = np.int32(box_r)
        print(box, box_r)
        cv.drawContours(b, [box], 0, 255, 1)
        cv.drawContours(b_r, [box_r], 0, 255, 1)
    show(b)
    show(b_r)

def show_some_cnt(rec_l_xR, rec_l_yR, rec_r_xR, rec_r_yR,N, M):
    b = np.zeros((N,M), np.uint8)
    box_l = [[x,y] for x, y in zip(rec_l_xR, rec_l_yR)]
    box_r = [[x,y] for x, y in zip(rec_r_xR, rec_r_yR)]
    cv.drawContours(b, [np.int32(box_l)], 0, 255, 3)
    cv.drawContours(b, [np.int32(box_r)], 0, 255, 3)
    show(b)

if __name__ == "__main__":
    print("go")
    draw_rot_rec()