"""
    Demo code for hop plants detection
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

NUM_ROWS = 34
MAX_DIST_TO_LINE = 5  # pixels, working for 1867 x 3402 image


def rows_from_file(rows, im_shape):
    f = open(rows)
    ret = []
    org_im_shape = None
    scale_x = None
    scale_y = None

    for line in f:
        if not org_im_shape:
            org_im_shape = eval(line)
            assert len(org_im_shape) == 2, org_im_shape
            if im_shape != org_im_shape:
                scale_x = im_shape[1] / org_im_shape[1]
                scale_y = im_shape[0] / org_im_shape[0]
            continue
        x_s, y_s = line.split(";")
        x = np.asarray(eval(x_s))
        if scale_x:
            x = x * scale_x
        y = np.asarray(eval(y_s))
        if scale_y:
            y = y * scale_y
        ret.append(np.asarray([x,y]))

    return ret


def save_hop_rows(hop_rows, im_shape):
    f = open("logs/detected_rows.txt", "w")
    f.write(str(im_shape))  # write image resolution
    f.write("\r\n")
    for row in hop_rows:
        x, y = row
        f.write(str(list(x)))
        f.write(";")
        f.write(str(list(y)))
        f.write("\r\n")
    f.close()


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


def rotate_image(bin_im, a_deg):
    rot_matrix = cv2.getRotationMatrix2D((N / 2, M / 2), a_deg, 1)
    rot_bin_im = cv2.warpAffine(bin_im, rot_matrix, (N, M))

    return rot_bin_im


def get_row(x, y):
    while True:
        coeffs = np.polyfit(x, y, 2)
        p = np.poly1d(coeffs)
        yp = p(x)

        dist = abs(yp - y)
        if max(dist) < MAX_DIST_TO_LINE:
            return coeffs
        max_diff_id = np.argmax(dist)
        x.pop(max_diff_id)
        y.pop(max_diff_id)


def rotate_points(xx, yy, a_deg):
    x_c = N/2
    y_c = M/2
    cos = np.cos(np.deg2rad(a_deg))
    sin = np.sin(np.deg2rad(a_deg))
    xx_r = cos * (xx-x_c) - sin * (yy-y_c) + x_c
    yy_r = sin * (xx-x_c) + cos * (yy-y_c) + y_c

    return  xx_r, yy_r


def detect_rows(bin_im):
    # irregular plot shape makes problem with angle calculation
    sample = bin_im[570:3100, :]  # The image section used for angle calculation. Tested on "Biochmel_rgb_200427.tif".
    # https://en.wikipedia.org/wiki/Image_moment#Examples_2
    moments = cv2.moments(sample)
    mu11 = moments["mu11"]
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    a = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))  # radians
    a_deg = np.rad2deg(a)
    print("Calculated angle: %.3f" % a_deg)
    rot_bin_im = rotate_image(bin_im, a_deg)

    num_wpixels_rows = np.sum(rot_bin_im, axis=1)
    #print(num_wpixels_rows.size)
    num_wpixels_rows = np.convolve(num_wpixels_rows, np.ones(20) / 20, mode="same")  # smooth data
    # num_wpixels_rows = np.resize(num_wpixels_rows, num_wpixels_rows.size//10)
    if g_debug:
        plt.plot(num_wpixels_rows)
        plt.show()
    mask_rows = num_wpixels_rows > 1e4  # Threshold for rows detection, may be modified.
    # It corresponds with number of green pixels times 255.
    edge = np.diff(mask_rows)
    assert sum(edge) == 2 * NUM_ROWS, "Number of rows should be %d, detected: %f" % (NUM_ROWS, sum(edge) / 2)

    edge_positions = np.where(edge == True)[0]
    edge_positions = np.reshape(edge_positions, (edge_positions.size // 2, 2))

    # detect contour, used for rows detection
    __, rot_bin_im2 = cv2.threshold(rot_bin_im, 127, 255, cv2.THRESH_BINARY)  # again, it was damaged during rotation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    rot_bin_im2 = cv2.morphologyEx(rot_bin_im2, cv2.MORPH_OPEN, element)  # removes very small objects
    contours, __ = cv2.findContours(rot_bin_im2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of green splotchs: %d" %(len(contours)))
    # cv2.drawContours(rot_bin_im2, contours, -1, (255), 1)

    if g_debug:
        show_im(rot_bin_im2)
        cv2.imwrite("logs/rot_bin_image.png", rot_bin_im2)

    centroids = []
    for cnt in contours:
        cnt_moments = cv2.moments(cnt)
        xc = cnt_moments["m10"] / cnt_moments["m00"]
        yc = cnt_moments["m01"] / cnt_moments["m00"]
        centroids.append([xc, yc])
    centroids = np.asarray(centroids)
    centroids_y = centroids[:, 1]

    sorted_centroids = []
    for y_start, y_end in edge_positions:
        mask = np.logical_and((centroids_y >= y_start), (centroids_y <= y_end))
        sorted_centroids.append(np.sort(np.asarray(centroids[mask]),0))

    rows = []
    for c_points in sorted_centroids:
        x = [c[0] for c in c_points]
        y = [c[1] for c in c_points]
        row_coeff = get_row(x, y)
        # get poits
        p = np.poly1d(row_coeff)
        xx = np.arange(x[0], x[-1])
        yy = p(xx)
        # rotate points back, according to orig. image
        xx_r, yy_r = rotate_points(xx, yy, a_deg)
        xx_r = np.round(xx_r).astype(np.uint32)
        yy_r = np.round(yy_r).astype(np.uint32)
        rows.append(np.array([xx_r, yy_r]))

    return rows


def detect_plants(im, rows):
    norm_g = norm_green(im)
    bin_im = threshold(norm_g)
    if rows:
        hop_rows = rows_from_file(rows, bin_im.shape)
    else:
        hop_rows = detect_rows(bin_im)
        save_hop_rows(hop_rows, bin_im.shape)

    # draw hop_rows
    im2 = im.copy()
    for row in hop_rows:
        row = row.T
        row = row.reshape((-1,1,2))
        cv2.polylines(im2, [np.int32(row)], False, (0, 0, 255))

    if g_debug:
        show_im(bin_im)
        cv2.imwrite("logs/bin_image.png", bin_im)
    show_im(im2)
    cv2.imwrite("logs/rows_in_im.png", im2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imfile', help='path to image file')
    parser.add_argument('--rows-file', help='path to file with rows')
    parser.add_argument('--debug', '-d', help='shows debug graphs and images', action='store_true')
    args = parser.parse_args()

    os.makedirs("logs", exist_ok = True)

    if args.rows_file:
        rows = args.rows_file
    else:
        rows = None
    g_debug = args.debug

    im = cv2.imread(args.imfile)
    if im is not None:
        M, N, K = im.shape
        print("Resolution: %d, %d, %d" %(M, N, K))
        detect_plants(im, rows)
    else:
        print("No image in path: %s" %args.imfile)
