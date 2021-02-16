"""
    Demo code for hop plants detection
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

NUM_ROWS = 34
MAX_DIST_TO_LINE = 5  # pixels, working for 1867 x 3402 image
PIX_SIZE = 36/555  # m, for 1867 x 3402 image
#PIX_SIZE = 36/555 * 22084/3402  # m, increase pixel size for 12124 x 22084 resolution
EXPECT_ROW_WIDTH = 1.2 # m, expected row width (plants are sought there)


def fill_polyline(polyline):
    """assumption: the x is a sequence and x_n+1 > x_n, for each x_n is only one y_n"""
    x, y = polyline
    if len(x)-1 >= x[-1] - x[0]:
        print("fill_polyline: fill polyline is not needed, input data are returned")
        return polyline
    ret_x = [x[0]]
    ret_y = [y[0]]
    for ii in range(len(x) - 1):
        x_diff = x[ii+1] - x[ii]
        y_diff = y[ii+1] - y[ii]
        for jj in range(1, x_diff+1):
            ret_x.append(x[ii] + jj)
            ret_y.append(y[ii] + int(round(jj * y_diff/x_diff)))

    return np.asarray([ret_x, ret_y])


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
        polyline = np.int32(np.round([x,y]))
        xy_points = fill_polyline(polyline)
        ret.append(xy_points)

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
        xx_r = np.round(xx_r)
        yy_r = np.round(yy_r)
        rows.append(np.array([xx_r, yy_r], np.int32))

    return rows


def blank_spaces_detect(bin_im, hop_rows, org_image):
    row_width_px = int(round(EXPECT_ROW_WIDTH / PIX_SIZE))
    # print used constants
    print("EXPECT_ROW_WIDTH: %.2f m" %EXPECT_ROW_WIDTH)
    print("row_width_px: %d" %row_width_px)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # !! the objects are moving ??
    #bin_im2 = cv2.morphologyEx(bin_im, cv2.MORPH_CLOSE, element)  # removes small holes
    #bin_im3 = cv2.morphologyEx(bin_im2, cv2.MORPH_OPEN, element)  # removes very small objects

    for row in hop_rows:
        im_mask = np.zeros(bin_im.shape, np.uint8)  # to be area of interest around one row
        one_row = im_mask.copy()
        row = row.T
        row = row.reshape((-1, 1, 2))
        cv2.polylines(im_mask, [row], False, 255, thickness=row_width_px)
        if g_debug:  # draw control contours in the org_image
            row_contours, __ = cv2.findContours(im_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(org_image, row_contours, -1, (0, 0, 255), 2)

        mask = im_mask == 255
        one_row[mask] = bin_im[mask]

        contours, __ = cv2.findContours(one_row.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if g_debug:
            cv2.drawContours(org_image, contours, -1, (0, 0, 255), 1)
            print(bin_im.shape, org_image.shape, one_row.shape)
            cv2.imwrite("logs/one_row.png", one_row)
            break
        centroids = []
        areas = []  # in m2
        size_data = []
        for cnt in contours:
            moments = cv2.moments(cnt)
            areas.append(moments["m00"] * PIX_SIZE ** 2)
            xc = moments["m10"] / moments["m00"]
            yc = moments["m01"] / moments["m00"]
            centroids.append([xc, yc])
            position, size, angle = cv2.minAreaRect(cnt)  # top-left corner(x,y), (width, height), angle of rotation
            size_data.append([position[0], position[1], angle])

    if g_debug:
        show_im(org_image)
        cv2.imwrite("logs/contours.png", org_image)


def detect_plants(im, rows):
    norm_g = norm_green(im)
    bin_im = threshold(norm_g)
    if g_debug:
        show_im(bin_im)
        cv2.imwrite("logs/bin_image.png", bin_im)

    if rows:
        hop_rows = rows_from_file(rows, bin_im.shape)
    else:
        hop_rows = detect_rows(bin_im)
        save_hop_rows(hop_rows, bin_im.shape)
    if g_debug:
        # draw hop_rows
        im2 = im.copy()
        for row in hop_rows:
            row = row.T
            row = row.reshape((-1,1,2))
            cv2.polylines(im2, [row], False, (0, 0, 255), thickness=2)

        show_im(im2)
        cv2.imwrite("logs/rows_in_im.png", im2)

    # detect blank spaces
    blank_spaces_detect(bin_im.copy(), hop_rows, im.copy())


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
