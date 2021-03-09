"""
    Demo code for hop plants detection
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json

NUM_ROWS = 34
SECTION_POINTS = [0,-1,570,3100]

#PIX_SIZE = 36/555  # m, for 1867 x 3402 image
PIX_SIZE = 0.01  # 36/555 * 3402/22084  # m, increase pixel size for 12124 x 22084 resolution
EXPECT_ROW_WIDTH = 0.6  # m, expected row width (plants are sought there)
MIN_PLANT_AREA = 0.01  # m^2
MAX_PLANT_DIST = 1.5  # m
MIN_PLANT_DIST = 0.6  # m
# element size is 3x3 px, one iteration corresponds to 1 px, formula is (distance to remove in meters) / pixel size
NUM_ERODE_ITER = int(round(0.05/PIX_SIZE))
SMOOTH = int(0.06/PIX_SIZE *20)
MAX_DIST_TO_LINE = int(round(0.06/PIX_SIZE * 5))  # pixels

N = M = 20 # for test only


def parse_im_data(im_data):
    with open(im_data) as f:
        data = json.load(f)
        plot_cnt = np.array(data["plot_cnt"])
        rot_rec = np.array(data["rot_rec"])

    return plot_cnt, rot_rec


def remove_parallel_cnt(sorted_size_data, angle):
    remove_idx = []
    for ii, rec_l in enumerate(sorted_size_data):
        if ii + 1 == len(sorted_size_data):
            break
        rec_r = sorted_size_data[ii+1]
        rec_l = cv2.boxPoints(rec_l)
        rec_r = cv2.boxPoints(rec_r)
        # draw pic
        """
        bb = np.zeros((20, 20), np.uint8)
        print(row)
        cv2.polylines(bb, [row], False, 255, thickness=1)
        rec_l_int = np.int32(rec_l)
        rec_r_int = np.int32(rec_r)
        cv2.drawContours(bb, [rec_l_int, rec_r_int], -1, 255, 1)
        show_im(bb)
        """
        rec_l_xR, rec_l_yR = rotate_points(rec_l[:, 0], rec_l[:, 1], angle)
        rec_r_xR, rec_r_yR = rotate_points(rec_r[:, 0], rec_r[:, 1], angle)
        len_rec_l = max(rec_l_xR) - min(rec_l_xR)
        len_rec_r = max(rec_r_xR) - min(rec_r_xR)

        if (len_rec_l > len_rec_r) and (max(rec_l_xR) > max(rec_r_xR)):
            remove_idx.append(ii+1)
        elif (len_rec_l <= len_rec_r) and (min(rec_r_xR) < min(rec_l_xR)):
            remove_idx.append(ii)

    ret_idx = np.arange(len(sorted_size_data))

    return np.delete(ret_idx, remove_idx)


def remove_soclose_cnt(dist_diff, sorted_areas):
    pass


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
    angle = None
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
        if not angle:
            angle = float(line)
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

    return ret, angle


def save_hop_rows(hop_rows, im_shape, angle):
    f = open("logs/detected_rows.txt", "w")
    f.write(str(im_shape) + "\r\n")  # write image resolution
    f.write("%.3f\r\n" %angle)
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


def sort_plants(row, centroids):
    # Row is almost perfect line so ignore curvature and use the first point only.
    start = row[0,:]
    dist_to_start = np.linalg.norm(centroids - start, axis=1)
    sort_idx = np.argsort(dist_to_start)

    return sort_idx, dist_to_start[sort_idx]


def check_space(r_start, r_end):
    r_start = cv2.boxPoints(r_start)
    r_end = cv2.boxPoints(r_end)
    min_dist = []
    min_dist_argv = []
    for xy in r_start:
        dists = np.linalg.norm(r_end - xy, axis=1)
        min_dist.append(np.min(dists))
        min_dist_argv.append(np.argmin(dists))

    min_pdist = min(min_dist)
    point_id_start = np.argmin(min_dist)
    point_id_end = min_dist_argv[point_id_start]
    point_start = r_start[point_id_start]
    point_end = r_end[point_id_end]

    return np.int32(point_start), np.int32(point_end), min_pdist


def detect_rows(bin_im, section, plot_cnt):
    # irregular plot shape makes problem with angle calculation
    # use section cnt
    sample = np.zeros(bin_im.shape, np.uint8)
    cv2.drawContours(sample, [section], 0, (255), -1)
    # https://en.wikipedia.org/wiki/Image_moment#Examples_2
    moments = cv2.moments(sample)
    mu11 = moments["mu11"]
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    a = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))  # radians
    a_deg = np.rad2deg(a)
    if g_user_angle:
        a_deg = g_user_angle
    print("Calculated angle: %.3f" % a_deg)
    rot_bin_im = rotate_image(bin_im, a_deg)
    if g_debug:
        show_im(rot_bin_im)
        cv2.imwrite("logs/rot_bin_image.png", rot_bin_im)

    num_wpixels_rows = np.sum(rot_bin_im, axis=1)
    #print(num_wpixels_rows.size)
    num_wpixels_rows = np.convolve(num_wpixels_rows, np.ones(SMOOTH) / SMOOTH, mode="same")  # smooth data
    # num_wpixels_rows = np.resize(num_wpixels_rows, num_wpixels_rows.size//10)
    if g_debug:
        plt.plot(num_wpixels_rows)
        plt.show()
    mask_rows = num_wpixels_rows > g_thr_num_wpixels  # Threshold for rows detection, may be modified.
    # It corresponds with number of green pixels times 255.
    edge = np.diff(mask_rows)
    assert sum(edge) == 2 * NUM_ROWS, "Number of rows should be %d, detected: %f" % (NUM_ROWS, sum(edge) / 2)

    edge_positions = np.where(edge == True)[0]
    edge_positions = np.reshape(edge_positions, (edge_positions.size // 2, 2))

    # detect contour, used for rows detection
    __, rot_bin_im2 = cv2.threshold(rot_bin_im, 127, 255, cv2.THRESH_BINARY)  # again, it was damaged during rotation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rot_bin_im2 = cv2.morphologyEx(rot_bin_im2, cv2.MORPH_OPEN, element, iterations=NUM_ERODE_ITER)  # removes very small objects
    contours, __ = cv2.findContours(rot_bin_im2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of green splotchs: %d" %(len(contours)))
    # cv2.drawContours(rot_bin_im2, contours, -1, (255), 1)

    if g_debug:
        show_im(rot_bin_im2)
        cv2.imwrite("logs/rot_bin_image2.png", rot_bin_im2)

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
    plot_mask = np.zeros(bin_im.shape, np.uint8)
    cv2.drawContours(plot_mask, [plot_cnt], 0, 1, -1)
    plot_mask = plot_mask.astype(bool)
    for c_points in sorted_centroids:
        x = [c[0] for c in c_points]
        y = [c[1] for c in c_points]
        row_coeff = get_row(x, y)
        # get poits
        p = np.poly1d(row_coeff)
        xn = rot_bin_im2.shape[1] -1  # last pixel row in the image
        xx = np.arange(0, xn)  # use the full width of the rot_image
        yy = p(xx)
        # rotate points back, according to orig. image
        xx_r, yy_r = rotate_points(xx, yy, a_deg)
        xx_r = np.int32(np.round(xx_r))
        yy_r = np.int32(np.round(yy_r))

        # remove outside values
        idx = np.logical_and(xx_r>0, xx_r<=xn)
        xx_r = xx_r[idx]
        yy_r = yy_r[idx]
        row_mask = np.zeros(bin_im.shape, dtype=bool)
        row_mask[yy_r,xx_r] = True
        cut_row_mask = np.logical_and(row_mask, plot_mask)
        yy_ret, xx_ret = np.where(cut_row_mask)

        rows.append(np.array([xx_ret, yy_ret], np.int32))

    return rows, a_deg


def blank_spaces_detect(bin_im, hop_rows, org_image, angle):
    row_width_px = int(round(EXPECT_ROW_WIDTH / PIX_SIZE))
    # print used constants
    print("EXPECT_ROW_WIDTH: %.2f m" % EXPECT_ROW_WIDTH)
    print("row_width_px: %d" % row_width_px)
    print("MIN_PLANT_AREA: %f m2" % MIN_PLANT_AREA)
    print("MAX_PLANT_DIST: %f m" % MAX_PLANT_DIST)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # even number of pixel (element size 2 or 4) brings problems with contour shift.
    bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_OPEN, element, iterations=NUM_ERODE_ITER)  # removes very small objects
    bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_CLOSE, element, iterations=NUM_ERODE_ITER)  # removes small holes

    for row in hop_rows:
        im_mask = np.zeros(bin_im.shape, np.uint8)  # to be area of interest around one row
        one_row = im_mask.copy()
        row = row.T
        row = row.reshape((-1, 1, 2))
        cv2.polylines(im_mask, [row], False, 255, thickness=row_width_px)
        # draw control contours in the org_image
        row_contours, __ = cv2.findContours(im_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(org_image, row_contours, -1, (0, 0, 255), 2)

        mask = im_mask == 255
        one_row[mask] = bin_im[mask]

        _contours, __ = cv2.findContours(one_row.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # TODO ??yes, we need all the points
        contours = []
        areas = []  # areas in m2
        centroids = []
        size_data = []
        for cnt in _contours:
            moments = cv2.moments(cnt)
            ar = moments["m00"] * PIX_SIZE ** 2
            if ar < MIN_PLANT_AREA:  # Note it is possible that a small objects are already removed by morphologyEx function.
                continue
            contours.append(cnt)
            areas.append(ar)
            xc = moments["m10"] / moments["m00"]
            yc = moments["m01"] / moments["m00"]
            centroids.append([xc, yc])
            minA_rec = cv2.minAreaRect(cnt)  # top-left corner(x,y), (width, height), angle of rotation
            size_data.append(minA_rec)

        cv2.drawContours(org_image, contours, -1, (0, 0, 255), 1)
        #print(bin_im.shape, org_image.shape, one_row.shape, len(contours))

        # calculate distances from row start and sort it
        centroids = np.int32(centroids)
        sort_idx, dist_to_start = sort_plants(row, centroids)
        sorted_centroids = centroids[sort_idx,:]
        sorted_size_data = [size_data[ii] for ii in sort_idx]
        sorted_areas = [areas[ii] for ii in sort_idx]
        contours_sorted = [contours[ii] for ii in sort_idx]

        # Filter contoures, try to avoid grass detection
        object_idx = remove_parallel_cnt(sorted_size_data, angle)
        #print(object_idx)
        dist_to_start = dist_to_start[object_idx]
        sorted_centroids = sorted_centroids[object_idx, :]
        sorted_size_data = [sorted_size_data[ii] for ii in object_idx]
        sorted_areas = [sorted_areas[ii] for ii in object_idx]
        contours_sorted = [contours_sorted[ii] for ii in object_idx]
        cv2.drawContours(org_image, contours_sorted, -1, (255, 0, 0), 1)  # cnt used for space detection

        dist_to_start = dist_to_start*PIX_SIZE  # dist in meters
        dist_diff = np.diff(dist_to_start)

        big_dist_id = np.where(dist_diff > MAX_PLANT_DIST)[0]
        big_dist_id_end = big_dist_id + 1

        spaces_start = sorted_centroids[big_dist_id]
        spaces_end = sorted_centroids[big_dist_id_end]
        rec_start = [sorted_size_data[ii] for ii in big_dist_id]
        rec_end = [sorted_size_data[ii] for ii in big_dist_id_end]

        # Draw spaces in the image
        for item in zip(spaces_start, spaces_end, rec_start, rec_end):
            start, end, r_start, r_end = item  # TODO do we need centroids here?
            start2, end2, pdist = check_space(r_start, r_end)
            pdist_m = pdist * PIX_SIZE  # dist in meters
            if pdist_m < MAX_PLANT_DIST - 0.2:
                continue
            cv2.line(org_image, tuple(start2), tuple(end2), (255, 0, 0), 2)

    show_im(org_image)
    cv2.imwrite("logs/org_image.png", org_image)


def detect_plants(im, im_data, rows):
    plot_cnt, section = parse_im_data(im_data)
    norm_g = norm_green(im)
    bin_im = threshold(norm_g)
    if g_debug:
        show_im(bin_im)
        cv2.imwrite("logs/bin_image.png", bin_im)

    if rows:
        hop_rows, angle = rows_from_file(rows, bin_im.shape)
    else:
        if g_debug:
            cv2.drawContours(im, [section], 0, (255,0,255), 2)
            show_im(im)
        hop_rows, angle = detect_rows(bin_im, section, plot_cnt)
        save_hop_rows(hop_rows, bin_im.shape, angle)
    if g_debug:
        # draw hop_rows
        im2 = im.copy()
        for row in hop_rows:
            row = row.T
            row = row.reshape((-1,1,2))
            cv2.polylines(im2, [row], False, (0, 0, 255), thickness=2)
            cv2.drawContours(im2, [plot_cnt], 0, (255,0,0), 2)

        show_im(im2)
        cv2.imwrite("logs/rows_in_im.png", im2)

    # detect blank spaces
    blank_spaces_detect(bin_im.copy(), hop_rows, im.copy(), angle)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imfile', help='path to image file')
    parser.add_argument('--im-data', help='Json file with image information for data processing')
    parser.add_argument('--rows-file', help='path to file with rows')
    parser.add_argument('--section', help='An image section used for angle calculation, default: "0,-1,570,3100"')
    parser.add_argument('--user-angle', help='User defined angle for rows detection')
    parser.add_argument('--debug', '-d', help='Shows debug graphs and images', action='store_true')
    parser.add_argument('--thr', help='Threshold for rows detection', default=6e4, type=float)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok = True)

    if args.rows_file:
        rows = args.rows_file
    else:
        rows = None
    g_debug = args.debug

    if args.section:
        g_section = args.section.split(",")
        g_section = [int(num) for num in g_section]
    else:
        g_section = SECTION_POINTS
    g_user_angle = False
    if args.user_angle:
        g_user_angle = float(args.user_angle)
    if args.thr:
        g_thr_num_wpixels = args.thr

    im = cv2.imread(args.imfile)
    if im is not None:
        M, N, K = im.shape
        print("Resolution: %d, %d, %d" % (M, N, K))
        detect_plants(im, args.im_data, rows)
    else:
        print("No image in path: %s" % args.imfile)
