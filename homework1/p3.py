import os
import sys
import cv2
import math
import numpy as np

def nearest_neighbor(new_image, new_h, new_w, image, h, w):
    hn, wn = round(h), round(w)
    if hn >= 0 and hn < image.shape[0] and wn >= 0 and wn < image.shape[1]:
        new_image[new_h, new_w] = image[hn, wn]

def bilinear(new_image, new_h, new_w, image, h, w):
    floor_h, floor_w = math.floor(h), math.floor(w)
    dh, dw = h - floor_h, w - floor_w
    if floor_h >= 0 and (floor_h + 1) < image.shape[0] and \
        floor_w >= 0 and (floor_w + 1) < image.shape[1]:
        new_image[new_h, new_w] = (1 - dh) * (1 - dw) * image[floor_h, floor_w] \
                                + dh * (1 - dw) * image[floor_h + 1, floor_w] \
                                + (1 - dh) * dw * image[floor_h, floor_w + 1] \
                                + dh * dw * image[floor_h + 1, floor_w + 1]
    else:
        nearest_neighbor(new_image, new_h, new_w, image, h, w)

def bicubic_conv(images, t):
    ts = np.array([[1, t, t ** 2, t ** 3]])
    weights = np.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]])

    p = np.dot(weights, images)
    p = 0.5 * np.dot(ts, p)
    return p

def bicubic(new_image, new_h, new_w, image, h, w):
    floor_h, floor_w = math.floor(h), math.floor(w)
    dh, dw = h - floor_h, w - floor_w
    if floor_h >= 1 and (floor_h + 2) < image.shape[0] and \
       floor_w >= 1 and (floor_w + 2) < image.shape[1]:
        b_1 = bicubic_conv(image[floor_h - 1:floor_h + 3, floor_w - 1], dh)
        b0 = bicubic_conv(image[floor_h - 1:floor_h + 3, floor_w], dh)
        b1 = bicubic_conv(image[floor_h - 1:floor_h + 3, floor_w + 1], dh)
        b2 = bicubic_conv(image[floor_h - 1:floor_h + 3, floor_w + 2], dh)
        bs = np.array([b_1, b0, b1, b2])
        new_image[new_h, new_w] = bicubic_conv(bs, dw)
    else:
        nearest_neighbor(new_image, new_h, new_w, image, h, w)

def rotate(image, angle, interpolation='nearest_neighbor'):
    new_rows, new_cols = image.shape[0], image.shape[1]
    mid_row, mid_col = new_rows // 2, new_cols // 2
    new_image = np.zeros_like(image, dtype=np.uint8)
    sin, cos = np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)

    new_h = np.arange(0, new_rows, 1)
    new_w = np.arange(0, new_cols, 1)
    new_hh, new_ww = np.meshgrid(new_h, new_w)
    new_hw = np.array([new_hh.flatten(), new_ww.flatten()]).T
    h = (new_hw[:, 0] - mid_row) * cos + (new_hw[:, 1] - mid_col) * sin + mid_row
    w = -(new_hw[:, 0] - mid_row) * sin + (new_hw[:, 1] - mid_col) * cos + mid_col

    for i in range(len(new_hw)):
        if interpolation == 'nearest_neighbor': 
            nearest_neighbor(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])
        elif interpolation == 'bilinear':
            bilinear(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])
        elif interpolation == 'bicubic':
            bicubic(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])


    return new_image

def resize(image, dsize, fx=1, fy=1, interpolation='nearest_neighbor'):
    # Use dsize
    if dsize != None and len(dsize) == 2:
        rows, cols = dsize
    # Use fx
    elif fx != None and fy != None:
        rows, cols = int(image.shape[0] * fx), int(image.shape[1] * fy)
    else:
        print('The proposed size information is not feasible.')
        return image

    new_image = np.zeros((rows, cols))
    scale_row, scale_col = image.shape[0] * 1.0 / rows, image.shape[1] * 1.0 / cols

    new_h = np.arange(0, rows, 1)
    new_w = np.arange(0, cols, 1)
    new_hh, new_ww = np.meshgrid(new_h, new_w)
    new_hw = np.array([new_hh.flatten(), new_ww.flatten()]).T
    h = new_hw[:, 0] * scale_row
    w = new_hw[:, 1] * scale_col

    for i in range(len(new_hw)):
        if interpolation == 'nearest_neighbor': 
            nearest_neighbor(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])
        elif interpolation == 'bilinear':
            bilinear(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])
        elif interpolation == 'bicubic':
            bicubic(new_image, new_hw[i, 0], new_hw[i, 1], image, h[i], w[i])

    return new_image

def black_region(image):
    black = {'row': [], 'col': []}

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if image[h, w] <= 30:
                black['row'].append(h)
                black['col'].append(w)

    return black

def image_registration(ori, tra):
    ones = np.ones((4, 1))
    trans = np.concatenate((tra, tra[:, 0:1] * tra[:, 1:], ones), axis=1)
    return np.dot(ori.T, np.linalg.inv(trans.T))

if __name__ == '__main__':
    print('Default image path: ./images')
    path = './images'
    if not os.path.exists(os.path.join(path, 'problem_3')):
        os.mkdir(os.path.join(path, 'problem_3'))

    if len(sys.argv) != 6:
        print('Please regard the following info as arguments:')
        print('\t<image_name> <rotate_degree> <fx> <fy> <interplation>\n')
        print('\t<image_name>\tstring')
        print('\t<rotate_degree>\tinterger, from -359 to 359, used for rotating')
        print('\t<fx> <fy>\tfloat or interger or None, used for resizing')
        print('\t<interplation>\tstring, used for rotating and resizing\n')
    else:
        img = cv2.imread(os.path.join(path, sys.argv[1]), cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)

        # Rotate image
        rot_img = rotate(img, int(sys.argv[2]), sys.argv[5])
        cv2.imwrite(os.path.join(path, 'problem_3', '3_1_rotate.png'), rot_img)

        # Resize image
        res_img = resize(img, None, float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
        cv2.imwrite(os.path.join(path, 'problem_3', '3_2_resize.png'), res_img)

    ### Problem 3-2 ###
    # Compute image registration
    ori_t = cv2.imread(os.path.join(path, 'T.png'), cv2.IMREAD_GRAYSCALE)
    ori_t = np.asarray(ori_t)
    tra_t = cv2.imread(os.path.join(path, 'T_transformed.png'), cv2.IMREAD_GRAYSCALE)
    tra_t = np.asarray(tra_t)

    ori_ctrl = []
    ori_black = black_region(ori_t)
    ori_ctrl.append([ori_black['row'][0], ori_black['col'][0]])
    w_max = np.argmax(ori_black['col'])
    ori_ctrl.append([ori_black['row'][w_max], ori_black['col'][w_max]])
    h_max = np.argmax(ori_black['row'])
    ori_ctrl.append([ori_black['row'][h_max], ori_black['col'][h_max]])
    ori_ctrl.append([ori_black['row'][-1], ori_black['col'][-1]])

    tra_ctrl = []
    tra_black = black_region(tra_t)
    tra_ctrl.append([tra_black['row'][0], tra_black['col'][0]])
    w_max = np.argmax(tra_black['col'])
    tra_ctrl.append([tra_black['row'][w_max], tra_black['col'][w_max]])
    w_min = np.argmin(tra_black['col'][len(tra_black['col'])//2:]) + len(tra_black['col'])//2
    tra_ctrl.append([tra_black['row'][w_min], tra_black['col'][w_min]])
    tra_ctrl.append([tra_black['row'][-1], tra_black['col'][-1]])

    reg = image_registration(np.asarray(ori_ctrl), np.asarray(tra_ctrl))
    print(f'registration matrix =\n{reg}')

    new_t = np.ones_like(ori_t) * 255
    for new_h in range(new_t.shape[0]):
        for new_w in range(new_t.shape[1]):
            tra = np.array([[new_h], [new_w], [new_h*new_w], [1]])
            ori = np.dot(reg, tra).astype(np.int16)
            h = ori[0, 0]
            w = ori[1, 0]
            if h >= 0 and h < ori_t.shape[0] and w >= 0 and w < ori_t.shape[1]:
                new_t[new_h, new_w] = ori_t[h, w]
    cv2.imwrite(os.path.join(path, 'problem_3', '3_3_T_registraion_test.png'), new_t)