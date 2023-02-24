import os
import cv2
import numpy as np

def gaussian_kernel(size=256, sigma=128, twodim=False):
    mid = size // 2
    if twodim == True:
        x, y = np.mgrid[-mid:mid, -mid:mid]
        gk = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    else:
        x = np.arange(-mid, mid)
        gk = np.exp(-0.5 * (x ** 2) / (sigma ** 2))
    return gk / gk.sum()

def main(kernel_size=256, two_dim=True,
         path='./images'):
    # 1. Read image
    img = cv2.imread(os.path.join(path, 'image_4.tif'), cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img)
    if not os.path.exists(os.path.join(path, 'problem_4')):
        os.mkdir(os.path.join(path, 'problem_4'))
    cv2.imwrite(os.path.join(path, 'problem_4', '4_1_chess.png'), img)

    # 2. Generate gaussian filter
    if two_dim == True:
        gauss = gaussian_kernel(kernel_size, 128, True)
    else:
        gauss = gaussian_kernel(kernel_size, 128, False)

    # 3. Generate image with padding
    rows, cols = img.shape
    pad_img = np.zeros((rows + kernel_size, cols + kernel_size))
    start = kernel_size // 2 + 1
    pad_img[start: start + rows, start: start + cols] = img

    pad_img[:start, start: start + cols] = np.repeat(img[0:1, :], start, axis=0)
    pad_img[start + rows:, start: start + cols] = np.repeat(img[-1:, :], start - 2, axis=0)
    pad_img[start: start + rows, :start] = np.repeat(img[:, 0:1], start, axis=1)
    pad_img[start: start + rows, start + cols:] = np.repeat(img[:, -1:], start - 2, axis=1)
    cv2.imwrite(os.path.join(path, 'problem_4', '4_2_chess_padding.png'), pad_img)

    # 4. Compute the shade
    shade = np.zeros_like(img)
    if two_dim == True:
        for h in range(rows):
            for w in range(cols):
                shade[h, w] = np.sum(pad_img[h: h + kernel_size, w: w + kernel_size] * gauss)
    else:
        tmp = np.zeros_like(pad_img)
        for h in range(rows):
            for w in range(cols):
                tmp[h + start, w + start] = np.sum(pad_img[h: h + kernel_size, w + start] * gauss)
        for h in range(rows):
            for w in range(cols):
                shade[h, w] = np.sum(tmp[h + start, w: w + kernel_size] * gauss)

    cv2.imwrite(os.path.join(path, 'problem_4', '4_3_shade.png'), shade)

    # 5. Acquire the result with shade correction
    correct_img = img / (shade + 1)
    correct_img *= 255 / (np.max(correct_img) - np.min(correct_img))
    cv2.imwrite(os.path.join(path, 'problem_4', '4_4_shade_correction.png'), correct_img)
            
if __name__ == '__main__':
    kernel_size = 256   # Size value is even in this case
    two_dim = True
    path = './images'

    main(kernel_size, two_dim, path)
    print('Check the directory: ./images/problem_4')