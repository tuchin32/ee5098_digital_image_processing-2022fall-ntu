import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(r_img, s_img):
    pixels = s_img.shape[0] * s_img.shape[1]
    r = []
    s = []
    for i in range(256):
        r_n = np.sum(r_img == i)
        r.append(float(r_n / pixels))
        s_n = np.sum(s_img == i)
        s.append(float(s_n / pixels))
    
    plt.subplot(2, 1, 1)
    plt.stem(r)
    plt.xlabel('r')
    plt.ylabel('p(r)')
    
    plt.subplot(2, 1, 2)
    plt.stem(s)
    plt.xlabel('s')
    plt.ylabel('p(s)')

    plt.tight_layout()
    plt.show()

def gamma_correction(image, scale=1.0, gamma=1.0):
    norm_img = image / 255.0
    gc_img = scale * np.power(norm_img, gamma)

    min, max = np.min(gc_img), np.max(gc_img)
    new_img = (gc_img - min) / (max - min) * 255

    # new_img = gc_img * 255
    return new_img.astype(np.uint8)

def histogram_equalization(image):
    pixels = image.shape[0] * image.shape[1]
    his_img = np.zeros_like(image)
    pdf = []
    out = []

    for i in range(256):
        n = np.sum(image == i)
        pdf.append(float(n / pixels))
        out.append(int(255 * np.sum(pdf)))
        his_img[image == i] = out[i]

    # plot_histogram(image, his_img)
    return his_img

def ssim(x, y):
    ux, sx = np.mean(x), np.std(x)
    uy, sy = np.mean(y), np.std(y)
    sxy = np.sum((x - ux) * (y - uy)) / (x.shape[0] * x.shape[1] - 1)
    return (2 * ux * uy) * (2 * sxy) / (ux ** 2 + uy ** 2) / (sx ** 2 + sy ** 2)

if __name__ == '__main__':
    # 1. Read image
    path = './images'
    img = cv2.imread(os.path.join(path, 'image_2.png'), cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img)
    if not os.path.exists(os.path.join(path, 'problem_2')):
        os.mkdir(os.path.join(path, 'problem_2'))

    ### Problem 2-1 ###
    # Gamma correction
    gamma_img = gamma_correction(img, 1.0, 2.2)
    cv2.imwrite(os.path.join(path, 'problem_2', '2_1_gamma_corection.png'), gamma_img)

    # Histogram Equalization
    his_img = histogram_equalization(img)
    cv2.imwrite(os.path.join(path, 'problem_2', '2_2_histogram_equlization.png'), his_img)


    ### Problem 2-2 ###
    # Compute gamma value
    gammas = np.arange(0.1, 20, 0.1)
    losses = []
    for gamma in gammas:
        gc_img = gamma_correction(img, 1.0, gamma)
        losses.append(ssim(gc_img, his_img))

    min_index = np.argmax(losses)
    gamma = gammas[min_index]
    print('As gamma = %f, the maximum similarity = %f' % (gamma, losses[min_index]))
    gc_img = gamma_correction(img, 1.0, gamma)
    cv2.imwrite(os.path.join(path, 'problem_2', '2_3_find_gamma.png'), gc_img)

    print('Check the directory: ./images/problem_2')