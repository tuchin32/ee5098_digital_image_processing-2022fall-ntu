import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_hsi(h, s, i, shape):
    x = i * (1 - s)

    scale = np.pi / 180
    cos_num = s * np.cos(h * scale)
    cos_den = np.cos((60 - h) * scale)
    y = np.divide(cos_num, cos_den, out=np.zeros(shape), where=(cos_den != 0))
    y = i * (1 + y)
    
    z = 3 * i - (x + y)

    return x, y, z

def hsi2bgr(his_image):
    rows, cols = his_image.shape[0], his_image.shape[1]
    bgr_image = np.zeros_like(his_image)
    h, s, i = cv2.split(his_image / 255)
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    h = h * 360
    rgb_list = ['br', 'gb', 'rg']
    sector = {'br': h < 360, 'gb': h < 240, 'rg': h < 120} 
    b_sec = {'br': None, 'gb': None, 'rg': None}
    g_sec = {'br': None, 'gb': None, 'rg': None}
    r_sec = {'br': None, 'gb': None, 'rg': None}

    g_sec['br'], b_sec['br'], r_sec['br'] = convert_hsi(h - 240, s, i, (rows, cols))
    r_sec['gb'], g_sec['gb'], b_sec['gb'] = convert_hsi(h - 120, s, i, (rows, cols))
    b_sec['rg'], r_sec['rg'], g_sec['rg'] = convert_hsi(h, s, i, (rows, cols))

    for color in rgb_list:
        # e.g. b[br_sector] = b_sec_br[br_sector] 
        b[sector[color]] = b_sec[color][sector[color]]
        g[sector[color]] = g_sec[color][sector[color]]
        r[sector[color]] = r_sec[color][sector[color]]
    
    bgr_image[:, :, 0] = b * 255
    bgr_image[:, :, 1] = g * 255
    bgr_image[:, :, 2] = r * 255

    return bgr_image

def bgr2hsi(bgr_image):
    rows, cols = bgr_image.shape[0], bgr_image.shape[1]
    hsi_image = np.zeros_like(bgr_image)

    # RGB values have been normalized to the range [0, 1]
    b, g, r = cv2.split(bgr_image / 255)

    # Compute H: compute theta and comapre the R value with G value
    cos_num = 0.5 * (2 * r - g - b)
    cos_den = ((r - g) ** 2 + (r - b) * (g - b)) ** 0.5
    cos_val = np.divide(cos_num, cos_den, out=np.zeros((rows, cols)), where=(cos_den != 0))
    cos_val[cos_val > 1] = 1
    cos_val[cos_val < -1] = -1
    theta = np.arccos(cos_val)
    h = np.copy(theta)
    h[b > g] = 2 * np.pi - theta[b > g]
    h = h / (2 * np.pi)

    # Compute I
    i = (r + g + b) / 3

    # Compute S
    rgb_min = np.minimum(np.minimum(r, g), b)
    s = 1 - np.divide(rgb_min, i, out=np.zeros((rows, cols)), where=(i != 0))

    hsi_image[:, :, 0] = h * 255
    hsi_image[:, :, 1] = s * 255
    hsi_image[:, :, 2] = i * 255

    return hsi_image

def gamma_correction(image, scale=1.0, gamma=1.0):
    norm_img = image / 255.0
    new_img = np.copy(image)

    for c in range(3):
        gc_img = scale * np.power(norm_img[:, :, c], gamma)

        min, max = np.min(gc_img), np.max(gc_img)
        new_img[:, :, c] = (gc_img - min) / (max - min) * 255

    return new_img.astype(np.uint8)

def histogram_equalization(image, const=0.1, gamma=0.9):
    new_image = np.copy(image)
    h, s, i = cv2.split(image / 255)

    # Affine the saturation
    s = s + const
    s[s > 1] = 1

    # Gamma correction of intensity
    i = np.power(i, gamma)

    new_image[:, :, 0] = h * 255
    new_image[:, :, 1] = s * 255
    new_image[:, :, 2] = i * 255
    return new_image

def plot_graph(img1, img2, title='Figure', title1='Image1', title2='Image2', path='./images'):
    plt.figure(title, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def main(path='./images'):
    # Read image
    img = cv2.imread(os.path.join(path, 'steak.jpg'))
    img = np.asarray(img)
    print(f'The shape of \'steak.jpg\': {img.shape}')

    # Image enhancement
    gam_img = gamma_correction(img, 1, 1.4)
    hsi_img = bgr2hsi(gam_img)
    new_img = histogram_equalization(hsi_img, const=0.002, gamma=0.88)
    bgr_img = hsi2bgr(new_img)

    # Show the result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plot_graph(img, bgr_img, '1_enhanced_steak', 'original', 'enhanced', path)

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images')