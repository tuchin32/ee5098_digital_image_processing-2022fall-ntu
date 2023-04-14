import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(size, gamma_h=2.2, gamma_l=0.25, c=1, d0=80):
    # Construct coordinate grid
    rows, cols = size[0], size[1]
    u = np.arange(0, rows)
    v = np.arange(0, cols)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    # High-pass filter
    d = (uu - rows // 2) ** 2 + (vv - cols // 2) ** 2
    homo = (gamma_h - gamma_l) * (1 - np.exp(-c * (d / (d0 ** 2)))) + gamma_l

    if len(size) == 3:
        homo = np.expand_dims(homo, axis=2)
        homo = np.concatenate((homo, homo, homo), axis=2)
    return homo

def filtering(img, param=(2.2, 0.2, 1, 80)):
    # Zero-padding
    img_pad = np.pad(img, ((0, img.shape[0]), (0, img.shape[1])))
    img_pad = np.log(img_pad + 0.01)

    # Fourier transform of input image
    img_fft = np.fft.fft2(img_pad)

    # Transfer function
    rh, rl, c, d0 = param
    homo = homomorphic_filter(img_pad.shape, rh, rl, c, d0)

    # Frequency-domain filtering
    img_out_fft = img_fft * homo

    # Return to spatial-domain
    img_out = np.fft.ifft2(img_out_fft).real
    img_out = img_out[:img.shape[0], :img.shape[1]]
    img_out = np.exp(img_out) - 0.01

    # Normalization
    img_out = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_out

def plot_graph(img, fft, title='Figure', title1='Image', title2='Filtered image', path='./images/problem_2'):
    plt.figure(title)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title1)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(1, 2, 2)
    plt.imshow(fft, cmap='gray', vmin=0, vmax=255)
    plt.title(title2)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def main(path='./images'):
    if not os.path.exists(os.path.join(path, 'problem_2')):
        os.mkdir(os.path.join(path, 'problem_2'))

    # Read image
    ein = cv2.imread(os.path.join(path, 'Einstein.tif'), cv2.IMREAD_GRAYSCALE)
    ein = np.asarray(ein)
    pho = cv2.imread(os.path.join(path, 'phobos.tif'), cv2.IMREAD_GRAYSCALE)
    pho = np.asarray(pho)
    plot_graph(ein, pho, title='2-0_original_images', title1='Einstein.tif', title2='phobos.tif')

    # Filtering
    ein_param = (2.2, 0.2, 1, 80)
    pho_param1 = (3.2, 0.2, 1, 4800)
    pho_param2 = (2.2, 0.4, 1, 1200)

    ein_out = filtering(ein, ein_param)
    pho_out1 = filtering(pho, pho_param1)
    pho_out2 = filtering(pho, pho_param2)

    # Superposition of two images
    pho_out = 0.7 * pho_out1 + 0.3 * pho_out2

    plot_graph(ein, ein_out, title='2-1_filtered_Einstein')
    plot_graph(pho, pho_out, title='2-2_filtered_phobos')

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images/problem_2')