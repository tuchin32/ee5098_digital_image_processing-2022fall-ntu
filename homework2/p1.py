import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fft(img, pure_im=False):
    f = np.fft.fft2(img)
    if pure_im == True:
        f.real = 0
    fshift = np.fft.fftshift(f)
    return f, np.abs(fshift)

def plot_graph(img, fft, title='Figure', title1='Image', title2='Magnitude Spectrum', path='./images/problem_1'):
    plt.figure(title)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.imshow(fft, cmap='gray')
    plt.title(title2)
    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def pad_image(image, rows, cols, odd_sym=False):
    image_pad = np.zeros((rows, cols))
    if odd_sym == True:
        mid_row = rows // 2
        mid_col = cols // 2
        image_pad[mid_row - 1:mid_row + 2, mid_col - 1:mid_col + 2] = image
    else:
        image_pad[:image.shape[0], :image.shape[1]] = image
    return image_pad


def main(path='./images'):
    if not os.path.exists(os.path.join(path, 'problem_1')):
        os.mkdir(os.path.join(path, 'problem_1'))

    # Read image
    img = cv2.imread(os.path.join(path, 'keyboard.tif'), cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img)

    ###--------------------------------###
    ### 1-a. Show the Fourier spectrum ###
    ###--------------------------------###
    _, img_fft_mag = fft(img)
    plot_graph(img, 20 * np.log(img_fft_mag), '1-a_fourier_transform')

    ###--------------------------------------------###
    ### 1-b. Enforce odd symmertry on Sobel kernel ###
    ###--------------------------------------------###
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Zero-padding
    rows = img.shape[0] + sobel.shape[0] - 1  # 1134 + 3 - 1 = 1136
    cols = img.shape[1] + sobel.shape[1] - 1  # 1360 + 3 - 1 = 1362

    img_pad = pad_image(img, rows, cols, odd_sym=False)
    sobel_pad = pad_image(sobel, rows, cols, odd_sym=True)

    # Generate transfer function H(u, v)
    sobel_fft, sobel_fft_mag = fft(sobel_pad, pure_im=True)
    sobel_fft_mag[:, sobel_fft_mag.shape[1] // 2:] *= -1
    plot_graph(sobel_pad[565:572, 678:685], sobel_fft_mag, '1-b_sobel_kernel', 'Kernel')

    ###---------------------------------###
    ### 1-c. Frequency-domain filtering ###
    ###---------------------------------###
    img_fft, _ = fft(img_pad)
    out_fft = img_fft * sobel_fft
    out = np.fft.ifftshift(np.fft.ifft2(out_fft)).real
    out_fft_mag = np.abs(np.fft.fftshift(out_fft))
    plot_graph(out, out_fft_mag, '1-c_frequency_domain_filtering')

    ###-------------------------------###
    ### 1-d. Spatial-domain filtering ###
    ###-------------------------------###
    # Derivative
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Get a complete contour by adding along two axis
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    
    plot_graph(grad_x, out, '1-d_spatial_domain_filtering', 'Spatial-Domain', 'Frequency-Domain')

    ###-------------------------------------------------------###
    ### 1-e. Frequency-domain filtering without odd symmertry ###
    ###-------------------------------------------------------###
    sobel_pad_wo = pad_image(sobel, rows, cols, odd_sym=False)
    sobel_fft_wo, _ = fft(sobel_pad_wo, pure_im=True)
    out_fft_wo = img_fft * sobel_fft_wo
    out_wo = np.fft.ifftshift(np.fft.ifft2(out_fft_wo)).real
    out_fft_mag_wo = np.abs(np.fft.fftshift(out_fft_wo))
    plot_graph(out_wo, out_fft_mag_wo, '1-e_frequency_domain_filtering_without_symmertry')

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images/problem_1')