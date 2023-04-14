import os
import cv2
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    maginitude = 20 * np.log(np.abs(fshift))
    return fshift, maginitude

def plot_graph(img, fft, title='Figure', title1='Image', title2='Image', path='./images/problem_4'):
    plt.figure(title)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(title1)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(1, 2, 2)
    plt.imshow(fft, cmap='gray')
    plt.title(title2)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def wiener_filter(degrad: np.complex128, K=1):
    wiener = np.conj(degrad) / (np.abs(degrad) ** 2 + K)
    return wiener

def degradation(img_shape, T=1, a=0.1, b=0.1):
    rows, cols = img_shape
    u = np.arange(rows)
    v = np.arange(cols)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    param = (uu * a + vv * b).astype(np.complex128)
    degrad = T * np.sinc(param) * np.exp(-1j * np.pi * param)
    return degrad

def wiener_filter1(img_shape, kernel, K=10):
    kernel /= np.sum(kernel)
    kernel_fft = np.fft.fftshift(np.fft.fft2(kernel, s=img_shape))
    wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    return wiener
    
def gaussian_kernel(kernel_size=3):
	h = signal.gaussian(kernel_size, std=1).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h


def main(path='./images'):
    if not os.path.exists(os.path.join(path, 'problem_4')):
        os.mkdir(os.path.join(path, 'problem_4'))

    # Read image
    pho = cv2.imread(os.path.join(path, 'Photographer_degraded.tif'), cv2.IMREAD_GRAYSCALE)
    pho = np.asarray(pho)
    foo = cv2.imread(os.path.join(path, 'Football players_degraded.tif'), cv2.IMREAD_GRAYSCALE)
    foo = np.asarray(foo)

    # Find the Fourier spectrum of the corrupted images
    pho_fft, pho_mag = fft(pho)
    foo_fft, foo_mag = fft(foo)
    plot_graph(pho, pho_mag, '4-0_photographer', title2='Maginitude spectrum')
    plot_graph(foo, foo_mag, '4-1_football_player', title2='Maginitude spectrum')

    # Wiener filtering
    kernel_pho = gaussian_kernel(3)
    wiener_pho = wiener_filter1(pho.shape, kernel_pho, K=10)

    pho_out_fft = pho_fft * wiener_pho
    pho_out = np.abs(np.fft.ifft2(np.fft.ifftshift(pho_out_fft)))
    plot_graph(pho, pho_out, '4-2_restored_photographer', title2='Restored image')

    degrad_foo = degradation(foo.shape, T=1, a=0.001, b=-0.0059)
    wiener_foo = wiener_filter(degrad_foo, K=5e-2)

    foo_out_fft = foo_fft * wiener_foo
    foo_out = np.abs(np.fft.ifft2(np.fft.ifftshift(foo_out_fft)))
    plot_graph(foo, foo_out, '4-3_restored_football_player', title2='Restored image')

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images/problem_4')