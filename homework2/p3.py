import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(img1, img2, img3=None, title='Figure', num_img=2,
               title1='Image', title2='Image', title3='Image', path='./images/problem_3'):
    plt.figure(title)

    plt.subplot(1, num_img, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.subplot(1, num_img, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    if num_img == 3:
        plt.subplot(1, num_img, 3)
        plt.imshow(img3, cmap='gray')
        plt.title(title3)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def notch_reject(img_shape, coord=(0, 0), d0=5, n=4):
    rows, cols = img_shape
    u_k, v_k = coord

    u = np.arange(rows)
    v = np.arange(cols)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    D_uv = np.sqrt((uu - rows / 2 + u_k) ** 2 + (vv - cols / 2 + v_k) ** 2)
    D_muv = np.sqrt((uu - rows / 2 - u_k) ** 2 + (vv - cols / 2 - v_k) ** 2)

    # Butterworth
    notch = np.ones((rows, cols))
    notch *= 1.0 / (1.0 + (d0 / D_uv) ** (2*n))
    notch *= 1.0 / (1.0 + (d0 / D_muv) ** (2*n))

    return notch

def compute_weight(image, noise, neighbor_size=3):
    weight = np.zeros_like(image)
    pad = neighbor_size // 2
    image_pad = np.pad(image, pad_width=((pad, pad), (pad, pad)), mode='edge')
    noise_pad = np.pad(noise, pad_width=((pad, pad), (pad, pad)), mode='edge')

    for x in range(weight.shape[0]):
        for y in range(weight.shape[1]):
            image_n = image_pad[x: x+neighbor_size, y: y+neighbor_size]
            noise_n = noise_pad[x: x+neighbor_size, y: y+neighbor_size]
            
            weight[x, y] = np.mean(image_n * noise_n) - np.mean(image_n) * np.mean(noise_n)
            weight[x, y] /= (np.mean(noise_n ** 2) - (np.mean(noise) ** 2))
            
    return weight


def main(path='./images'):
    if not os.path.exists(os.path.join(path, 'problem_3')):
        os.mkdir(os.path.join(path, 'problem_3'))

    # Read image
    mar = cv2.imread(os.path.join(path, 'Martian terrain.tif'), cv2.IMREAD_GRAYSCALE)
    mar = np.asarray(mar)

    # Find Fourier spectrum of corrupted image
    mar_fft = np.fft.fft2(mar)
    mar_fft_mag = np.abs(np.fft.fftshift(mar_fft))
    plot_graph(mar, 20 * np.log(mar_fft_mag), title='3-0_fourier_transform',
               title1='Image', title2='Magnitude spectrum')


    ###### Part 1. Notch Reject Filtering ######
    # Find interference spike and obtain approximate noise pattern
    coords = [(114, 177), (160, 191), (196, 191),
              (95, 220), (117, 225), (137, 232), (160, 239)]
    mar_ctr = (133, 137)
    notch_r = np.ones(mar.shape)

    for coord in coords: 
        notch_r *= notch_reject(mar.shape, (coord[0] - mar_ctr[0], coord[1] - mar_ctr[1]))

    # Compare the original spectrum with that without noise
    mar_r = np.fft.fftshift(mar_fft) * notch_r
    plot_graph(20 * np.log(mar_fft_mag), 20 * np.log(np.abs(mar_r)), title='3-1_notch_reject',
               title1='Original spectrum', title2='Notch-rejected spectrum')

    # Check the result of notch_reject_filtering
    mar_filtered = np.fft.ifft2(np.fft.ifftshift(mar_r)).real


    ###### Part 2. Optimum Notch Filtering ######
    # Find noise
    notch_p = 1 - notch_r
    noise_fft = np.fft.fftshift(mar_fft) * notch_p
    noise = np.fft.ifft2(np.fft.ifftshift(noise_fft)).real
    plot_graph(noise, 20 * np.log(np.abs(noise_fft)), title='3-2_noise',
               title1='Noise', title2='Magnitude spectrum')

    # Find weight
    weight = compute_weight(mar, noise, neighbor_size=73)

    # Optimum notch filtering
    mar_restored = mar - weight * noise
    plot_graph(mar, mar_restored, mar_filtered, '3-3_optimum_notch_filtering',
               num_img=3, title1='Image', title2='Restored image', title3='NR filtered image')

    # Test different neighborhood size
    weights = {'size': [13, 43, 73], 'w': []}
    for s in weights['size']:
        w = compute_weight(mar, noise, neighbor_size=s)
        weights['w'].append(mar - w * noise)

    plot_graph(weights['w'][0], weights['w'][1], weights['w'][2], '3-4_neighborhood_sixe_comparison',
               num_img=3, title1='Size = 13', title2='Size = 43', title3='Size = 73')
    

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images/problem_3')