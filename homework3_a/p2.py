import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def indicator(x, u):
    r_nk = np.zeros((x.shape[0], u.shape[0]))

    for n in range(x.shape[0]):
        dists = []
        for k in range(u.shape[0]):
            # distance^2 = (x_x - u_x)^2 + (x_y - u_y)^2 + (x_z - u_z)^2
            dists.append(np.sum((x[n] - u[k]) ** 2))
        d_min = np.argmin(dists)
        r_nk[n][d_min] = 1.0
    return r_nk

def center_updating(x, r_nk):
    u_k_new = []

    for k in range(r_nk.shape[1]):
        r_n = r_nk[:, k:(k + 1)]
        r_tmp = np.hstack((r_n, r_n, r_n))

        sum_rx = np.sum(r_tmp * x, axis=0)
        sum_r = np.sum(r_n, axis=None)

        u_k_new.append((sum_rx / sum_r))

    return u_k_new

def print_table(u, k):
    u = (u * 255).astype(np.uint8)
    print(f'============= K = {k} =============')
    print('K-means |      R      G      B')
    for i in range(k):
        print(f'{i:5}   |  {u[i, 0]:6} {u[i, 1]:6} {u[i, 2]:6}')
    print('=================================')

def segmentation(u, r_nk, img_size):
    new_img = np.array(u)[np.where(r_nk == 1)[1]]
    new_img = (new_img * 255).reshape(img_size).astype(np.uint8)
    return new_img

def plot_graph(img1, img2, img3, img4, title='Figure', path='./images'):
    plt.figure(title, figsize=(10, 8))

    image_list = [img1, img2, img3, img4]
    title_list = ['original', 'k-means', 'masked', 'segmented']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image_list[i])
        plt.title(title_list[i])

    plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def main(path='./images'):
    # Read image
    img = cv2.imread(os.path.join(path, '20-2851.tif'))
    img = np.asarray(img)
    print(f'The shape of \'20-2851.tif\': {img.shape}.')

    # Preprocessing: blur the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.blur(img, (50, 50))


    # Initialization
    k = 3
    iter_max = 10
    data = np.reshape(blur, (-1, img.shape[2])) / 255
    # u = data[np.random.choice(len(data), k, replace=False)]
    u = np.array([[0.79677167, 0.67812622, 0.91373442],
                  [0.93855526, 0.89371823, 0.93614056],
                  [0.89896254, 0.83981157, 0.92769433]])

    # K-means
    print('Execute K-means clustering. Please wait for a while...')
    for _ in range(iter_max):
        r_nk = indicator(data, u)
        u_k_new = np.array(center_updating(data, r_nk))
        
        if (u_k_new == u).all:
            break
        else:
            u = u_k_new
    print('Done!')
    print_table(u_k_new, k)
    kmeans_img = segmentation(u_k_new, r_nk, img.shape)

    # Find the cluster close to the 'purple' cells
    u_k_mask = np.zeros_like(u_k_new)
    u_k_mask[np.argmin(np.sum(u_k_new, axis=1)), :] = np.ones(3)

    # Build binary mask to indicate the area in interest
    mask_img = segmentation(u_k_mask, r_nk, img.shape)

    # Partition the cell part
    new_img = np.copy(mask_img)
    new_img[mask_img > 0] = img[mask_img > 0]

    # Plot the images
    plot_graph(img, kmeans_img, mask_img, new_img, '2_color_segmentation.jpg')

if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images')