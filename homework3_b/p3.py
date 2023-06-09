import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(img1, img2, title='Figure', title1='Image1', title2='Image2', path='./images'):
    plt.figure(title, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    # plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight', dpi=1200, pad_inches = 0.1)
    plt.show()

def get_brick_size(image, upper=30):
    h, w = np.meshgrid(np.arange(1, upper + 1), np.arange(1, upper + 1), indexing='ij')
    coordinate = np.stack((h, w), axis=-1).reshape(-1, 2)

    for coord in coordinate.tolist():
        # print(f'Brick size: {c}')
        kernel = np.ones((coord[0], coord[1]),np.uint8)
        erosion = cv2.erode(image, kernel, iterations = 1)
        if np.sum(erosion) == 0:
            print(f'The brick size is about {coord[0] - 1} x {coord[1] - 1}')
            return coord[0] - 1, coord[1] - 1

def find_overlapping(image):
    kernel_vertical = np.ones((1, 31),np.uint8)
    erosion_vertical = cv2.erode(image, kernel_vertical, iterations = 1)
    ver_img = image - erosion_vertical
    # plot_graph(erosion_vertical, ver_img, title='ref1_vertical', title1='Vertical erosion', title2='Original - erosion')

    kernel_horizontal = np.ones((26, 1),np.uint8)
    erosion_horizontal = cv2.erode(image, kernel_horizontal, iterations = 1)
    hor_img = image - erosion_horizontal
    # plot_graph(erosion_horizontal, hor_img, title='ref2_horizontal', title1='Horizontal erosion', title2='Original - erosion')

    kernel = np.ones((18, 24),np.uint8)
    open_ver = cv2.morphologyEx(ver_img, cv2.MORPH_OPEN, kernel)
    open_hor = cv2.morphologyEx(hor_img, cv2.MORPH_OPEN, kernel)
    overlap = (image[:-1, :-1] - open_ver[1:, 1:]) + (image[:-1, :-1] - open_hor[1:, 1:])
    overlap = np.pad(overlap, [(0, 1), (0, 1)], mode='edge')
    # plot_graph(open_ver, image[:-1, :-1] - open_ver[1:, 1:], title='ref3_open_vertical', title1='Vertical opening', title2='Original - opening')
    # plot_graph(open_hor, image[:-1, :-1] - open_hor[1:, 1:], title='ref4_open_horizontal', title1='Horizontal opening', title2='Original - opening')
    return overlap.astype(np.uint8)

def main(path='./images'):
    # Read image
    img = cv2.imread(os.path.join(path, 'bricks.tif'), cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img)
    print(f'The shape of \'bricks.tif.\': {img.shape}')

    img_pad = np.pad(img, [(25, 25), (25, 25)], mode='constant', constant_values=0)
    brick_h, brick_w = get_brick_size(img_pad)

    # a. Find the overlapping regions
    overlap = find_overlapping(img_pad)
    overlap = overlap[25:-25, 25:-25]
    assert overlap.shape == img.shape
    plot_graph(img, overlap, title='3-0_overlap', title1='Original', title2='Overlapping bricks')

    # b. Find the isolated regions that are not adjacent to the boundary
    overlap_margin = find_overlapping(img)
    isolate = img - overlap_margin
    assert isolate.shape == img.shape
    plot_graph(img, isolate, title='3-1_isolate', title1='Original', title2='Isolated bricks')

    # b-1. Consider a broader boundary
    img_mar = np.zeros_like(img)
    img_mar[9:-11, 9:-11] = (img - overlap)[10:-10, 10:-10]
    isolate_broader = cv2.morphologyEx(img_mar, cv2.MORPH_OPEN, kernel=np.ones((22, 28),np.uint8))
    assert isolate_broader.shape == img.shape
    plot_graph(img, isolate_broader, title='3-2_broader_boundary', title1='Original', title2='Isolated bricks (broader boundary)')


if __name__ == '__main__':
    main(path='./images')
    print('Check the results in directory: ./images')