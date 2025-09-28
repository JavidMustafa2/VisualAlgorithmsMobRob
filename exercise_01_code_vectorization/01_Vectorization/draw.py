import numpy as np
import time
import cv2


def draw_radial_for(img):
    """
    Draw a radial gradient image using for loop.

    Args:
        img: A numpy array with shape [height, width, 3].
    """

    max_dist_to_center = np.sqrt(img.shape[0]**2 + img.shape[1]**2) // 2

    # a for loop, nothing else
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist_to_center = np.sqrt((i - img.shape[0] // 2)**2 +
                                     (j - img.shape[1] // 2)**2)
            img[i, j] = 255 * (1 - dist_to_center / max_dist_to_center)
    return img


def draw_radial_vec(img):
    """
    Draw a radial gradient image using vectorization.

    Args:
        img: A numpy array with shape [height, width, 3].
    """
    # get max distance to center, half the diagonal length
    max_dist_to_center = np.sqrt(img.shape[0]**2 + img.shape[1]**2) // 2

    # create a meshgrid of pixel coordinates
    # X are the x coordinates of each pixel in the image
    # Y are the y coordinates of each pixel in the image

    X, Y = np.meshgrid(np.arange(img.shape[0]) - img.shape[0] // 2,
                       np.arange(img.shape[1]) - img.shape[1] // 2)

    # after flatten, shape of X or Y is (height * width, )
    dist_to_center = np.sqrt(X.flatten()**2 + Y.flatten()**2)

    # calculate pixel values
    img_array = (255 * (1 - dist_to_center /
                        max_dist_to_center)).astype(np.uint8)
    img[..., 0:] = img_array.reshape([img.shape[0],
                                      img.shape[1]])[..., np.newaxis]
    return img


def show_mesh_grid(max_num=5):
    """
    Show the meshgrid.

    Args:
        max_num: the maximum number of pixels to show
    """
    X, Y = np.meshgrid(np.arange(max_num), np.arange(max_num))
    print("X: \n{}".format(X))
    print("Y: \n{}".format(Y))


def draw():
    width = 800
    height = 800
    show_mesh_grid()

    img = np.zeros((height, width, 3), np.uint8)

    # for loop version
    start = time.time()
    img_for = draw_radial_for(img)
    for_loop_time = time.time() - start

    # vectorized version
    start = time.time()
    img_loop = draw_radial_vec(img)
    vec_time = time.time() - start

    print("Time for for loop: {:.8f}s".format(for_loop_time))
    print("Time for vectorization: {:.8f}s".format(vec_time))
    print("Speedup: {:.2f}x".format(for_loop_time / vec_time))
    print("Difference: {:.8f}".format(np.sum(np.abs(img_for - img_loop))))

    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    draw()
