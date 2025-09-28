import time
import cv2
import numpy as np


def show_image(img: np.ndarray):
    """
    Show the image.

    Args:
        img: the image to be shown
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask_out_image_for_loop(img: np.ndarray,
                            row_indices: list,
                            col_indices: list) -> np.ndarray:
    """
    Mask out the image using for loop.

    Args:
        img: the image to be masked out
    """
    for row in range(row_indices[0], row_indices[1]):
        for col in range(col_indices[0], col_indices[1]):
            for channel in range(3):
                img[row, col, channel] = 0

    return img


def mask_out_image_numpy(img: np.ndarray,
                         row_indices: list,
                         col_indices: list) -> np.ndarray:
    """
    Mask out the image using numpy.

    Args:
        img: the image to be masked out
    """
    img[row_indices[0]:row_indices[1],
        col_indices[0]:col_indices[1],
        :] = 0
    return img


def main():
    img = cv2.imread("./data/images/img_0001.jpg")
    show_image(img)
    row_indices = [100, 400]
    col_indices = [100, 400]

    start_time = time.time()
    img_maksed_for = mask_out_image_for_loop(img, row_indices, col_indices)
    for_loop_time = time.time() - start_time

    start_time = time.time()
    img_maksed_vec = mask_out_image_numpy(img, row_indices, col_indices)

    vec_time = time.time() - start_time

    show_image(img_maksed_vec)
    print("Time for for loop: {:.8f}s".format(for_loop_time))
    print("Time for vectorization: {:.8f}s".format(vec_time))
    print("Speedup: {:.2f}x".format(for_loop_time / vec_time))
    print("Difference: {:.8f}".format(
        np.sum(np.abs(img_maksed_vec - img_maksed_for))))


if __name__ == "__main__":
    main()
