import math
import numpy as np

from distort_points import distort_points
from project_points import project_points

def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    h,w = img.shape[-2:]
    X,Y = np.meshgrid(np.arange(w),np.arange(h))
    px = np.stack([X,Y],axis=-1).reshape([h*w, 2])
    distorted = distort_points(px,D,K)
    x = np.round(distorted[:, 0].astype(int))
    y = np.round(distorted[:, 1].astype(int))
    x = np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)
    intensity = img[y,x]

    und_img = intensity.reshape(img.shape).asype(np.uint8) 
    return und_img
    
