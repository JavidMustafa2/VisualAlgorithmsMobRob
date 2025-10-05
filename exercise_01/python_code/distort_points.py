import numpy as np


def distort_points(pixel: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    uo = K[0,2]; vo = K[1,2]; 
    xp = pixel[:, 0]-uo; yp = pixel[:, 1]-vo
    r_sq = (xp)**2 + (yp)**2
    pixel_du = ((1+D[0]*r_sq+D[1]*(r_sq)**2)*(xp)+uo)
    pixel_dv = (1+D[0]*r_sq+D[1]*(r_sq)**2)*(yp)+vo
    pixels_d = np.stack([pixel_du,pixel_dv],axis = -1)
    return pixels_d
