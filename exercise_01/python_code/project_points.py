import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   matrix_transform,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """

    uo = K[0,2]; vo = K[1,2]; 
    Transform = K@(matrix_transform)
    pixel_h = (Transform @ points_3d.T).T
    pixel = pixel_h[:, :2] / pixel_h[:, 2:3]
    u = pixel[:, :1]; v = pixel[:, 1:2]
    r_sq = (u - uo)**2 + (v-vo)**2
    pixel_du = ((1+D[0]*r_sq+D[1]*(r_sq)**2)*(u-uo)+uo)
    pixel_dv = (1+D[0]*r_sq+D[1]*(r_sq)**2)*(v-vo)+vo
    pixels_d = np.hstack([pixel_du,pixel_dv])
    return pixels_d
    for i, coords in enumerate(points_3d):
                world_to_camera_to_pixel = np.transpose(K@matrix_transform@np.transpose(coords))
            
                camera_xyz[i] = world_to_camera_to_pixel
                pixel_xyz[i] = np.array([camera_xyz[i][0]/camera_xyz[i][2],camera_xyz[i][1]/camera_xyz[i][2]])
                u = pixel_xyz[i][0]; v = pixel_xyz[i][1]
                r_sq = (u-uo)**2 +(v-vo)**2
                pixel_xyz[i] = np.array([(1+D[0]*r_sq+D[1]*(r_sq)**2)*(u-uo)+uo, (1+D[0]*r_sq+D[1]*(r_sq)**2)*(v-vo)+vo])    
    return pixel_xyz


    