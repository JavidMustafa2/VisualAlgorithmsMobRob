import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized
from load_camera_poses import load_camera_poses
from matplotlib import pyplot as plt

def main():
    
    poses_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\poses.txt'
    image_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\images_undistorted\img_0001.jpg'
    K_matrix_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\K.txt'
    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    poses = load_camera_poses(poses_filepath) #(x,y) np matrix

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    x = np.linspace(0,0.32,9)
    y = np.linspace(0,0.20,6)
    z = np.zeros(6*9)
    vertices = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [0, 0, 1],  # 4
    [1, 0, 1],  # 5
    [1, 1, 1],  # 6
    [0, 1, 1],  # 7
])
    def cube(side_length=0.16, starting_x =0.10,starting_y=0.10,staring_z=0.0):
       
        cube_matrix = np.array([[starting_x,starting_y,staring_z,1],
                            [starting_x+side_length,starting_y,staring_z,1],
                            [starting_x+side_length,starting_y+side_length,staring_z,1],
                            [starting_x,starting_y+side_length,staring_z,1],
                            [starting_x,starting_y,staring_z-side_length,1],
                            [starting_x+side_length,starting_y,staring_z-side_length,1],
                            [starting_x+side_length,starting_y+side_length,staring_z-side_length,1],
                            [starting_x,starting_y+side_length,staring_z-side_length,1]])
        return cube_matrix
    def draw_cube(cube_pixel_coords,image):
        edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  
                    (4, 5), (5, 6), (6, 7), (7, 4),  
                    (0, 4), (1, 5), (2, 6), (3, 7)   
                ]
        for i, j in edges:
            pt1 = tuple(cube_pixel_coords[i])  
            pt2 = tuple(cube_pixel_coords[j])
            image = cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        return image
  

    
    
    x,y = np.meshgrid(x,y)
    
    #getting corner positions and also appending ones for transformation matrix
    xyzcoords = np.column_stack((x.ravel(),y.ravel(),z.ravel(),np.ones(6*9)))
    
  
    
    
    #loading K matrix
    with open(K_matrix_filepath, 'r') as file:
        
        K_matrix = np.genfromtxt(file)
        
    

    # load one image with a given index
    # reading the image in grayscale
    gray_image = cv2.imread(image_filepath)


    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    #transform matrix

    matrix_transform = pose_vector_to_transformation_matrix(poses[0])


    # transform 3d points from world to current camera pose
    world_xyz = xyzcoords

    camera_xyz = np.zeros((54,3))
    pixel_xyz = np.zeros((54,2))
    cube_camera_xyz = np.zeros((8,3))
    cube_pixel_xyz = np.zeros((8,2))
    copy = gray_image.copy()
    for i, coords in enumerate(world_xyz):
        world_to_camera_to_pixel = np.transpose(K_matrix@matrix_transform@np.transpose(coords))
     
        camera_xyz[i] = world_to_camera_to_pixel
        pixel_xyz[i] = np.array([camera_xyz[i][0]/camera_xyz[i][2],camera_xyz[i][1]/camera_xyz[i][2]])
    
    cubexyz = cube(0.08,0.08,0.08,0)
    
    for i,coords in enumerate(cubexyz):
        cube_world_to_camera_to_pixel = np.transpose(K_matrix@matrix_transform@np.transpose(coords))
     
        cube_camera_xyz[i] = cube_world_to_camera_to_pixel
        cube_pixel_xyz[i] = np.array([cube_camera_xyz[i][0]/cube_camera_xyz[i][2],cube_camera_xyz[i][1]/cube_camera_xyz[i][2]])
 
    cube_pixel_xyz = np.round(cube_pixel_xyz).astype(np.int32) 

    for i in range(54):
        center = tuple(np.round(pixel_xyz[i]).astype(int))
        print(center)
        copy = cv2.circle(copy, center, 5, (0, 0, 255), 2)


    copy = draw_cube(cube_pixel_xyz,copy)

# Display the image
    cv2.imshow('Image with Circle and cube', copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # undistort image with bilinear interpolation
    """ Remove this comment if you have completed the code until here
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    """

    # calculate the cube points to then draw the image
    # TODO: Your code here
    
    # Plot the cube
    """ Remove this comment if you have completed the code until here
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    """


if __name__ == "__main__":
    main()
