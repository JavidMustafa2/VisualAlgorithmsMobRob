import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import os
from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized
from load_camera_poses import load_camera_poses
from matplotlib import pyplot as plt

def main():


# Folder containing images
    image_folder = r"C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\images"

    # Get list of all jpg images in the folder, sorted
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    print(f"Found {len(image_paths)} images.")

    poses_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\poses.txt'
  
    K_matrix_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\K.txt'
    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame
    D_filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\D.txt'
    poses = load_camera_poses(poses_filepath) #(x,y) np matrix

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    x = np.linspace(0,0.32,9)
    y = np.linspace(0,0.20,6)
    z = np.zeros(6*9)
    vertices = np.array([
    [0, 0, 0],  
    [1, 0, 0],  
    [1, 1, 0], 
    [0, 1, 0],   
    [0, 0, 1],  
    [1, 0, 1],  
    [1, 1, 1],   
    [0, 1, 1],  
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
    
    #loading distortions

    with open(D_filepath,'r' ) as file:
        D = np.genfromtxt(file)
    

    # load one image with a given index
    # reading the image in grayscale
    for i, image_filepath in enumerate(image_paths):
        gray_image = cv2.imread(image_filepath)


        # project the corners on the image
        # compute the 4x4 homogeneous transformation matrix that maps points
        # from the world to the camera coordinate frame

        #transform matrix

        matrix_transform = pose_vector_to_transformation_matrix(poses[i])


        # transform 3d points from world to current camera pose

        copy = gray_image.copy()
        uo = K_matrix[0,2]; vo = K_matrix[1,2]; 
        pixel_xyz = project_points(xyzcoords,matrix_transform,K_matrix,D)
        cubexyz = cube(0.08,0.08,0.08,0)
        for i in range(54):
            center = tuple(np.round(pixel_xyz[i]).astype(int))
        
            copy = cv2.circle(copy, center, 5, (0, 0, 255), 2)
        cube_pixel_xyz = project_points(cubexyz,matrix_transform,K_matrix,D)

        cube_pixel_xyz = np.round(cube_pixel_xyz).astype(np.int32) 

   


        copy = draw_cube(cube_pixel_xyz,copy)

        distorted_image = gray_image
      
        undistorted_image = undistort_image(distorted_image,K_matrix,D)
        print(np.shape(undistorted_image))
        cv2.imshow('undistortedvideo',undistorted_image)
        key = cv2.waitKey(3) 
        if key == 27: 
             break
        # cv2.imshow('distorteddvideo',distorted_image)
        # key = cv2.waitKey(3) 
        # if key == 27: 
        #      break
        # cv2.imshow("Cube Overlay", copy)
        # key = cv2.waitKey(3) 
        # if key == 27: 
        #     break

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



if __name__ == "__main__":
    main()
