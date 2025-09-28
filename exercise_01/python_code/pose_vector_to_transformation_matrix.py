import numpy as np
from vector_to_skew import vector_to_skew as skew

def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    rotations = pose_vec[:3]
    translations = np.transpose(np.take(pose_vec,[3,4,5]))
    theta = np.linalg.norm(rotations)
    unit_vector = np.transpose(rotations / theta)
    rotation_matrix = np.eye(3) + np.sin(theta)*skew(unit_vector) + (1-np.cos(theta))*(skew(unit_vector)**2)
    transformation_matrix = np.empty((4,4))
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translations
    transformation_matrix[3, :] = [0,0,0,1]
    
    return transformation_matrix

#testing code that runs if you run this file specifically
def main():

    filepath = r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\poses.txt'

    from load_camera_poses import load_camera_poses

    poses = load_camera_poses(filepath)

    for pose in poses:
        
        t_mat = pose_vector_to_transformation_matrix(pose)
        
        
    
    

if __name__ == "__main__":
    main()

