import numpy as np


def load_camera_poses(filepath):
    '''this function expects a text file with x rows and y columns, it returns a numpy matrix of x columns of numpy.ndarray vectirs with y columns of numpy.float64 values in each vector '''
    with open(filepath, 'r') as file:
        
        poses = np.genfromtxt(file)
        
    return poses
def main():

    load_camera_poses(r'C:\Users\fmust\Downloads\VisAlgsMobRob\VisualAlgorithmsMobRob\exercise_01\data\poses.txt')

if __name__ == "__main__":
    main()