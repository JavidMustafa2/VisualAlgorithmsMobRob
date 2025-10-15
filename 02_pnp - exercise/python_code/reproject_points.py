import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points
    X_h = np.hstack([P, np.ones((P.shape[0],1))]) 
    
    x_h = (K @ M_tilde @ X_h.T).T 
    u = x_h[:,0] / x_h[:,2]
    v = x_h[:,1] / x_h[:,2]
    
    coords = np.array([u,v]).T
    return coords

    


    
   
  
