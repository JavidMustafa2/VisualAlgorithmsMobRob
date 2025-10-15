import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    
    # Convert 2D to normalized coordinates
    n = np.shape(p)[0]
    w_h = np.hstack([P,np.ones((np.shape(P)[0],1))])
    p_h = np.hstack([p,np.ones((np.shape(p)[0],1))]) ## currently n by 3
    normalized = (np.linalg.inv(K) @ p_h.T).T #back to n by 3 but now normalized
    
    # Build measurement matrix Q
    # Q must be a 2n by 12 matrix
    x_p_h = normalized[ : ,0]
    y_p_h = normalized[ :, 1]
  
    zeros = np.zeros((w_h.shape[0], 4))

    first_row = np.hstack([w_h, zeros, -x_p_h[:, None] * w_h])
  
    second_row = np.hstack([zeros,w_h,-y_p_h[:, None] * w_h])
    
    Q = np.empty((2*n, 12))
    Q[0::2, :] = first_row   
    Q[1::2, :] = second_row


    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    #doing decomposition:
    U, S, V = np.linalg.svd(Q)
   
    M = (V.T[: ,-1].reshape(3,4))

    R = M[: ,0:3]
    T = M[: ,3]

    # Extract [R | t] with the correct scale
    # 

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    Ur, Sr, Vr = np.linalg.svd(R)
    
    R_Orth = (Ur @ Vr)

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_Orth) / np.linalg.norm(R)
    
    
    # Build M_tilde with the corrected rotation and scale
    # TODO: Your code here

    M_tilde = np.hstack([R_Orth,alpha*T[: ,None]])
  
    return M_tilde
