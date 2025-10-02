function M_tilde = estimatePoseDLT(p, P, K)
% Estimates the pose of a camera using a set of 2D-3D correspondences and a
% given camera matrix
%
% p: [nx2] vector containing the undistorted coordinates of the 2D points
% P: [nx3] vector containing the 3D point positions
% K: [3x3] camera matrix
%
% M_tilde: [3x4] projection matrix under the form M_tilde=[R_tilde|alpha*t] where R is a rotation
%    matrix. M_tilde encodes the transformation that maps points from the world
%    frame to the camera frame

% Convert 2D points to normalized coordinates

% TODO: Your code here

% Build the measurement matrix Q
% TODO: Your code here

% Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
% TODO: Your code here

%% Extract [R|t] with the correct scale from M_tilde ~ [R|t]

% TODO: Your code here

% Find the closest orthogonal matrix to R
% https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
% TODO: Your code here

% Normalization scheme using the Frobenius norm:
% recover the unknown scale using the fact that R_tilde is a true rotation matrix
% TODO: Your code here

% Build M_tilde with the corrected rotation and scale
% TODO: Your code here

end

