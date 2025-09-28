close all;
clear all;

% Load camera poses
% Each row i of matrix 'poses' contains the transformations that transforms
% points expressed in the world frame to points expressed in the camera frame.

% TODO: Your code here

% Define 3D corner positions

% [Nx3] matrix containing the corners of the checkerboard as 3D points
% (X,Y,Z), expressed in the world coordinate system

% TODO: Your code here

% Load camera intrinsics

% TODO: Your code here

% Load one image with a given index
% TODO: Your code here

% Project the corners on the image

% Compute the 4x4 homogeneous transformation matrix that maps points from the world
% to the camera coordinate frame
% TODO: Your code here

% Transform 3d points from world to current camera pose
% TODO: Your code here


% Undistort image with bilinear interpolation
%{ 
Remove this comment if you have completed the code until here
tic;
img_undistorted = undistortImage(img,K,D,1);
disp(['Undistortion with bilinear interpolation completed in ' num2str(toc)]);

% Vectorized undistortion without bilinear interpolation
tic;
img_undistorted_vectorized = undistortImageVectorized(img,K,D);
disp(['Vectorized undistortion completed in ' num2str(toc)]);

figure();
subplot(1, 2, 1);
imshow(img_undistorted);
title('With bilinear interpolation');
subplot(1, 2, 2);
imshow(img_undistorted_vectorized);
title('Without bilinear interpolation');
%}

% Draw a cube on the undistorted image
% TODO: Your code here

%{ 
Remove this comment if you have completed the code until here
figure();
imshow(img_undistorted); hold on;

lw = 3;

% base layer of the cube
line([cube_pts(1,1), cube_pts(1,2)],[cube_pts(2,1), cube_pts(2,2)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,1), cube_pts(1,3)],[cube_pts(2,1), cube_pts(2,3)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2), cube_pts(1,4)],[cube_pts(2,2), cube_pts(2,4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3), cube_pts(1,4)],[cube_pts(2,3), cube_pts(2,4)], 'color', 'red', 'linewidth', lw);

% top layer
line([cube_pts(1,1+4), cube_pts(1,2+4)],[cube_pts(2,1+4), cube_pts(2,2+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,1+4), cube_pts(1,3+4)],[cube_pts(2,1+4), cube_pts(2,3+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2+4), cube_pts(1,4+4)],[cube_pts(2,2+4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3+4), cube_pts(1,4+4)],[cube_pts(2,3+4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);

% vertical lines
line([cube_pts(1,1), cube_pts(1,1+4)],[cube_pts(2,1), cube_pts(2,1+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2), cube_pts(1,2+4)],[cube_pts(2,2), cube_pts(2,2+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3), cube_pts(1,3+4)],[cube_pts(2,3), cube_pts(2,3+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,4), cube_pts(1,4+4)],[cube_pts(2,4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);

hold off;
set(gca,'position',[0 0 1 1],'units','normalized')
%}

