close all;
clear all;

K = load('../data/K.txt');

p_W_corners = 0.01 * load('../data/p_W_corners.txt');

num_corners = length(p_W_corners);

% Load the 2D projected points (detected on the undistorted image)
all_pts2d = load('../data/detected_corners.txt');

num_images = 210;
% Loop through all the images
% TODO: Your code here

%% Generate video of the camera motion

%{ 
Remove this comment if you have completed the code until here
fps = 30;
plotTrajectory3D(fps, translations', quaternions', p_W_corners');
%} 
