clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

data2utrue = load('u_true6.mat');
utrue = data2utrue.u_true;

% %plot the solution in 3D with dot
% figure('name','utrue_dot')
% plot3(meshXY(1,:),meshXY(2,:),utrue, 'r.');
% hold on

figure('name', 'Utrue_surf')
surfAndPlot_U(meshXY, utrue)
hold on