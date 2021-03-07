clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

data2scale = load('test_solus2Scale.mat');
uscale = double(data2scale.Us2relu);

% %plot the solution in 3D with dot
% figure('name','uscale_dot')
% plot3(meshXY(1,:),meshXY(2,:),uscale, 'r.');
% hold on

figure('name', 'Uscale_surf')
surfAndPlot_U(meshXY, uscale)
hold on