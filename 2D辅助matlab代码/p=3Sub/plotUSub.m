clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

data2sub = load('USub.mat');
usub = double(data2sub.UTEST);

% %plot the solution in 3D with dot
% figure('name','utrue_dot')
% plot3(meshXY(1,:),meshXY(2,:),usub, 'r.');
% hold on

figure('name', 'Usub_surf')
surfAndPlot_U(meshXY, usub)
hold on

data2coarse = load('Unormal2Sub.mat');
ucoarse = double(data2coarse.UNORMAL);
figure('name','Usub_coarse')
surfAndPlot_U(meshXY, ucoarse)
hold on

data2fine = load('Uscale2Sub.mat');
ufine = double(data2fine.USCALE);
figure('name','Usub_fine')
surfAndPlot_U(meshXY, ufine)
hold on