clc;
clear all
close all

dara2scale = load('pERR2Scale.mat');
pErr2scale = dara2scale.pERR2s2ReLU;
figure('name','pError2scale')
[X,Y]=meshgrid(1:1:40,1:1:40);
pError = reshape(pErr2scale, 40,40);
surf(X,Y,pError)
colorbar;
caxis([0 1e-4])
hold on


dara2perr2sub = load('pERR2Sub.mat');
poErr2sub = dara2perr2sub.pERR2tanh;
figure('name','pError2subspace')
[X,Y]=meshgrid(1:1:40,1:1:40);
pError2subspace = reshape(poErr2sub, 40,40);
surf(X,Y,pError2subspace)
colorbar
caxis([0 1e-4])
hold on