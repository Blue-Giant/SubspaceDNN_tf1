clear all
close all
clc
data = load('pla.mat')
meshXY = data.pf;
u_true = data.uf;
save('meshXY6', 'meshXY');
save('u_true6','u_true');