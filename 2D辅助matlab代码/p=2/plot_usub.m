clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

q = 6;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size
geom = assemble_fmesh(geom);

figure('name','true')
data2utrue = load('u_true6.mat');
utrue = data2utrue.u_true;
mesh_true = plot_fun(geom,utrue);
% title('reference solution')
hold on

figure('name','Usubspace')
usubspace = load('U2subspace.mat');
usub = usubspace.UTEST;
mesh_usubspace = plot_fun(geom,usub);
% title('fourier solution')
hold on

figure('name','Ucoarse')
unormal = load('Unormal2subspace.mat');
ucoarse = unormal.UNORMAL;
mesh_ucoarse = plot_fun(geom,ucoarse);
% title('fourier solution')
hold on

figure('name','Ufine')
usubscale = load('Uscale2subspace.mat');
ufine = usubscale.USCALE;
mesh_ufine = plot_fun(geom,ufine);
% title('fourier solution')
hold on


solu_diff2sub = utrue - usub;
pointErr_sub = solu_diff2sub.^2;
figure('name','perror2subspace')
pointerr2fourier = plot_fun2err(geom,pointErr_sub);
% title('point-wise error2srelu')
hold on

