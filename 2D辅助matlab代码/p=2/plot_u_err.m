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

figure('name','scale_test')
data2Scale = load('test_solus2Scale.mat');
uscale = data2Scale.Us2relu;
mesh_uscale = plot_fun(geom,uscale);
% title('scale solution')
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



solu_diff2scale = utrue - uscale;
pointErr_scale = solu_diff2scale.^2;
figure('name','perror2scale')
pointerr2scale = plot_fun2err(geom,pointErr_scale);
% title('point-wise error2srelu')
hold on

solu_diff2sub = utrue - usub;
pointErr_sub = solu_diff2sub.^2;
figure('name','perror2subspace')
pointerr2fourier = plot_fun2err(geom,pointErr_sub);
% title('point-wise error2srelu')
hold on

epoch = linspace(0,101,101);

errors2scale = load('test_Err2Scale.mat');
mse2scale = errors2scale.mse;
rel2scale = errors2scale.rel;

errors2subspace = load('test_Err2subspace.mat');
mse2sub = errors2subspace.mse;
rel2sub = errors2subspace.rel;


figure('name','Errors')
m2scale = plot(epoch,mse2scale,'r:', 'linewidth',2);
set(gca,'yscale','log')
grid on
xlim([0,102])
% title('Errors', 'Fontsize',18)
hold on
xlabel('epoch/1000')

r2scale=plot(epoch,rel2scale, 'm-.', 'linewidth',2);
hold on

m2fourier=plot(epoch,mse2sub, 'g-v', 'linewidth',2);
hold on

r2fourier=plot(epoch, rel2sub, 'c-*', 'linewidth',2);
hold on

legend({'MSE-Scale', 'REL-Scale', 'MSE-Sub', 'REL-Sub'}, 'Fontsize',18)

% % legend({'MSE-S', 'REL-S', 'MSE-HT', 'REL-HT','MSE-HS', 'REL-HS'}, 'Fontsize',18)

% lgd1=legend([m2scale,m2ftanh,m2fourier],'MSE-s2ReLU','MSE-FT','MSE-FS','orientation','horizontal','location','North');
% set(lgd1,'FontSize',16);
% lgd1.Position = [0.3  0.775  0.45  0.2];
% legend boxoff;
% ah=axes('position',get(gca,'position'),'visible','off');
% lgd2=legend(ah,[r2scale,r2ftanh,r2fourier],'REL-s2ReLU','REL-FT','REL-FS','orientation','horizontal','location','North');
% set(lgd2,'FontSize',16);
% lgd2.Position = [0.3  0.7  0.45  0.2];
% legend boxoff;
% 
