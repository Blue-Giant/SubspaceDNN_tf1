clc;
clear all
close all

% figure('name','true')
% data2utrue = load('test_solus2Fourier.mat');
% utrue = data2utrue.Utrue;
% imshow(utrue)
% title('true solution')
% hold on
% 
% figure('name','fourier_test')
% data2Fourier = load('test_solus2Fourier.mat');
% ufourier=data2Fourier.Us2relu;
% imshow(ufourier);
% title('fourier solution')
% hold on
% 
% solu_diff2fourier = utrue - ufourier;
% pointErr_fourier = solu_diff2fourier.^2;
% figure('name','perror2fourier')
% imshow(pointErr_fourier);
% title('point-wise error')
% hold on
% 
% 
% figure('name','scale_test')
% data2Scale = load('test_solus2Scale.mat');
% uscale = data2Scale.Us2relu;
% imshow(uscale);
% title('scale solution')
% hold on
% 
% 
% solu_diff2scale = utrue - uscale;
% pointErr_scale = solu_diff2scale.^2;
% figure('name','perror2scale')
% imshow(pointErr_scale);
% title('point-wise error')
% hold on
% 


errors2scale = load('test_Err2Scale.mat');
mse2scale = errors2scale.mse;
rel2scale = errors2scale.rel;

errors2fourier = load('test_Err2sub.mat');
mse2subspace = errors2fourier.mse;
rel2subspace = errors2fourier.rel;

figure('name','Errors')
plot(mse2scale,'r:', 'linewidth',2)
set(gca,'yscale','log')
grid on
xlim([0,102])
% title('Errors', 'Fontsize',18)
hold on

plot(rel2scale, 'm-.', 'linewidth',2)
% semilogy(relFourier)
hold on

plot(mse2subspace, 'g-v', 'linewidth',2)
% semilogy(mseScale)
hold on

plot(rel2subspace, 'c-*', 'linewidth',2)
% semilogy(relScale)
hold on

legend({'MSE-Scale', 'REL-Scale', 'MSE-Sub', 'REL-Sub'}, 'Fontsize',18)
