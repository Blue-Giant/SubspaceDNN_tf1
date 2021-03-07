clc
clear all
close all

data2Subspace = load('test_Err2subspace.mat');
data2Scale = load('test_Err2scale.mat');

mseSubspace = data2Subspace.mse;
relSubspace = data2Subspace.rel;

mseScale = data2Scale.mse;
relScale = data2Scale.rel;

figure('name','errors')
m2scale = plot(mseScale, 'm--', 'linewidth',2);
set(gca,'yscale','log')
grid on
xlim([0,62])
hold on

r2scale = plot(relScale, 'g-*', 'linewidth',2);
hold on

m2subspace = plot(mseSubspace, 'k-', 'linewidth',2);
hold on

r2subspace = plot(relSubspace, 'b-.', 'linewidth',2);
hold on

legend({'Mse2scale', 'Rel2scale', 'Mse2sub', 'Rel2sub'}, 'Fontsize',18)
% legend({'MSE-H', 'REL-H', 'MSE-S', 'REL-S'}, 'Fontsize',18)
% legend({'MSE-HS', 'REL-HS','MSE-HT', 'REL-HT', 'MSE-S', 'REL-S'}, 'Fontsize',18)

% lgd1=legend([m2scale,m2ftanh,m2fourier],'MSE-s2ReLU','MSE-FT','MSE-FS','orientation','horizontal','location','North');
% set(lgd1,'FontSize',16);
% lgd1.Position = [0.3  0.775  0.45  0.2];
% legend boxoff;
% ah=axes('position',get(gca,'position'),'visible','off');
% lgd2=legend(ah,[r2scale,r2ftanh,r2fourier],'REL-s2ReLU','REL-FT','REL-FS','orientation','horizontal','location','North');
% set(lgd2,'FontSize',16);
% lgd2.Position = [0.3  0.7  0.45  0.2];
% legend boxoff;