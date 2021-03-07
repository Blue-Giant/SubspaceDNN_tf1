clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

figure('name','true')
data2utrue = load('u_true6.mat');
utrue = data2utrue.u_true;
plot_solu(meshXY,utrue);
% title('reference solution')
hold on

figure('name','scale_test')
data2Scale = load('test_solus2Scale.mat');
uscale = data2Scale.Us2relu;
plot_solu(meshXY,uscale);
% title('scale solution')
hold on

figure('name','subspace_test')
Usubspace = load('U2subspace.mat');
usub=Usubspace.UTEST;
plot_solu(meshXY,usub);
% title('fourier solution')
hold on

figure('name','coarse_test')
Ufine = load('Unormal2subspace.mat');
ufine=Ufine.UNORMAL;
plot_solu(meshXY,ufine);
% title('fourier solution')
hold on

figure('name','fine_test')
Ufine = load('Uscale2subspace.mat');
ufine=Ufine.USCALE;
plot_solu(meshXY,ufine);
% title('fourier solution')
hold on


solu_diff2scale = utrue - uscale;
pointErr_scale = solu_diff2scale.^2;
figure('name','perror2scale')
plot_err(meshXY,pointErr_scale);
% title('point-wise error2srelu')
hold on

solu_diff2sub = utrue - usub;
pointErr_sub = solu_diff2sub.^2;
figure('name','perror2sub')
plot_err(meshXY,pointErr_sub);
% title('point-wise error2srelu')
hold on

% % 均方误差和相对误差
% epoch = linspace(0,101,101);
% errors2scale = load('test_Err2Scale.mat');
% mse2scale = errors2scale.mse;
% rel2scale = errors2scale.rel;
% 
% errors2fourier = load('test_Err2Fourier.mat');
% mse2fourier = errors2fourier.mse;
% rel2fourier = errors2fourier.rel;
% 
% errors2tanh = load('test_Err2tanh.mat');
% mse2ftanh = errors2tanh.mse;
% rel2ftanh = errors2tanh.rel;
% 
% figure('name','Errors')
% m2scale = plot(mse2scale,'r:', 'linewidth',2);
% set(gca,'yscale','log')
% grid on
% xlim([0,102])
% % title('Errors', 'Fontsize',18)
% hold on
% xlabel('epoch/1000')
% 
% r2scale = plot(rel2scale, 'm-.', 'linewidth',2);
% hold on
% 
% m2ftanh = plot(mse2ftanh, 'b:.', 'linewidth',2);
% hold on
% 
% r2ftanh = plot(rel2ftanh, '--', 'linewidth',2);
% hold on
% 
% m2fourier = plot(mse2fourier, 'c-v', 'linewidth',2);
% hold on
% 
% r2fourier = plot(rel2fourier, 'g-*', 'linewidth',2);
% hold on
% 
% % legend({'MSE-S', 'REL-S', 'MSE-H', 'REL-H'}, 'Fontsize',18)
% % legend({'MSE-S', 'REL-S', 'MSE-HT', 'REL-HT','MSE-HS', 'REL-HS'}, 'Fontsize',18)
% 
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
% x1 = epoch(71:91);
% m2scale1 = m2scale(71:91);
% r2scale1 = r2scale(71:91);
% axes('Position',[0.35,0.3,0.4,0.25]); % 生成子图                                                                           
% plot(x1,m2scale1,'b:','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% xlim([min(x1),max(x1)]);             % 设置坐标轴范围
% hold on
% 
% plot(x1,r2scale1,'b:','linewidth', 2);   % 绘制局部曲线图
% hold on
% 
% 
% m2ftanh1 = m2ftanh(71:91);
% r2ftanh1 = r2ftanh(71:91);
%                                                                     
% plot(x1,m2ftanh1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
% 
% plot(x1,r2ftanh1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
% 
% m2fourier1 = m2fourier(71:91);
% r2fourier1 = r2fourier(71:91);
% 
% plot(x1,m2fourier1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
% 
% plot(x1,r2fourier1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
