clc;
clear all
close all

% 均方误差和相对误差
epoch = linspace(0,101,101);
errors2scale = load('test_Err2Scale.mat');
mse2scale = errors2scale.mse;
rel2scale = errors2scale.rel;

errors2sub = load('test_Err2subspace.mat');
mse2sub = errors2sub.mse;
rel2sub = errors2sub.rel;


figure('name','Errors')
m2scale = plot(mse2scale,'r:', 'linewidth',2);
set(gca,'yscale','log')
grid on
xlim([0,102])
% title('Errors', 'Fontsize',18)
hold on
xlabel('epoch/1000')

r2scale = plot(rel2scale, 'm-.', 'linewidth',2);
hold on

m2fourier = plot(mse2sub, 'b--', 'linewidth',2);
hold on

r2fourier = plot(rel2sub, 'k.:', 'linewidth',2);
hold on

legend({'MSE2scale', 'REL2scale', 'MSE2sub', 'REL2sub'}, 'Fontsize',18)
% legend({'MSE-S', 'REL-S', 'MSE-HT', 'REL-HT','MSE-HS', 'REL-HS'}, 'Fontsize',18)

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
% mse2scale1 = mse2scale(71:91);
% rel2scale1 = rel2scale(71:91);
% axes('Position',[0.45,0.55,0.3,0.2]); % 生成子图                                                                           
% plot(x1,mse2scale1,'r:','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% xlim([min(x1),max(x1)]);             % 设置坐标轴范围
% hold on
% set(gca,'ytick',[])                  % 加上这个，子图的纵坐标不显示刻度
% 
% % plot(x1,rel2scale1,'b:','linewidth', 2);   % 绘制局部曲线图
% % hold on
% 
% 
% mse2ftanh1 = mse2ftanh(71:91);
% rel2ftanh1 = rel2ftanh(71:91);
%                                                                     
% plot(x1,mse2ftanh1,'b:.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
% 
% % plot(x1,rel2ftanh1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% % hold on
% 
% mse2fourier1 = mse2sub(71:91);
% rel2fourier1 = rel2sub(71:91);
% 
% plot(x1,mse2fourier1,'c-v','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% hold on
% 
% % plot(x1,rel2fourier1,'g-*','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% % hold on
% 
% % axes('Position',[0.45,0.55,0.3,0.2]); % 生成子图                                                                           
% % plot(x1,mse2scale1,'r:','linewidth', 2);   % 绘制局部曲线图                                                                                                                
% % xlim([min(x1),max(x1)]);             % 设置坐标轴范围
% % hold on
% % set(gca,'ytick',[])
