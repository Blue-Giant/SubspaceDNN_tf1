clc
clear all
close all

data2Scale = load('test_solus2Scale.mat');
uScale = data2Scale.Us2relu;

Utrue2sub = load('Utrue2subspace.mat');
Ucoarse2sub = load('Unormal2subspace.mat');
Ufine2sub = load('Uscale2subspace.mat');
U2subspace = load('U2subspace.mat');

uTrue = Utrue2sub.Utrue;
Ucoarse = Ucoarse2sub.UNORMAL;
Ufine = Ufine2sub.USCALE;
Usubspace = U2subspace.UTANH;

data_x = load('testData2X');
x = data_x.Points2X;
trueNormal = x.*(1-x);
trueScale = uTrue - trueNormal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('name','solus')
% title('Solutions \epsilon=0.01', 'Fontsize',18)
plot(x, uTrue, 'k--', 'linewidth',1.5)
ylim([-0.01, 0.26])
grid on
hold on

plot(x, uScale, 'm-.', 'linewidth',1.5)
hold on

plot(x, Usubspace, 'b:', 'linewidth',1.5)
hold on


% legend({'ufourier', 'uscale', 'utrue'}, 'Fontsize',15)
lg = legend({'exact','scale','subspace'},'orientation','horizontal', 'Fontsize',18);
lg.Position = [0.35,0.125,0.4,0.1];
legend('boxoff')

x1 = x(401:601);
utrue1 = uTrue(400:600); 
uScale1 = uScale(400:600); 
usubspace1 = Usubspace(400:600);


axes('Position',[0.35,0.32,0.4,0.25]); % 生成子图                                                                         
plot(x1,utrue1,'k--','linewidth', 2);   % 绘制局部曲线图  
xlim([min(x1),max(x1)]);             % 设置坐标轴范围
hold on
set(gca,'ytick',[])                  % 加上这个，子图的纵坐标不显示刻度
                                                                      
plot(x1,uScale1,'m-.','linewidth', 2);   % 绘制局部曲线图                                                                                                                
hold on

plot(x1,usubspace1,'b:','linewidth', 2);   % 绘制局部曲线图                                                                                                                
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('name','coarse')
plot(x, Ucoarse, '-.', 'linewidth',1.5)
ylim([-0.01, 0.26])
grid on
hold on

plot(x, trueNormal, 'b--', 'linewidth',1.5)
hold on

lgcoarse = legend({'sub-coarse','exact-coarse'},'orientation','horizontal', 'Fontsize',18);
lgcoarse.Position = [0.325,0.125,0.4,0.1];
legend('boxoff')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('name','fine')
plot(x, Ufine, 'm:', 'linewidth',1.5)
ylim([-0.005, 0.03])
grid on
hold on

plot(x, trueScale, 'r-.', 'linewidth',1.5)
hold on

lgfine = legend({'sub-fine','exact-fine'},'orientation','horizontal', 'Fontsize',18);
lgfine.Position = [0.325,0.85,0.4,0.1];
legend('boxoff')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('name','exact-fine')
plot(x, trueScale, 'm:', 'linewidth',1.5)
grid on
hold on
