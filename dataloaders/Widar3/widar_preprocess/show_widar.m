clc;clear;close all;
% load('/home/chenjiayi/workspace/willm/wifi_data/widar3_raw/processed/all_csi_1.mat')
load("/home/chenjiayi/workspace/willm/wifi_data/widar_washed_denoised/all_csi_1.mat")
% size: pa*tx*rx*sc, n*3*1*30
amp = csi_result{5};

[T, F] = size(amp);
T = 1000;
t = 1:T;
figure;
hold on;
% 每隔10个特征画一条曲线
for f = 1:10:F
    plot(t, amp(1:T, f), 'DisplayName', sprintf('Feature %d', f));
end
xlabel('Time');
ylabel('Feature Value');
title('Feature Curves over Time');
legend show;
grid on;
saveas(gcf, 'feature_plot_real.png');
