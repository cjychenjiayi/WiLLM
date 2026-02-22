% 华硕路由器优化，3*3变成4*3是第一根copy到第三根了，2*3则是三个都一样
function [csi,deleted_tx,deleted_rx,deleted_sc] = copy_empty_csi_v2(csi_origin)
[TX,RX,SC] = size(csi_origin{1}.csi);
PA = length(csi_origin);

csi = csi_origin;
deleted_tx = [];
deleted_rx = [];
deleted_sc = [];
for i1 = 1:PA
    pa = i1;
    [tx,rx,sc] = size(csi_origin{i1}.csi);
    if tx ~= TX
		if tx == 4
			csi{pa}.csi = csi{pa}.csi([1,2,4],:,:);
			csi{pa}.Ntx = 3;
		elseif tx == 2
		        new_csi = zeros(3, 3, 30) + 1i*zeros(3, 3, 30);
                new_csi(1, :, :) = csi{pa}.csi(1, :, :);
                new_csi(2, :, :) = csi{pa}.csi(2, :, :);
                new_csi(3, :, :) = (csi{pa}.csi(1, :, :) + csi{pa}.csi(2, :, :)) / 2;
                csi{pa}.csi = new_csi;
				csi{pa}.Ntx = 3;
        else
            new_csi = zeros(3, 3, 30) + 1i*zeros(3, 3, 30);
            new_csi(1, :, :) = csi{pa}.csi(1, :, :);
            new_csi(2, :, :) = new_csi(1, :, :);
            new_csi(3, :, :) = new_csi(1, :, :);
            csi{pa}.csi = new_csi;
		    csi{pa}.Ntx = 3;
        end
    end
end
% disp("deleted empty csi")
% disp("empty tx")
% disp(deleted_tx)
% disp("empty rx")
% disp(deleted_rx)
% disp("empty packages")
% disp(deleted_sc)