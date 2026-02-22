function [csi,deleted_tx,deleted_rx,deleted_sc] = copy_empty_csi(csi_origin)
[TX,RX,SC] = size(csi_origin{1}.csi);
PA = length(csi_origin);

csi = csi_origin;
deleted_tx = [];
deleted_rx = [];
deleted_sc = [];
for i1 = 1:PA
    [tx,rx,sc] = size(csi_origin{i1}.csi);
    if tx ~= TX
        csi{i1}.csi = cat(1,csi_origin{i1}.csi,csi_origin{i1}.csi);
    end
end
% disp("deleted empty csi")
% disp("empty tx")
% disp(deleted_tx)
% disp("empty rx")
% disp(deleted_rx)
% disp("empty packages")
% disp(deleted_sc)