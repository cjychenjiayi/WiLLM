function csi = scale_csi_5300(csi_origin,method)
% method:5300(Intel NIC),other

csi = csi_origin;
if method == "5300"
    for pa = 1:length(csi_origin)
        try
            csi{pa}.csi = get_scaled_csi_sm(csi_origin{pa});
        catch ME
            fprintf('Error at pa = %d\n', pa);
            rethrow(ME);  % 保持原始报错信息
        end
    end
else
    for pa = 1:length(csi_origin)
        try
            csi{pa}.csi = get_scaled_csi(csi_origin{pa});
        catch ME
            fprintf('Error at pa = %d\n', pa);
            rethrow(ME);
        end
    end
end
end
