function csi = interpolate_timestamp(csi_origin,method)
[TX,RX,SC] = size(csi_origin{1}.csi);
PA = length(csi_origin);

original_timestamp = zeros(PA,1);
original_amp = zeros(TX,RX,SC,PA);
for pa = 1:PA
    original_timestamp(pa) = csi_origin{pa}.timestamp_low;
    original_amp(:,:,:,pa) = csi_origin{pa}.csi;
end
% 解缠，数据约一小时后会绕回
original_timestamp = unwrap(original_timestamp,2^32);

interval = (original_timestamp(end)-original_timestamp(1))/(PA-1);
new_timestamp = original_timestamp(1):interval:original_timestamp(end);

csi = csi_origin;
for tx = 1:TX
    for rx = 1:RX
        for sc = 1:SC
            switch method
                case 'linear'
                original_amp(tx,rx,sc,:) = interp1(original_timestamp,squeeze(original_amp(tx,rx,sc,:)), new_timestamp, 'linear', 'extrap');
                otherwise
                    error("invalid interpolation method!")
            end
        end
    end
end

for pa = 1:PA
    csi{pa}.timestamp_low = new_timestamp(pa);
    csi{pa}.csi = original_amp(:,:,:,pa);
end

end