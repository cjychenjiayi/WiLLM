function amp= get_amplitude(csi)
% empty pa*tx*rx*sc
[TX,RX,SC] = size(csi{1}.csi);
PA = length(csi);
amp = zeros(PA,TX,RX,SC);
for pa = 1:PA
    amp(pa,:,:,:) = abs(csi{pa}.csi);
end