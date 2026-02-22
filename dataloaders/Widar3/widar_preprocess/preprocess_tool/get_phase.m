function phase = get_phase(csi)
% amp:amplitude*tx*rx*package
[TX,RX,SC] = size(csi{1}.csi);
PA = length(csi);
phase = zeros(PA,TX,RX,SC);
for pa = 1:PA
    phase(pa,:,:,:) = angle(csi{pa}.csi);
end