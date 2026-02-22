% TensorBeat: Tensor Decomposition for Monitoring Multiperson Breathing Beats with Commodity WiFi
function phase_diff = difference_phase(phase)
[PA,TX,RX,SC] = size(phase);
if RX == 3
phase_diff = zeros(PA,TX,2,SC);
phase_diff(:,:,1,:) = phase(:,:,1,:)-phase(:,:,3,:);
phase_diff(:,:,2,:) = phase(:,:,2,:)-phase(:,:,3,:);
else
    error("RX is not 3!");
end
end