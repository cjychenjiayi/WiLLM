function amp_window = get_window(amp,n_timestamps,step_size)
% the same to phase tensor
[PA,TX,RX,SC] = size(amp);
N = floor((PA-n_timestamps+1)/step_size);
amp_window = zeros(N,n_timestamps,TX,RX,SC);
for n = 1:N
    amp_window(n,:,:,:,:) = amp((n-1)*step_size+1:(n-1)*step_size+n_timestamps,:,:,:);
end
end
