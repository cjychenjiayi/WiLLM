function amp = amp_DFT2(amp_original)
% n*pa*sc*trx
amp = amp_original;
for n = 1:size(amp,1)
for trx = 1:size(amp,4)
    slice_fft = fft2(amp_original(n,:, :, trx));
    
    % 将零频率移至中心
    slice_fft_centered = fftshift(slice_fft);
    
    % 存储处理后的切片
    amp(n,:, :, trx) = slice_fft_centered;
end
end
end
