function amp_after_DWT = amp_DWT(amp)
scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
[PA,TX,RX,SC] = size(amp);
amp_after_DWT = zeros(PA,TX,RX,SC);
for tx = 1:TX
    for rx = 1:RX
        for sc = 1:SC
            fiber = squeeze(amp(:,tx,rx,sc));
            if exist('wden','file') == 2
                % use wavelet denoising if available
                fiber = wden(fiber,'sqtwolog','s',scal,5,'sym6');
            else
                % fallback: simple median filter when Wavelet toolbox unavailable
                win = 5;
                fiber = movmedian(fiber, win);
            end
            amp_after_DWT(:,tx,rx,sc) = fiber;
        end
    end
end
end