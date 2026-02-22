function amp_after_hp = amp_hampel(amp)
[PA,TX,RX,SC] = size(amp);
amp_after_hp = zeros(PA,TX,RX,SC);
for tx = 1:TX
    for rx = 1:RX
        for sc = 1:SC
            fiber = squeeze(amp(:,tx,rx,sc));
            fiber = hampel(fiber);
            amp_after_hp(:,tx,rx,sc) = fiber;
        end
    end
end
end