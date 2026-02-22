function phase = normalize_data(phase_original)
% N*TX*RX*SC
min_val = min(phase_original(:));
max_val = max(phase_original(:));
phase = (phase_original - min_val)./(max_val-min_val);

end
