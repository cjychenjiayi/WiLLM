function amp = amp_normalize_sample_DFT2(tmp)
% n*pa*tx*rx*sc
sz = size(tmp);
amp = amp_normalize(tmp);
amp = reshape(amp,sz(1),sz(2),sz(3)*sz(4),sz(5));
amp = permute(amp,[1,2,4,3]);% n*pa*sc*trx
amp = amp_DFT2(amp);
amp = abs(amp);
end
