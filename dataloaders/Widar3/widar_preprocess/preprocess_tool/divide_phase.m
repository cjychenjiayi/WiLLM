function phase = divide_phase(csi)
sz = size(csi);
phase = zeros(sz(1),sz(2),sz(3),2,sz(5));
phase(:,:,:,1,:) = csi(:,:,:,2,:)./csi(:,:,:,1,:);
phase(:,:,:,2,:) = csi(:,:,:,3,:)./csi(:,:,:,1,:);
phase = reshape(phase,sz(1),sz(2),sz(3)*2,sz(5));
phase = permute(phase,[1,2,4,3]);
phase = phase(:,:,1:2:28,:);

phase = angle(phase);
phase = fft(phase,[],2);%时间维度
phase = abs(phase);
phase = log10(phase + 1);
end
