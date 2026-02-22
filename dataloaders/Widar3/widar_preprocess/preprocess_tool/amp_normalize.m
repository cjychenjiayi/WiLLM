function amp = amp_normalize(amp_original)
% n*pa*tx*rx*sc
X0 = amp_original(1,:,:,:,:);
amp = amp_original;
for i = 1:size(amp,1)
amp(i,:,:,:,:) = amp_original(i,:,:,:,:)./X0;
end
end
