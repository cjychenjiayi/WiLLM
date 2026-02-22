function csi = interpolate_amp0(csi_original)
    csi = csi_original;
    [TX, RX, SC] = size(csi_original{1}.csi);
    for i = 1:length(csi_original)
        for tx = 1:TX
            for rx = 1:RX
                fiber = csi_original{i}.csi(tx,rx,:);
                pos0 = find(fiber == 0);
                if pos0
                    amp = sum(abs(fiber))/(length(fiber)-length(pos0));
                    agl = sum(angle(fiber))/(length(fiber)-length(pos0));
                    fiber(pos0) = amp*exp(1i * agl);
                    csi{i}.csi(tx,rx,:) = fiber;
                end
            end
        end
    end
end
