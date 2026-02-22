function out = read_bfee(inBytes)
% Pure-MATLAB implementation of read_bfee (fallback when MEX not available)
% inBytes: uint8 vector of the beamforming payload
bytes = uint8(inBytes(:));

timestamp_low = double(bytes(1)) + bitshift(double(bytes(2)),8) + bitshift(double(bytes(3)),16) + bitshift(double(bytes(4)),24);
bfee_count = double(bytes(5)) + bitshift(double(bytes(6)),8);
Nrx = double(bytes(9));
Ntx = double(bytes(10));
rssi_a = double(bytes(11));
rssi_b = double(bytes(12));
rssi_c = double(bytes(13));
noise = double(bytes(14));
if noise > 127, noise = noise - 256; end
agc = double(bytes(15));
antenna_sel = double(bytes(16));
len_field = double(bytes(17)) + bitshift(double(bytes(18)),8);
fake_rate_n_flags = double(bytes(19)) + bitshift(double(bytes(20)),8);

calc_len = floor((30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8);
if len_field ~= calc_len
    error('Wrong beamforming matrix size.');
end

payload = bytes(21:end);
csi = complex(zeros(Ntx, Nrx, 30));
idx = 0;
for i = 1:30
    idx = idx + 3;
    remainder = mod(idx,8);
    for j = 0:(Nrx*Ntx-1)
        byte_idx = floor(idx/8) + 1;
        b1 = double(payload(byte_idx));
        b2 = double(payload(byte_idx+1));
        real_raw = bitand( bitor( bitshift(b1, -remainder), bitshift(b2, 8-remainder) ), 255);

        byte_idx2 = floor((idx+8)/8) + 1;
        b3 = double(payload(byte_idx2));
        b4 = double(payload(byte_idx2+1));
        imag_raw = bitand( bitor( bitshift(b3, -remainder), bitshift(b4, 8-remainder) ), 255);

        if real_raw > 127, real_raw = real_raw - 256; end
        if imag_raw > 127, imag_raw = imag_raw - 256; end

        tx = floor(j / Nrx) + 1;
        rx = mod(j, Nrx) + 1;
        csi(tx, rx, i) = complex(real_raw, imag_raw);

        idx = idx + 16;
    end
end

perm = [ bitand(antenna_sel, 3) + 1, bitand(bitshift(antenna_sel, -2), 3) + 1, bitand(bitshift(antenna_sel, -4), 3) + 1 ];

out.timestamp_low = timestamp_low;
out.bfee_count = bfee_count;
out.Nrx = Nrx;
out.Ntx = Ntx;
out.rssi_a = rssi_a;
out.rssi_b = rssi_b;
out.rssi_c = rssi_c;
out.noise = noise;
out.agc = agc;
out.perm = perm;
out.rate = fake_rate_n_flags;
out.csi = csi;
end
