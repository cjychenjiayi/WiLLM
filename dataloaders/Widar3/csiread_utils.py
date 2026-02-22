import numpy as np
import struct


def read_csi_file(filename):
    """
    Read Intel 5300 CSI .dat file
    Returns a list of CSI dictionaries
    """

    ret = []
    broken_perm = False
    triangle = [1, 3, 6]

    with open(filename, "rb") as f:

        f.seek(0, 2)
        length = f.tell()
        f.seek(0, 0)

        cur = 0

        while cur < (length - 3):

            header = f.read(3)
            if len(header) < 3:
                break

            field_len = struct.unpack(">H", header[:2])[0]
            code = header[2]
            cur += 3

            if code != 187:
                f.seek(field_len - 1, 1)
                cur += field_len - 1
                continue

            bytes_data = f.read(field_len - 1)
            cur += field_len - 1

            if len(bytes_data) != field_len - 1:
                break

            entry = read_bfee(bytes_data)
            ret.append(entry)

            perm = entry["perm"]
            Nrx = entry["Nrx"]

            if Nrx == 1:
                continue

            if sum(perm) != triangle[Nrx - 1]:
                if not broken_perm:
                    broken_perm = True
                    print(
                        f"WARN ONCE: Found CSI ({filename}) "
                        f"with Nrx={Nrx} invalid perm={perm}"
                    )
            else:
                csi = entry["csi"]
                reordered = csi[:, perm[:Nrx] - 1, :]
                csi[:, :Nrx, :] = reordered
                entry["csi"] = csi

    return ret


def read_bfee(inBytes):
    """
    Parse one CSI entry (equivalent to read_bfee.c)
    """

    inBytes = np.frombuffer(inBytes, dtype=np.uint8)

    timestamp_low = (
        inBytes[0]
        + (inBytes[1] << 8)
        + (inBytes[2] << 16)
        + (inBytes[3] << 24)
    )

    bfee_count = inBytes[4] + (inBytes[5] << 8)

    Nrx = inBytes[8]
    Ntx = inBytes[9]

    rssi_a = inBytes[10]
    rssi_b = inBytes[11]
    rssi_c = inBytes[12]

    noise = np.int8(inBytes[13]).item()
    agc = inBytes[14]
    antenna_sel = inBytes[15]

    length = inBytes[16] + (inBytes[17] << 8)
    fake_rate_n_flags = inBytes[18] + (inBytes[19] << 8)

    calc_len = (30 * (int(Nrx) * int(Ntx) * 8 * 2 + 3) + 7) // 8

    if length != calc_len:
        raise ValueError("Wrong beamforming matrix size")

    payload = inBytes[20:]

    csi = np.zeros((Ntx, Nrx, 30), dtype=np.complex128)

    index = 0

    for sc in range(30):

        index += 3
        remainder = index % 8

        for nr in range(Nrx):
            for nt in range(Ntx):

                byte_index = index // 8

                tmp = (
                    (payload[byte_index] >> remainder)
                    | (payload[byte_index + 1] << (8 - remainder))
                )
                real = np.int8(tmp & 0xFF)

                tmp = (
                    (payload[byte_index + 1] >> remainder)
                    | (payload[byte_index + 2] << (8 - remainder))
                )
                imag = np.int8(tmp & 0xFF)

                csi[nt, nr, sc] = complex(real, imag)

                index += 16

    perm = np.array([
        (antenna_sel & 0x3) + 1,
        ((antenna_sel >> 2) & 0x3) + 1,
        ((antenna_sel >> 4) & 0x3) + 1
    ])

    return {
        "timestamp_low": float(timestamp_low),
        "bfee_count": float(bfee_count),
        "Nrx": int(Nrx),
        "Ntx": int(Ntx),
        "rssi_a": int(rssi_a),
        "rssi_b": int(rssi_b),
        "rssi_c": int(rssi_c),
        "noise": int(noise),
        "agc": int(agc),
        "perm": perm,
        "rate": float(fake_rate_n_flags),
        "csi": csi
    }