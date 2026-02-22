import numpy as np
import copy

def dbinv(x):
    return 10.0 ** (x / 10.0)


def dbpow(x):
    return 10.0 * np.log10(x)


sm_1 = np.array([[1]], dtype=np.complex128)

sm_2_20 = np.array([
    [1, 1],
    [1, -1]
], dtype=np.complex128) / np.sqrt(2)

sm_2_40 = np.array([
    [1, 1j],
    [1j, 1]
], dtype=np.complex128) / np.sqrt(2)

sm_3_20_phase = np.array([
    [-2*np.pi/16,       -2*np.pi/(80/33),   2*np.pi/(80/3)],
    [ 2*np.pi/(80/23),   2*np.pi/(48/13),   2*np.pi/(240/13)],
    [-2*np.pi/(80/13),   2*np.pi/(240/37),  2*np.pi/(48/13)]
], dtype=np.float64)
sm_3_20 = np.exp(1j * sm_3_20_phase) / np.sqrt(3)

sm_3_40_phase = np.array([
    [-2*np.pi/16,       -2*np.pi/(80/13),    2*np.pi/(80/23)],
    [-2*np.pi/(80/37),  -2*np.pi/(48/11),   -2*np.pi/(240/107)],
    [ 2*np.pi/(80/7),   -2*np.pi/(240/83),  -2*np.pi/(48/11)]
], dtype=np.float64)
sm_3_40 = np.exp(1j * sm_3_40_phase) / np.sqrt(3)


def remove_sm(csi, rate):
    M, _, S = csi.shape
    if M == 1:
        return csi.copy()

    if (rate & 2048) == 2048:
        if M == 3:
            sm = sm_3_40
        elif M == 2:
            sm = sm_2_40
        else:
            raise ValueError("Unsupported TX number")
    else:
        if M == 3:
            sm = sm_3_20
        elif M == 2:
            sm = sm_2_20
        else:
            raise ValueError("Unsupported TX number")

    ret = np.zeros_like(csi, dtype=np.complex128)

    for i in range(S):
        t = csi[:, :, i]
        H = t.T @ sm.conj().T
        ret[:, :, i] = H.T

    return ret


def get_total_rss(csi_st):
    rssi_mag = 0.0

    if csi_st["rssi_a"] != 0:
        rssi_mag += dbinv(csi_st["rssi_a"])
    if csi_st["rssi_b"] != 0:
        rssi_mag += dbinv(csi_st["rssi_b"])
    if csi_st["rssi_c"] != 0:
        rssi_mag += dbinv(csi_st["rssi_c"])

    if rssi_mag == 0:
        return -np.inf

    return dbpow(rssi_mag) - 44 - csi_st["agc"]


def scale_csi_5300(csi_origin):
    csi = copy.deepcopy(csi_origin)

    for pa in range(len(csi)):
        if csi[pa] is None:
            continue
        csi[pa]["csi"] = get_scaled_csi_sm(csi[pa])

    return csi


def get_scaled_csi_sm(csi_st):
    ret = get_scaled_csi(csi_st)
    ret = remove_sm(ret, csi_st["rate"])
    return ret


def get_scaled_csi(csi_st):
    csi = csi_st["csi"]

    csi_pwr = np.sum(np.abs(csi) ** 2)
    if csi_pwr == 0:
        return csi.copy()

    rssi_pwr = dbinv(get_total_rss(csi_st))
    scale = rssi_pwr / (csi_pwr / 30.0)

    noise_db = -92 if csi_st["noise"] == -127 else csi_st["noise"]
    thermal_noise_pwr = dbinv(noise_db)

    quant_error_pwr = scale * (csi_st["Nrx"] * csi_st["Ntx"])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    if total_noise_pwr == 0:
        return csi.copy()

    ret = csi * np.sqrt(scale / total_noise_pwr)

    if csi_st["Ntx"] == 2:
        ret *= np.sqrt(2)
    elif csi_st["Ntx"] == 3:
        ret *= np.sqrt(dbinv(4.5))

    return ret