import numpy as np
import pywt


def remove_empty_csi(csi_origin):
    csi_clean = []

    for item in csi_origin:
        if item is None:
            continue

        if (
            isinstance(item, dict)
            and "csi" in item
            and item["csi"] is not None
            and len(item["csi"]) > 0
        ):
            csi_clean.append(item)

    return csi_clean


def get_amplitude(csi):
    PA = len(csi)
    TX, RX, SC = csi[0]["csi"].shape

    amp = np.zeros((PA, TX, RX, SC), dtype=np.float64)

    for pa in range(PA):
        amp[pa] = np.abs(csi[pa]["csi"])

    return amp


def hampel_filter(signal, window):
    n = len(signal)
    new_signal = signal.copy()
    k = 1.4826

    if n < 2 * window:
        return signal

    for i in range(window, n - window):
        win = signal[i - window:i + window + 1]
        median = np.median(win)
        mad = k * np.median(np.abs(win - median))

        if mad == 0:
            continue

        if np.abs(signal[i] - median) > 3 * mad:
            new_signal[i] = median

    return new_signal


def amp_hampel(amp, window):
    PA, TX, RX, SC = amp.shape
    out = np.zeros_like(amp)

    for tx in range(TX):
        for rx in range(RX):
            for sc in range(SC):
                fiber = amp[:, tx, rx, sc]
                out[:, tx, rx, sc] = hampel_filter(fiber, window)

    return out


def wavelet_denoise(signal, wavelet="sym6", level=5):
    max_level = pywt.dwt_max_level(
        len(signal),
        pywt.Wavelet(wavelet).dec_len
    )
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    new_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:
            new_coeffs.append(c)
        else:
            new_coeffs.append(pywt.threshold(c, uthresh, mode="soft"))

    rec = pywt.waverec(new_coeffs, wavelet)
    return rec[:len(signal)]


def amp_DWT(amp):
    PA, TX, RX, SC = amp.shape
    out = np.zeros_like(amp)

    for tx in range(TX):
        for rx in range(RX):
            for sc in range(SC):
                fiber = amp[:, tx, rx, sc]
                fiber = wavelet_denoise(fiber)
                out[:, tx, rx, sc] = fiber[:PA]

    return out