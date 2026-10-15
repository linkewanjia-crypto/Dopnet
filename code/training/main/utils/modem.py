import numpy as np

MODULATIONS = ["bpsk", "qpsk", "16qam", "64qam"]
MOD_TO_ID = {name: i for i, name in enumerate(MODULATIONS)}
ID_TO_MOD = {i: name for name, i in MOD_TO_ID.items()}


def bits_per_symbol(modulation):
    return {
        "bpsk": 1,
        "qpsk": 2,
        "16qam": 4,
        "64qam": 6,
    }[str(modulation).lower()]


def modulation_order(modulation):
    return 2 ** bits_per_symbol(modulation)


def _gray_to_binary(g):
    g = np.asarray(g, dtype=np.int64)
    b = g.copy()
    shift = 1
    # Enough for the QAM orders used here.
    while shift < 8:
        b ^= (b >> shift)
        shift <<= 1
    return b


def labels_to_bits(labels, modulation):
    modulation = str(modulation).lower()
    bps = bits_per_symbol(modulation)
    labels = np.asarray(labels, dtype=np.int64)
    shifts = np.arange(bps - 1, -1, -1, dtype=np.int64)
    return ((labels[..., None] >> shifts) & 1).astype(np.int64)


def labels_to_symbols(labels, modulation):
    modulation = str(modulation).lower()
    labels = np.asarray(labels, dtype=np.int64)

    if modulation == "bpsk":
        return (2.0 * labels.astype(np.float64) - 1.0).astype(np.complex64)

    M = modulation_order(modulation)
    side = int(np.sqrt(M))
    bps_axis = bits_per_symbol(modulation) // 2

    # Label bits: first half -> I Gray code, second half -> Q Gray code.
    qmask = (1 << bps_axis) - 1
    gq = labels & qmask
    gi = labels >> bps_axis

    bi = _gray_to_binary(gi)
    bq = _gray_to_binary(gq)

    i = 2.0 * bi - (side - 1)
    q = 2.0 * bq - (side - 1)

    # Average square-QAM energy = 2/3*(M-1).
    norm = np.sqrt((2.0 / 3.0) * (M - 1))
    return ((i + 1j*q) / norm).astype(np.complex64)


def random_symbols(shape, modulation, rng):
    M = modulation_order(modulation)
    labels = rng.integers(0, M, size=shape, dtype=np.int64)
    return labels, labels_to_symbols(labels, modulation)


def demodulate_to_labels(symbols, modulation):
    symbols = np.asarray(symbols)
    M = modulation_order(modulation)
    const_labels = np.arange(M, dtype=np.int64)
    const = labels_to_symbols(const_labels, modulation)
    dist = np.abs(symbols[..., None] - const.reshape((1,) * symbols.ndim + (M,))) ** 2
    return np.argmin(dist, axis=-1).astype(np.int64)


def bit_error_count(pred_labels, true_labels, modulation):
    pb = labels_to_bits(pred_labels, modulation).reshape(-1)
    tb = labels_to_bits(true_labels, modulation).reshape(-1)
    return int(np.count_nonzero(pb != tb)), int(tb.size)

