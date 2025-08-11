import numpy as np
def exp_decay_weights(n, half_life):
    if n <= 0:
        return np.array([])
    lam = np.log(2) / max(1e-9, half_life)
    idx = np.arange(n)
    w = np.exp(lam * idx)
    return w / w.sum()
def weighted_mean_var(values, half_life=12):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float('nan'), float('nan')
    w = exp_decay_weights(n, half_life)
    m = float(np.dot(x, w))
    var = float(np.dot(w, (x - m)**2)) * (n/(n-1)) if n > 1 else 0.0
    return m, var