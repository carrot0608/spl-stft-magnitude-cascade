"""Nine-path complex S21 representations."""
import numpy as np
from scipy.signal import stft


def build_features(spectra):
    """Return STFT maps (N,9,17,23) and absolute dB vectors (N,1809)."""
    spectra = np.asarray(spectra)
    if spectra.ndim != 3 or spectra.shape[1:] != (9, 201):
        raise ValueError('Expected complex spectra with shape (N,9,201).')
    if not np.iscomplexobj(spectra) or not np.isfinite(spectra).all():
        raise ValueError('S21 must contain finite complex values.')
    cubes = []
    for sample in spectra:
        delay = np.abs(np.fft.ifft(sample, axis=1)).astype(np.float32)
        delay -= delay.mean(axis=1, keepdims=True)
        maps = []
        for path in delay:
            _, _, z = stft(path, window='hann', nperseg=32, noverlap=24,
                           nfft=32, boundary=None, padded=True)
            maps.append(np.log1p(np.abs(z).astype(np.float32)))
        cube = np.stack(maps).astype(np.float32)
        cube = (cube-cube.mean(axis=(1,2), keepdims=True)) / (cube.std(axis=(1,2), keepdims=True)+1e-6)
        cubes.append(cube)
    magnitude = (20*np.log10(np.abs(spectra).astype(np.float64)+1e-12)).reshape(len(spectra), -1)
    return np.stack(cubes), magnitude.astype(np.float32)
