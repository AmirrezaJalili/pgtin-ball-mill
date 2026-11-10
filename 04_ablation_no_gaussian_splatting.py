"""Ablation: no Gaussian splatting. Each feature lights up one pixel of the plain grid."""
import numpy as np

from ablation_common import run


def one_pixel_bank(positions, side, sigma=0.35):
    # No upscaling, and a blob so narrow that it stays almost entirely in its own pixel.
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    bank = np.zeros((len(positions), side, side), dtype=np.float32)
    for j, (r, c) in enumerate(positions):
        bank[j] = np.exp(-((yy - (r + 0.5)) ** 2 + (xx - (c + 0.5)) ** 2) / (2.0 * sigma ** 2))
    return bank


if __name__ == "__main__":
    run("no_gaussian_splatting", __doc__, bank_builder=one_pixel_bank)
