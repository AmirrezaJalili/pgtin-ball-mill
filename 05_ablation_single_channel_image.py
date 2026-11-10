"""Ablation: only the value channel is kept, without deviation and saliency."""
import numpy as np

from ablation_common import pgtin, run


def value_channel_only(X, bank, median, saliency):
    return pgtin._splat(X, bank)[:, None].astype(np.float32)


if __name__ == "__main__":
    run("single_channel_image", __doc__, image_builder=value_channel_only)
