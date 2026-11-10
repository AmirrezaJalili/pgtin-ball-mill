"""Ablation: features are placed on the grid at random instead of by IGTD."""
import numpy as np

from ablation_common import run


def random_layout(X, side, seed):
    cells = [(r, c) for r in range(side) for c in range(side)]
    order = np.random.default_rng(seed).permutation(len(cells))
    return [cells[i] for i in order[:X.shape[1]]]


if __name__ == "__main__":
    run("random_layout_instead_of_igtd", __doc__, layout=random_layout)
