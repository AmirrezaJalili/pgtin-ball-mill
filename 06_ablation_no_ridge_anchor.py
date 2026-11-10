"""Ablation: no ridge anchor. The CNN corrects the training mean instead of ridge."""
from ablation_common import run

if __name__ == "__main__":
    run("no_ridge_anchor", __doc__, use_ridge_anchor=False)
