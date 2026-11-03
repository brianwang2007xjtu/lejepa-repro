import os
import sys

import torch

# Make the project root importable (models/, losses/, ...) regardless of pytest's rootdir.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch.set_printoptions(precision=6, sci_mode=False)


def pytest_report_header(config):
    return (
        f"lejepa-reproduction tests | torch={torch.__version__} | "
        f"cuda={torch.cuda.is_available()}"
    )
