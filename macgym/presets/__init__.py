"""Preset robot files shown in the playground editor (each defines `robot` and `task`)."""
import os

NAMES = ("walker", "cheetah", "hopper", "tripod", "worm", "blank")
DIR = os.path.dirname(os.path.abspath(__file__))


def source(name):
    """Return the Python source of a preset, or None if unknown."""
    if name not in NAMES:
        return None
    with open(os.path.join(DIR, f"{name}.py")) as f:
        return f.read()
