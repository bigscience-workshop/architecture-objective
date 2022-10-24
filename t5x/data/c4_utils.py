# Copyright 2022
import os

def get_c4_files():
    """name of the c4 files"""

    return {
        "train": [f"c4-train.{i:05}-of-01024.json" for i in range(1022)],
        "val": [f"c4-train.01022-of-01024.json"],
        "test": [f"c4-train.01023-of-01024.json"],
        }
