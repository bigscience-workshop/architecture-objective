# Copyright 2022
import os

def get_c4_files():
    """name of the c4 files"""

    return {
        "train": [f"c4-train.{i:05}-of-01024.json" for i in range(1024)],
        "val": [f"c4-train.{i:05}-of-01024.json" for i in range(1024)],
        "test": [f"c4-train.{i:05}-of-01024.json" for i in range(1024)],
        }