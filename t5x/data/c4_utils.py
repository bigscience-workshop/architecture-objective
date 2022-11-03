# Copyright 2022
import os

def get_c4_files(num_files_c4=1022):
    """name of the c4 files"""

    return {
        "train": [f"c4-train.{i:05}-of-01024.json" for i in range(num_files_c4)],
        "validation": [f"c4-train.01022-of-01024.json"],
        "test": [f"c4-train.01023-of-01024.json"],
        }
