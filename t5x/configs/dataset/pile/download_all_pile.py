import argparse
import functools
import subprocess
from multiprocessing import Pool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--procs", type=int, required=True, help="Number of processes."
    )
    parser.add_argument(
        "--base-dir", type=str, required=True, help="Folder to download the document to"
    )
    return parser.parse_args()


def download_and_unztd(relative_path, base_dir):
    BASE_PILE_URL = "https://the-eye.eu/public/AI/pile"
    local_path = f"{base_dir}/{relative_path}"

    # Create folder
    process = subprocess.Popen(["mkdir", "-p", local_path.rsplit("/", 1)])
    process.wait()

    # download files
    process = subprocess.Popen(['wget', "-O", local_path , f"{BASE_PILE_URL}/{relative_path}"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()

    # decompress files
    process = subprocess.Popen(['zstd', '-d', local_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()

def main():
    args = get_args()

    pile_urls = {
        "train": [
            f"train/{i:02d}.jsonl.zst" for i in range(30)
        ],
        "test": [
            f"test.jsonl.zst"
        ],
        "val": [
            f"val.jsonl.zst"
        ]
    }
    base_dir = args.base_dir
    gcp_base = "gs://bigscience/pile/raw"

    process = subprocess.Popen(["mkdir", "-p", base_dir])
    process.wait()

    pool = Pool(args.procs)

    pool.imap(
        functools.partial(download_and_unztd, base_dir=base_dir),
        [local_path for _, local_paths in pile_urls for local_path in local_paths]
    )

    process = subprocess.Popen(["gsutil", "cp", "-r", base_dir, gcp_base])
    process.wait()

if __name__ == "__main__":
    main()