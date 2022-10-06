# Adapted from thomasw21 from a version of this script downloading the Pile.
# Original can be found at https://github.com/bigscience-workshop/architecture-objective/blob/ec3989be6922e443138c8d5beba60166f88049c7/t5x/configs/dataset/pile/download_all_pile.py
from tqdm import tqdm

import argparse
import functools
import subprocess
from multiprocessing import Pool
import wget

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--procs", type=int, required=True, help="Number of processes."
    )
    parser.add_argument(
        "--local-base-dir", type=str, required=True, help="Folder to download the document to"
    )
    return parser.parse_args()


def download_unztd_and_send_to_gcloud(relative_path, local_base_dir, gcp_base):
    BASE_PILE_URL = "https://the-eye.eu/public/AI/STAGING/c4/en/"
    local_path = f"{local_base_dir}/{relative_path}"

    # Create folder
    process = subprocess.Popen(["mkdir", "-p", local_path.rsplit("/", 1)[0]])
    process.wait()

    # download files
    wget.download(f"{BASE_PILE_URL}/{relative_path}", local_path)
    process.wait()

    # decompress files
    process = subprocess.Popen(['gzip', '-d', local_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()

    assert local_path.endswith(".gz")
    local_uncompressed_path = local_path[:-3]
    assert relative_path.endswith(".gz")
    gcp_uncompressed_path = f"{gcp_base}/{relative_path[:-3]}"

    # upload to gcp
    process = subprocess.Popen(["gsutil", "cp", local_uncompressed_path, gcp_uncompressed_path])
    process.wait()

    # delete file locally
    process = subprocess.Popen(['rm', local_path])
    process.wait()
    process = subprocess.Popen(['rm', local_uncompressed_path])
    process.wait()

def main():
    args = get_args()

    urls = [
        f"c4-train.{i:05}-of-01024.json.gz"
        for i in range(1024)
    ]

    c4_urls = {
        "train": urls
    }
    local_base_dir = args.local_base_dir
    gcp_base = "gs://t5x-test/data/c4/raw/"

    process = subprocess.Popen(["mkdir", "-p", local_base_dir])
    process.wait()

    # pool = Pool(args.procs)
    # pool.map(
    #     functools.partial(download_unztd_and_send_to_gcloud, local_base_dir=local_base_dir, gcp_base=gcp_base),
    #     [local_path for _, local_paths in pile_urls.items() for local_path in local_paths]
    # )
    for local_path in [local_path for _, local_paths in tqdm(c4_urls.items()) for local_path in local_paths]:
        download_unztd_and_send_to_gcloud(local_path, local_base_dir=local_base_dir, gcp_base=gcp_base)

if __name__ == "__main__":
    main()