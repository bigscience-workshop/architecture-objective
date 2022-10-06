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
    BASE_PILE_URL = "https://the-eye.eu/public/AI/pile"
    local_path = f"{local_base_dir}/{relative_path}"

    # Create folder
    process = subprocess.Popen(["mkdir", "-p", local_path.rsplit("/", 1)[0]])
    process.wait()

    # download files
    wget.download(f"{BASE_PILE_URL}/{relative_path}", local_path)
    process.wait()

    # decompress files
    process = subprocess.Popen(['zstd', '-d', local_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()

    assert local_path.endswith(".zst")
    local_uncompressed_path = local_path[:-4]
    assert relative_path.endswith(".zst")
    gcp_uncompressed_path = f"{gcp_base}/{relative_path[:-4]}"

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
    local_base_dir = args.local_base_dir
    gcp_base = "gs://bigscience/pile/raw"

    process = subprocess.Popen(["mkdir", "-p", local_base_dir])
    process.wait()

    # pool = Pool(args.procs)
    # pool.map(
    #     functools.partial(download_unztd_and_send_to_gcloud, local_base_dir=local_base_dir, gcp_base=gcp_base),
    #     [local_path for _, local_paths in pile_urls.items() for local_path in local_paths]
    # )
    for local_path in [local_path for _, local_paths in pile_urls.items() for local_path in local_paths]:
        download_unztd_and_send_to_gcloud(local_path, local_base_dir=local_base_dir, gcp_base=gcp_base)

if __name__ == "__main__":
    main()