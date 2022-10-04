import argparse
import clipboard

from utils import extract_json


def process_eai_tasks(eai_results: dict, to_clipboard: bool = True) -> str:
    spreadsheet_content = ""
    for dataset in eai_results["results"]:
        spreadsheet_content += f"{dataset}"
        for metric in eai_results["results"][dataset]:
            if "stderr" not in metric:
                spreadsheet_content += f"\t{metric}\n"

    if to_clipboard:
        clipboard.copy(spreadsheet_content)
        print("🟢 Tasks & metrics added to clipboard, ready to paste in spreadsheet!")

    return spreadsheet_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get formatted EAI tasks for c/p in spreadsheet"
    )
    parser.add_argument(
        "-i", "--eai-results-file", type=str, help="Path to an EAI .json result file"
    )
    args = parser.parse_args()

    process_eai_tasks(extract_json(args.eai_results_file))
