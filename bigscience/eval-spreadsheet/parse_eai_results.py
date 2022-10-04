import argparse
import clipboard

from utils import extract_json


def process_eai_results(eai_results: dict, to_clipboard: bool = True) -> str:
    spreadsheet_content, previous_metric = "", ""
    for dataset in eai_results["results"]:
        for metric in eai_results["results"][dataset]:
            if previous_metric not in metric:
                spreadsheet_content += "\n"
            spreadsheet_content += f"{eai_results['results'][dataset][metric]}\t"
            previous_metric = metric

    if to_clipboard:
        clipboard.copy(spreadsheet_content)
        print("🟢 Results added to clipboard, ready to paste in spreadsheet!")

    return spreadsheet_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get formatted EAI results for c/p in spreadsheet"
    )
    parser.add_argument(
        "-i", "--eai-results-file", type=str, help="Path to an EAI .json result file"
    )
    args = parser.parse_args()

    process_eai_results(extract_json(args.eai_results_file))
