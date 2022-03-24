import argparse
import clipboard
import numpy as np

from utils import extract_json


def process_ps_tasks(ps_results: dict, to_clipboard: bool = True) -> str:
    spreadsheet_content, previous_tasks = "", set()
    for task_prompt in ps_results:
        task, prompt = process_task_prompt(task_prompt)
        print(task, prompt)
        if task not in previous_tasks:
            spreadsheet_content += f"{task}\tacc_avg\n\tacc_best\n"
        previous_tasks.add(task)

    if to_clipboard:
        clipboard.copy(spreadsheet_content)
        print("🟢 Results added to clipboard, ready to paste in spreadsheet!")

    return spreadsheet_content


def process_ps_results(ps_resuls:dict, to_clipboard: bool = True) -> str:
    spreadsheet_content = ""
    formatted_results = {}
    for task_prompt in ps_resuls:
        task, prompt = process_task_prompt(task_prompt)

        if task not in formatted_results:
            formatted_results[task] = {}

        formatted_results[task][prompt] = ps_resuls[task_prompt]["accuracy"]

    for task in formatted_results:
        results = list(formatted_results[task].values())
        acc_avg, acc_std, acc_best = np.median(results) / 100, np.std(results) / 100, max(results) / 100
        spreadsheet_content += f"{acc_avg}\t{acc_std}\n{acc_best}\n"

    if to_clipboard:
        clipboard.copy(spreadsheet_content)
        print("🟢 Results added to clipboard, ready to paste in spreadsheet!")

    return spreadsheet_content


def process_task_prompt(task_prompt: str) -> tuple[str, str]:
    task_prompt = task_prompt[:-11]  # Remove 'score_eval' string at the end

    task, prompt = None, None
    if "anli" in task_prompt:
        task = "anli" + task_prompt[-3:]
        prompt = task_prompt[5:-3]
    elif "hellaswag" in task_prompt:
        task = "hellaswag"
        prompt = task_prompt[10:]
    elif "story_cloze" in task_prompt:
        task = "story_cloze"
        prompt = task_prompt[17:]
    elif "super_glue" in task_prompt:
        if "cb" in task_prompt:
            task = "cb"
            prompt = task_prompt[14:]
        elif "copa" in task_prompt:
            task = "copa"
            prompt = task_prompt[16:]
        elif "rte" in task_prompt:
            task = "rte"
            prompt = task_prompt[15:]
        elif "wic" in task_prompt:
            task = "wic"
            prompt = task_prompt[15:]
        elif "wsc" in task_prompt:
            task = "wsc"
            prompt = task_prompt[15:]
    elif "winogrande" in task_prompt:
        task = "winogrande"
        prompt = task_prompt[25:]

    if task is None or prompt is None:
        raise ValueError(f"Failed to parse task/prompt: {task_prompt}")

    return task, prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get formatted t0 tasks & results for c/p in spreadsheet"
    )
    parser.add_argument(
        "-i",
        "--ps-results-file",
        type=str,
        help="Path to a t0 .json result file",
    )
    parser.add_argument(
        "-t",
        "--get-tasks",
        action="store_true",
        help="Get tasks headers instead of results",
    )
    args = parser.parse_args()

    if args.get_tasks:
        process_ps_tasks(extract_json(args.ps_results_file))
    else:
        process_ps_results(extract_json(args.ps_results_file))
