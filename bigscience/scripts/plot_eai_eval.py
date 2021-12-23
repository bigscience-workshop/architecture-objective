import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# gcloud alpha compute tpus tpu-vm scp thomas-dev-tpu:~/arch_objective_exps_v2 .  --zone us-central2-b --recurse

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--normalised', action="store_true", help="Whether to plot normalised scores or not. Each task has a random baseline and we compute how far away we're from that baseline")
    return parser.parse_args()

def load_data(dir_path: Path):
    def get_experiment_name(filename):
        name, empty = filename.rsplit(".json", maxsplit=1)
        assert empty == ""
        name = name.replace("_bs2048", "")
        return name

    all_results = {}
    for child in dir_path.iterdir():
        child_name = get_experiment_name(child.name)
        with open(child, "r") as fi:
            results = json.load(fi)["results"]
        all_results[child_name] = results

    return all_results

RANDOM_BASELINE={
    "arc_challenge_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "arc_easy_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "boolq_acc": 0.5,
    "copa_acc": 0.5,
    "headqa_acc": 0.25,
    "hellaswag_acc": 0.25,
    "lambada_acc": 0., # Safe to say that random models won't perform well at all.
    "logiqa_acc": 0.25,
    "mathqa_acc": (4360 * 1/ 5 - (4475 - 4360) * 1/ 4) / 4475,
    "mrpc_acc": 0.5,
    "multirc_acc": 0., # TODO: I couldn't figure it out
    "openbookqa_acc": 0.25,
    "piqa_acc": 0.5,
    "prost_acc": 0.25,
    "pubmedqa_acc": 1/3,
    "qnli_acc": 0.5,
    "qqp_acc": 0.5,
    "race_acc": 0.25, # Source: https://arxiv.org/pdf/1704.04683.pdf table 5
    "rte_acc": 0.5,
    "sciq_acc": 0.25,
    "sst_acc": 0.5,
    "triviaqa_acc": 0.,
    "webqs_acc": 0.,
    "wic_acc": 0.5,
    "winogrande_acc": 0.5,
    "wnli_acc": 0.5,
    "wsc_acc": 0.5
}
def normalise_score(score, evaluation_name, metric_name):
    key = f"{evaluation_name}_{metric_name}"
    if key not in RANDOM_BASELINE:
        raise ValueError(f"{key} doesn't have a random baseline set yet.")
    return (score - RANDOM_BASELINE[key]) / (1 - RANDOM_BASELINE[key])

def main():
    args = get_args()

    # Define directories
    results_dir = Path(__file__).resolve().parent.parent / "results" / "eai_eval"
    subprocess.run(["mkdir", "-p", results_dir])

    # Update data locally
    # gsutil rsync -rd gs://bigscience-t5x/arc_objective_exps_v2/eai_eval ../results/eai_eval
    subprocess.run(["gsutil", "rsync", "-rd", "gs://bigscience-t5x/arc_objective_exps_v2/eai_eval", results_dir])

    # Load data
    data = load_data(results_dir)

    # Get evaluation_metric
    evaluation_metrics = [
        ("arc_challenge", "acc"),
        ("arc_easy", "acc"),
        ("boolq", "acc"),
        ("copa", "acc"),
        ("headqa", "acc"),
        ("hellaswag", "acc"),
        ("lambada", "acc"),
        ("logiqa", "acc"),
        ("mathqa", "acc"),
        ("mrpc", "acc"),
        ("multirc", "acc"),
        ("openbookqa", "acc"),
        ("piqa", "acc"),
        ("prost", "acc"),
        ("pubmedqa", "acc"),
        ("qnli", "acc"),
        ("qqp", "acc"),
        ("race", "acc"),
        ("rte", "acc"),
        ("sciq", "acc"),
        ("sst", "acc"),
        ("triviaqa", "acc"),
        ("webqs", "acc"),
        ("wic", "acc"),
        ("winogrande", "acc"),
        ("wnli", "acc"),
        ("wsc", "acc"),
    ]

    # Plot data
    ind = np.arange(len(evaluation_metrics))
    width = 1/ (len(data.keys()) + 1) # the width of the bars

    fig, ax = plt.subplots()

    for i, (experiment_name, experiment) in enumerate(data.items()):
        if args.normalised:
            normalised_scores = [
                normalise_score(experiment[evaluation_name][metric_name], evaluation_name, metric_name)
                for (evaluation_name, metric_name) in evaluation_metrics
            ]
            ax.bar(ind + i * width, normalised_scores, width, label=experiment_name)
        else:
            scores = [experiment[evaluation_name][metric_name] for (evaluation_name, metric_name) in evaluation_metrics]
            ax.bar(ind + i * width, scores, width, label=experiment_name)

    # add some text for labels, title and axes ticks
    if args.normalised:
        ax.set_ylabel('Normalised scores')
    else:
        ax.set_ylabel('Scores')
    ax.set_title('EAI harness')
    ax.set_xticks(ind + len(data.keys()) / 2 * width)
    ax.set_xticklabels(
        (f"{evaluation_name}_{metric_name}" for evaluation_name, metric_name in evaluation_metrics),
        rotation=80,
        ha="right"
    )

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
