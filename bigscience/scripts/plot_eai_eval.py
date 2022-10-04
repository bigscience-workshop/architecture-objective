import json
import re
import subprocess
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# gcloud alpha compute tpus tpu-vm scp thomas-dev-tpu:~/arch_objective_exps_v2 .  --zone us-central2-b --recurse

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--all', action="store_true", help="Plot all results in a single plot")
    parser.add_argument('--per-arch', action="store_true", help="Plot results grouped by architectures")
    parser.add_argument('--per-objective', action="store_true", help="Plots results grouped by objectives")
    parser.add_argument('--per-t0-adapted', action="store_true", help="Plots only T0 adapted models")
    parser.add_argument('--normalised', action="store_true", help="Whether to plot normalised scores or not. Each task has a random baseline and we compute how far away we're from that baseline")
    parser.add_argument('--only-t0-eval', action="store_true", help="Only plot the results from the t0 evaluation set")

    args = parser.parse_args()

    assert args.all or args.per_arch or args.per_objective or args.per_t0_adapted

    return args

def load_data(dir_path: Path):
    def remove_eai_eval(filename):
        name, empty = filename.rsplit(".json", maxsplit=1)
        assert empty == ""
        name = name.replace("_eai_eval", "")
        name = name.replace("_bs2048", "")
        name = name.replace("_c4", "")
        return name

    all_results = {}
    for child in dir_path.iterdir():
        child_name = remove_eai_eval(child.name)
        with open(child, "r") as fi:
            results = json.load(fi)["results"]
        all_results[child_name] = results

    print(all_results.keys())
    return all_results

def get_experiment_name(key: str):
    name = re.sub(r"_([0-9]*)$", r" [\1]", key)
    name = name.replace("span_corruption", "SC")
    name = re.sub(r"^enc_dec", "ED", name)
    name = re.sub(r"^nc_dec", "NCD", name)
    name = re.sub(r"^c_dec", 'CD', name)
    name = name.replace("full_lm", "FLM")
    name = name.replace("prefix_lm", "PLM")
    name = re.sub(r"t0_adapt_([0-9]+)", r"T0(\1)", name)
    if name[:3] == "CD_":
        name = re.sub(r"lm_adapt_([0-9]+)", r"FLM(\1)", name)
        name = re.sub(r"t0_adapt_nc_([0-9]+)", r"T0 AS NC (\1)", name)
        name = re.sub(r"nc_sc_([0-9]+)", r"SC as NC(\1)", name)
        name = re.sub(r"nc_t0_([0-9]+)", r"T0 as NC(\1)", name)
    elif name[:4] == "NCD_" or name[:3] == "ED_":
        if "flm_adapt" in name:
            name = re.sub(r"flm_adapt_([0-9]+)", r"FLM AS CD(\1)", name)
        else:
            name = re.sub(r"lm_adapt_([0-9]+)", r"PLM(\1)", name)
    else:
        raise NotImplementedError
    name = name.replace("_", " + ")
    return name

RANDOM_BASELINE={
    "anli_r1_acc": 1/3,
    "anli_r2_acc": 1 / 3,
    "anli_r3_acc": 1 / 3,
    "arc_challenge_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "arc_easy_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "boolq_acc": 0.5,
    "cb_acc": 0.5,
    "copa_acc": 0.5,
    "headqa_acc": 0.25,
    "headqa_en_acc": 0.25,
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

def plot_tasks(data, evaluation_metrics):
    data, sorted_experiment_keys = data

    fig, axs = plt.subplots(3, 11)
    agg_fig, agg_axs = plt.subplots(1,3)
    axs = axs.flatten()
    agg_axs = agg_axs.flatten()

    assert len(axs) >= len(evaluation_metrics)
    for (evaluation_name, metric_name, _), ax in zip(evaluation_metrics, axs):
        key = f"{evaluation_name}_{metric_name}"
        ax.set_title(key)
        ax.axhline(
            RANDOM_BASELINE[key],
            0, len(data),
            label="Random"
        )

    agg_axs[0].set_title("Average")
    agg_axs[0].axhline(
        np.mean(list(RANDOM_BASELINE.values())),
        0, len(data),
        label="Random"
    )
    agg_axs[1].set_title("Normalised average")
    agg_axs[1].axhline(
        0,
        0, len(data),
        label="Random"
    )

    for i, experiment_key in enumerate(sorted_experiment_keys):
        experiment_name = get_experiment_name(experiment_key)
        experiment = data[experiment_key]
        scores = [experiment[evaluation_name][metric_name] for (evaluation_name, metric_name, _) in evaluation_metrics]
        normalised_score = [
            normalise_score(experiment[evaluation_name][metric_name], evaluation_name, metric_name)
            for (evaluation_name, metric_name, _) in evaluation_metrics
        ]

        for j, score in enumerate(scores):
            axs[j].scatter(i, score, s=50, alpha=0.4, label=experiment_name)

        agg_axs[0].scatter(i, np.mean(scores), s=50, alpha=0.4, label=experiment_name)
        agg_axs[1].scatter(i, np.mean(normalised_score), s=50, alpha=0.4, label=experiment_name)

    last_ax_id = len(evaluation_metrics) -1
    axs[last_ax_id].legend(bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs[last_ax_id + 1:]:
        ax.set_visible(False)
    agg_axs[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    for ax in agg_axs[2:]:
        ax.set_visible(False)

def plot_bar(data, evaluation_metrics, normalised):
    ind = np.arange(len(evaluation_metrics))
    width = 1 / (len(data.keys()) + 1)  # the width of the bars

    fig, ax = plt.subplots()

    for i, (experiment_name, experiment) in enumerate(data.items()):
        if normalised:
            normalised_scores = [
                normalise_score(experiment[evaluation_name][metric_name], evaluation_name, metric_name)
                for (evaluation_name, metric_name, _) in evaluation_metrics
            ]
            ax.bar(ind + i * width, normalised_scores, width, label=experiment_name)
        else:
            scores = [experiment[evaluation_name][metric_name] for (evaluation_name, metric_name, _) in evaluation_metrics]
            ax.bar(ind + i * width, scores, width, label=experiment_name)

    # add some text for labels, title and axes ticks
    if normalised:
        ax.set_ylabel('Normalised scores')
    else:
        ax.set_ylabel('Scores')
    ax.set_title('EAI harness')
    ax.set_xticks(ind + len(data.keys()) / 2 * width)
    ax.set_xticklabels(
        (f"{evaluation_name}_{metric_name}" for evaluation_name, metric_name, _ in evaluation_metrics),
        rotation=80,
        ha="right"
    )
    ax.legend()

# All evaluation tasks available: (dataset_name, metric_name, is_t0_eval)
Evaluation = namedtuple('Evaluation', ['task', 'metric', "is_t0_eval"])
ALL_EVALUATION = [
    Evaluation("anli_r1", "acc", True),
    Evaluation("anli_r2", "acc", True),
    Evaluation("anli_r3", "acc", True),
    Evaluation("arc_challenge", "acc", False),
    Evaluation("arc_easy", "acc", False),
    Evaluation("boolq", "acc", False),
    Evaluation("cb", "acc", True), # https://github.com/EleutherAI/lm-evaluation-harness/pull/254
    Evaluation("copa", "acc", True),
    Evaluation("headqa_en", "acc", False),
    Evaluation("hellaswag", "acc", True),
    Evaluation("lambada", "acc", False),
    Evaluation("logiqa", "acc", False),
    Evaluation("mathqa", "acc", False),
    Evaluation("mrpc", "acc", False),
    Evaluation("multirc", "acc", False),
    Evaluation("openbookqa", "acc", False),
    Evaluation("piqa", "acc", False),
    Evaluation("prost", "acc", False),
    Evaluation("pubmedqa", "acc", False),
    Evaluation("qnli", "acc", False),
    Evaluation("qqp", "acc", False),
    Evaluation("race", "acc", False),
    Evaluation("rte", "acc", True),
    Evaluation("sciq", "acc", False),
    Evaluation("sst", "acc", False),
    Evaluation("triviaqa", "acc", False),
    Evaluation("webqs", "acc", False),
    Evaluation("wic", "acc", True),
    Evaluation("winogrande", "acc", True),
    Evaluation("wnli", "acc", False),
    Evaluation("wsc", "acc", True),
]

def main():
    args = get_args()

    # Define directories
    results_dir = Path(__file__).resolve().parent.parent / "results" / "eai_eval"
    subprocess.run(["mkdir", "-p", results_dir])

    # Update data locally
    # gsutil rsync -rd gs://bigscience-t5x/arch_objective_exps_v2/eai_eval ../results/eai_eval
    subprocess.run(["gsutil", "-m", "rsync", "-rd", "gs://bigscience-t5x/arch_objective_exps_v2/eai_eval", results_dir])

    # Load data
    data = load_data(results_dir)

    # Get evaluation_metric
    if args.only_t0_eval:
        evaluation_metrics = [elt for elt in ALL_EVALUATION if elt.is_t0_eval]
    else:
        evaluation_metrics = ALL_EVALUATION

    # Plot data
    # plot_bar(data, evaluation_metrics, args.normalised)


    # sort experiments
    LM_ADAPT_FROM = [28000, 30000, 58768]
    PRETRAIN_AND_T0_ADAPT_STEPS = [(32768, 37768), (65536, 70536), (131072, 141072), (169984, 179984), (196608, 206608)]
    def key_architecture(experiment_name):
        if experiment_name[0] == 'c':
            return 0
        elif experiment_name[0] == 'n':
            return 1
        elif experiment_name[0] == 'e':
            return 2
        else:
            raise NotImplementedError

    def key_objective(experiment_name):
        suffixes = []
        for max_steps, _ in PRETRAIN_AND_T0_ADAPT_STEPS:
            suffixes += [
                f"lm_{max_steps}",
                f"span_corruption_{max_steps}",
                *[f"{lm_type}_adapt_{lm_adapt}_{max_steps}" for lm_adapt in LM_ADAPT_FROM for
                  lm_type in ["_lm", "_flm", "_plm"]]
            ]
        for t0_adapt_from, max_steps in PRETRAIN_AND_T0_ADAPT_STEPS:
            suffixes += [
                f"lm_t0_adapt_{t0_adapt_from}_{max_steps}",
                f"lm_t0_adapt_nc_{t0_adapt_from}_{max_steps}",
                f"span_corruption_t0_adapt_{t0_adapt_from}_{max_steps}",
                *[f"{lm_type}_adapt_{lm_adapt}_t0_adapt_{t0_adapt_from}_{max_steps}" for lm_adapt in LM_ADAPT_FROM for lm_type in ["_lm", "_flm", "_plm"]],
                f"-nc_sc_{t0_adapt_from}-nc_t0_{max_steps}"
            ]

        for i, suffix in enumerate(suffixes):
            if experiment_name.endswith(suffix):
                return i
        raise NotImplementedError(f"{experiment_name}")
    sorted_experiment_keys = sorted(data.keys(), key=lambda x: (key_objective(x), key_architecture(x)))

    if args.all:
        plot_tasks((data, sorted_experiment_keys), evaluation_metrics)

    def plot_per_group(group_fn):
        t5x_objective_keys = set(group_fn(x) for x in sorted_experiment_keys)
        for group_id in t5x_objective_keys:
            t5x_experiments_per_group = [x for x in sorted_experiment_keys if group_id == group_fn(x)]
            plot_tasks((data, t5x_experiments_per_group), evaluation_metrics)

    if args.per_objective:
        plot_per_group(key_objective)
    if args.per_arch:
        plot_per_group(key_architecture)
    if args.per_t0_adapted:
        def key_is_t0_adapted(experiment_name):
            return "_t0" in experiment_name
        plot_per_group(key_is_t0_adapted)
    plt.show()

if __name__ == "__main__":
    main()
