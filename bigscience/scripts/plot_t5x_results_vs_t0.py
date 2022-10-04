import csv
import json
import re
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--all', action="store_true", help="Plot all results in a single plot")
    parser.add_argument('--per-arch', action="store_true", help="Plot results grouped by architectures")
    parser.add_argument('--per-objective', action="store_true", help="Plots results grouped by objectives")
    parser.add_argument('--per-t0-adapted', action="store_true", help="Plots only T0 adapted models")
    parser.add_argument('--aggregated-results', action="store_true", help="Plots agregated results")
    args = parser.parse_args()

    assert args.all or args.per_arch or args.per_objective or args.per_t0_adapted

    return args

def load_t0_results(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))

def load_t5x_results(dir_path: Path):
    def remove_t0_eval(filename:str):
        name = filename.replace("_t0_eval", "")
        name = name.replace("_bs2048", "")
        name = name.replace("_c4", "")
        return name

    all_results = {}
    for child in dir_path.iterdir():
        filepath = child / "results.json"
        if filepath.is_file():
            with open(filepath, "r") as fi:
                results = json.load(fi)
            all_results[remove_t0_eval(child.name)] = results
    print(all_results.keys())
    return all_results

def get_experiment_name(filename: str):
    name = re.sub(r"_([0-9]*)$", r" [\1]", filename)
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

TASKS = {
    'super_glue_copa': ('COPA', 0.5),
    'anli_r1': ('ANLI R1', 1/3),
    'anli_r2': ('ANLI R2', 1/3),
    'anli_r3': ('ANLI R3', 1/3),
    'super_glue_cb': ('CB', 1/3),
    'super_glue_rte': ('RTE', 0.5),
    'super_glue_wsc.fixed': ('WSC', 0.5),
    'winogrande_winogrande_xl': ('Winogrande', 0.5),
    'super_glue_wic': ('WiC', 0.5),
    'hellaswag': ('HellaSwag', 0.25),
    'story_cloze_2016': ('StoryCloze', 0.5),
}
def plot(t5x_data, t0_data):
    args = get_args()

    t5x_data, t5x_experiments = t5x_data
    assert len(TASKS) == 11
    fig, axs = plt.subplots(2, 6, figsize=(20, 8))
    axs = axs.flatten()

    task_min_score = {}
    task_max_score = {}
    task_median_score = {}
    for n, (task, (task_name, random_baseline)) in enumerate(TASKS.items()):
        t5lm_scores = [float(r["score"]) for r in t0_data
                       if r["runs"] == "xxl-lm-d4-091621"
                       and r["dataset_name"] == task
                       and r["metric_name"] == "accuracy (Rank)"
                       and r["score"]]
        t0_scores = [float(r["score"]) for r in t0_data
                     if r["runs"] == "xxl-lm-d4-091621-512"
                     and r["dataset_name"] == task
                     and r["metric_name"] == "accuracy (Rank)"
                     and r["score"]]
        t5x_scores_with_name = [
            (
                get_experiment_name(name),
                [s["accuracy"] for k, s in t5x_data[name].items() if task.replace("anli_", "") in k]
            )
            for name in t5x_experiments
        ]

        all_experiment_scores_with_name = [("T5 + LM", t5lm_scores), ("T0", t0_scores), *t5x_scores_with_name]
        # Plot
        axs[n].axhline(100 * random_baseline, 0, len(all_experiment_scores_with_name), label="Random")
        for i, (exp_name, scores) in enumerate(all_experiment_scores_with_name):
            axs[n].scatter([i] * len(scores), scores, s=50, alpha=0.4, label=exp_name)
        axs[n].set_title(task_name)

        # Gather median values
        task_min_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.min(scores)) for (exp_name, scores) in all_experiment_scores_with_name]
        task_max_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.max(scores)) for (exp_name, scores) in all_experiment_scores_with_name]
        task_median_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.median(scores)) for (exp_name, scores) in all_experiment_scores_with_name]

    last_ax_id = len(TASKS) - 1
    axs[last_ax_id].legend(bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs[last_ax_id + 1:]:
        ax.set_visible(False)

    if args.aggregated_results:
        # ====== Plot agregated values =======
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        axs = axs.flatten()
        last_ax_id=0
        experiment_names = [elt[0] for elt in next(iter(task_median_score.values()))]

        def plot_scores_with_name(median_score_with_name, max_score, min_score, ax, title):
            assert len(median_score_with_name) == len(max_score) and len(median_score_with_name) == len(min_score)
            ax.axhline(
                median_score_with_name[0][1],
                0, len(median_score_with_name) - 1,
                label=median_score_with_name[0][0]
            )
            for i, ((name, median_score), max_score, min_score) in enumerate(zip(median_score_with_name[1:], max_score[1:], min_score[1:])):
                ax.errorbar(
                    i, median_score, ((median_score - min_score,), (max_score - median_score,)),
                    fmt="o", elinewidth=1, label=name)
            ax.set_title(title)

        def get_average_normalised_score(task_scores):
            normalised_scores = []
            for scores_with_name in task_scores.values():
                random_name, random_baseline = scores_with_name[0]
                assert random_name == "Random"
                normalised_scores_per_task = [(scores - random_baseline) / (100 - random_baseline) for _, scores in
                                              scores_with_name]
                normalised_scores.append(normalised_scores_per_task)
            return np.mean(normalised_scores, axis=0)

        def get_average_score(task_scores):
            return np.mean(
                [[scores for _, scores in scores_with_name] for scores_with_name in task_scores.values()], axis=0)

        # Plot average task score
        average_task_median_score = get_average_score(task_median_score)
        assert len(experiment_names) == len(average_task_median_score)
        average_task_media_score_with_name = list(zip(experiment_names, average_task_median_score))
        del average_task_median_score
        plot_scores_with_name(
            median_score_with_name=average_task_media_score_with_name,
            max_score=get_average_score(task_max_score),
            min_score=get_average_score(task_min_score),
            ax=axs[last_ax_id],
            title=f"Average of task median scores"
        )
        last_ax_id += 1

        # Plot average of task median normalised scores `normalised_score = (score - random) / (1 - random)`
        average_task_normalised_median_score = get_average_normalised_score(task_median_score)
        assert len(experiment_names) == len(average_task_normalised_median_score)
        average_task_normalised_median_score_with_name = list(
            zip(experiment_names, average_task_normalised_median_score))
        del average_task_normalised_median_score
        plot_scores_with_name(
            median_score_with_name=average_task_normalised_median_score_with_name,
            max_score=get_average_normalised_score(task_max_score),
            min_score=get_average_normalised_score(task_min_score),
            ax=axs[last_ax_id],
            title=f"Average of task normalised median scores"
        )
        last_ax_id += 1

        axs[last_ax_id -1].legend(bbox_to_anchor=(1, 1), loc="upper left")
        for ax in axs[last_ax_id:]:
            ax.set_visible(False)


def main():
    args = get_args()

    # Define directories
    results_dir = Path(__file__).resolve().parent.parent / "results" / "t0_eval"
    t0_results_dir = results_dir / "t0"
    t5x_results_dir = results_dir / "t5x"
    subprocess.run(["mkdir", "-p", t0_results_dir])
    subprocess.run(["mkdir", "-p", t5x_results_dir])

    # Sync previous results
    # gsutil cp gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv ../results/t0_eval/t0
    if not (t0_results_dir / "all_datasets_and_runs.csv").exists():
        subprocess.run(["gsutil", "cp", "gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv", t0_results_dir])
    # gsutil rsync -rd gs://bigscience-t5x/arch_objective_exps_v2/t0_eval ../results/t0_eval/t5x
    subprocess.run(["gsutil", "-m", "rsync", "-rd", "-x", ".*inference_eval", "gs://bigscience-t5x/arch_objective_exps_v2/t0_eval", t5x_results_dir])

    # Load results
    t0_data = load_t0_results(t0_results_dir / "all_datasets_and_runs.csv")
    t5x_data = load_t5x_results(t5x_results_dir)

    # Plot results
    # We group experiments by:
    #  - objective
    #  - architecture
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
        for max_steps,_ in PRETRAIN_AND_T0_ADAPT_STEPS:
            suffixes += [
                f"lm_{max_steps}",
                *[f"{lm_type}_adapt_{lm_adapt}_{max_steps}" for lm_adapt in LM_ADAPT_FROM for
                  lm_type in ["_lm", "_flm", "_plm"]]
            ]
        for t0_adapt_from, max_steps in PRETRAIN_AND_T0_ADAPT_STEPS:
            suffixes += [
                f"lm_t0_adapt_{t0_adapt_from}_{max_steps}",
                f"lm_t0_adapt_nc_{t0_adapt_from}_{max_steps}",
                f"span_corruption_t0_adapt_{t0_adapt_from}_{max_steps}",
                *[f"{lm_type}_adapt_{lm_adapt}_t0_adapt_{t0_adapt_from}_{max_steps}" for lm_adapt in LM_ADAPT_FROM for
                  lm_type in ["_lm", "_flm", "_plm"]],
                f"-nc_sc_{t0_adapt_from}-nc_t0_{max_steps}"
            ]

        for i, suffix in enumerate(suffixes):
            if experiment_name.endswith(suffix):
                return i
        raise NotImplementedError(f"{experiment_name}")

    t5x_experiments = list(t5x_data.keys())
    # Define single ordering
    t5x_experiments = sorted(t5x_experiments, key=lambda x: (key_objective(x), key_architecture(x)))

    if args.all:
        plot((t5x_data, t5x_experiments), t0_data)

    def plot_per_group(group_fn):
        t5x_objective_keys = set(group_fn(x) for x in t5x_experiments)
        for group_id in t5x_objective_keys:
            t5x_experiments_per_group = [x for x in t5x_experiments if group_id == group_fn(x)]
            plot((t5x_data, t5x_experiments_per_group), t0_data)
    if args.per_objective:
        plot_per_group(key_objective)
    if args.per_arch:
        plot_per_group(key_architecture)
    if args.per_t0_adapted:
        def key_is_t0_adapted(experiment_name):
            return "_t0" in experiment_name
        plot_per_group(key_is_t0_adapted)

    plt.show()
    print("Finished")

if __name__ == "__main__":
    main()