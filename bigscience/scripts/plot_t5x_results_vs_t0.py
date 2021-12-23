import csv
import json
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

def load_t0_results(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))

def load_t5x_results(dir_path: Path):
    def get_experiment_name(filename):
        name = filename.split("_t0_eval_")[0]
        name = name.replace("_bs2048", "")
        return name

    all_results = {}
    for child in dir_path.iterdir():
        child_name = get_experiment_name(child.name)
        with open(child / "results.json", "r") as fi:
            results = json.load(fi)
        all_results[child_name] = results
    return all_results

def main():
    # Define directories
    results_dir = Path(__file__).resolve().parent.parent / "results" / "t0_eval"
    t0_results_dir = results_dir / "t0"
    t5x_results_dir = results_dir / "t5x"
    subprocess.run(["mkdir", "-p", t0_results_dir])
    subprocess.run(["mkdir", "-p", t5x_results_dir])

    # Sync previous results
    # gsutil cp gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv ../results/t0_eval/t0
    subprocess.run(["gsutil", "cp", "gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv", t0_results_dir])
    # gsutil rsync -rd gs://bigscience-t5x/arc_objective_exps_v2/t0_eval ../results/t0_eval/t5x
    subprocess.run(["gsutil", "rsync", "-rd", "gs://bigscience-t5x/arc_objective_exps_v2/t0_eval", t5x_results_dir])

    # Load results
    t0_data = load_t0_results(t0_results_dir / "all_datasets_and_runs.csv")
    t5x_data = load_t5x_results(t5x_results_dir)

    # Get tasks list
    tasks = {
        'super_glue_copa': 'COPA',
        'anli_r1': 'ANLI R1',
        'anli_r2': 'ANLI R2',
        'anli_r3': 'ANLI R3',
        'super_glue_cb': 'CB',
        'super_glue_rte': 'RTE',
        'super_glue_wsc.fixed': 'WSC',
        'winogrande_winogrande_xl': 'Winogrande',
        'super_glue_wic': 'WiC',
        'hellaswag': 'HellaSwag',
        'story_cloze_2016': 'StoryCloze',
    }

    # Plot results
    fig, axs = plt.subplots(2, 6, figsize=(20, 8))
    axs = axs.flatten()
    t5x_experiments = list(t5x_data.keys()) # defined single ordering
    for n, (task, name) in enumerate(tasks.items()):
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
        t5x_scores = [[s["accuracy"] for k, s in t5x_data[name].items() if task.replace("anli_", "") in k]
                      for name in t5x_experiments]
        for i, scores in enumerate([t5lm_scores, t0_scores, *t5x_scores]):
            axs[n].scatter([i] * len(scores), scores, s=200, alpha=0.2)
        axs[n].set_title(name)
    axs[10].legend(["T5+LM", "T0", *t5x_experiments], bbox_to_anchor=(1, 1), loc="upper left")
    axs[11].set_visible(False)
    # plt.plot()
    plt.show()
    print("Finished")

if __name__ == "__main__":
    main()