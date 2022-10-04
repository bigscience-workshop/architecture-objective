from t5x import utils
import seqio

from ..gins import task

def main():
    ranking_task = "super_glue_cb_GPT_3_style_score_eval"
    assert ranking_task.endswith("score_eval")
    ds = utils.get_dataset(
        utils.DatasetConfig(
            ranking_task,
            task_feature_lengths={"inputs": 1024, "targets": 256},
            split="validation",
            batch_size=1,
            shuffle=False,
            seed=None,
            use_cached=True,
            pack=True,
            use_custom_packing_ops=False,
            use_memory_cache=False,
        ),
        0,
        1,
        seqio.PassThroughFeatureConverter
    )

    iter_ds = iter(ds)
    while True:
        sample = next(iter_ds)
        if sample["is_correct"]:
            print("")
            print(sample["inputs_pretokenized"])
            print(f"Expected label: {sample['targets_pretokenized']}")
            input()

if __name__ == "__main__":
    main()