import functools
from typing import Sequence, Optional

import datasets
import seqio
from seqio import TaskRegistry, ShardInfo
from t5.data import preprocessors, get_default_vocabulary
import tensorflow as tf


VOCABULARY = get_default_vocabulary()

class HuggingFaceDatasetsSource(seqio.DataSource):
    def __init__(
        self,
        dataset_name: str,
        subset_name: str,
        num_shards: int,
        caching_permitted: bool = True
    ):
        """HuggingFaceDatasetsSource constructor.
        Args:
          dataset_name: HF dataset name.
          subset_name: HF dataset subset.
          num_shards: The number of shards, this is useful when processing large files in parallel.
          caching_permitted: indicates whether this data source may be cached.
            Default True.
        """
        self._dataset_fn = dataset_name
        self._subset_name = subset_name
        self._num_shards = num_shards

        # Get dataset information
        info = datasets.get_dataset_infos(dataset_name)
        subset_name = subset_name
        splits = list(info[subset_name].splits.keys())
        num_input_examples = {split_name: split_info.num_examples for split_name, split_info in info[subset_name].splits.items()}
        self._columns = list(info[subset_name].features.keys())

        super().__init__(
            splits=splits,
            num_input_examples=num_input_examples,
            caching_permitted=caching_permitted)

    @property
    def supports_arbitrary_sharding(self) -> bool:
        return False

    def get_dataset(
        self,
        split: str,
        shuffle: bool = True,
        seed: Optional[int] = None,
        shard_info: Optional[ShardInfo] = None
    ) -> tf.data.Dataset:
        dataset = datasets.load_dataset(
            self._dataset_fn,
            self._subset_name,
            split=split,
        )
        dataset = dataset.shard(num_shards=shard_info.num_shards, index=shard_info.index)
        if shuffle:
            dataset = dataset.shuffle(seed)
        return dataset.to_tf_dataset(
            columns=self._columns,
            batch_size=1,
            shuffle=False
        )

    def list_shards(self, split: str) -> Sequence[str]:
        return [str(i) for i in range(self._num_shards)]

TaskRegistry.add(
    "oscar_fr_lm_objective",
    source=HuggingFaceDatasetsSource(
        "oscar",
        "unshuffled_deduplicated_fr",
        num_shards=1024
    ),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.targets_for_prefix_lm_objective,
        preprocessors.pack_prefix_lm_encoder_decoder,
    ],
    output_features={
        "encoder_input_tokens": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "decoder_target_tokens": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "decoder_input_tokens": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "encoder_segment_ids": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "encoder_positions": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "decoder_segment_ids": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "decoder_positions": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        "decoder_loss_weights": seqio.Feature(vocabulary=VOCABULARY, add_eos=False),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=VOCABULARY, add_eos=True),
    },
    metric_fns=[]
)